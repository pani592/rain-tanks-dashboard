from datetime import datetime, timedelta
import pandas as pd
from numpy import interp, roll

from plotting import *
from policies import *

# Helper Functions
def daterange(start_date, end_date, self):
    '''Creates the timesteps according to the demand intervals'''

    for n in range(int((end_date - start_date).days*24*(60/self.Tank.demand_interval))):
        yield start_date + timedelta(minutes=n*self.Tank.demand_interval)


# CLASS DEFINITIONS
class Tank(object):
    def __init__(self, capacity, init_volume, area, harvest_ratio):
        ''' Initiate Tank and Catchment System
            Parameters:
            -----------
            capacity (float): volume of the tank in litres
            init_volume (float): initial volume of the tank in litres
            area (float): area of the catchment in meters squared.
            harvest_ratio (float): "amount captured" between 0-1. e.g. 0.9 means 90% of rainfall is captured.
        '''
        # Tank Parameters
        self.capacity = capacity
        self.volume = init_volume

        # Catchment Parameters
        self.area = area
        self.harvest_ratio = harvest_ratio

    def set_demand(self, interval, occupancy):
        ''' Set demand parameters for the model
            Parameters:
            -----------
            interval (float): demand interval of interest, in mins. Is also the smallest time step of the model.
                            (Currently can only be 60 (1 hr) or 1440 (24 hr))
            occupancy (float): number of occupants in dwelling.
        '''
        # Daily Demand Factors
        if interval == 60:
            self.demand_day = pd.read_csv(f"data/transformed_data/demand_daily_{interval}min.txt", sep="\t", engine='python')
            self.demand_day['TimeOfDay'] = pd.DatetimeIndex(pd.to_datetime(self.demand_day['TimeOfDay'])).time # Convert to DT
        else:
            self.demand_day = None

        # Monthly Demand Factors
        self.demand_month = pd.read_csv('data/transformed_data/month_fac.txt', sep=", ", engine='python')

        # Other Demand Factors
        self.occupancy = occupancy
        self.demand_interval = interval  # either 60 or 1440

        # 195 l/p/day --> interval (mins) * l/p/min.
        self.base_demand = self.demand_interval * 195/24/60


class Simulation(object):
    def __init__(self, Tank, start_date, end_date):
        ''' Initilze simulation object.
            Parameters:
            -----------
            Tank: Tank object that has been created.
            start_date: start date of simulation
            end_date: end date of simulation

            Notes:
            ------
            Dates are in the format %m/%d/%Y
        '''
        # Convert dates to datetime objects.
        format = '%m/%d/%Y' 
        self.Tank = Tank
        self.start_date = datetime.strptime(start_date, format)
        self.end_date = datetime.strptime(end_date, format)

        # Object to record statistics
        self.data_records = pd.DataFrame(columns=['DateAndTime', 'OverFlow',
                                                  'PolicyFlow', 'Volume',
                                                  'Demand', 'Rainfall_litres',
                                                  'TankDemand', 'MainsDemand'])

        # Rainfall data
        if self.Tank.demand_interval == 60:
            # Rainfall within date range of interest (only non zero values)
            rain = pd.read_pickle("data/transformed_data/1b_Whenuapai_rf_nozeros.pkl").reset_index(drop=True)
            self.rain_interval = 1  # Hourly
        elif self.Tank.demand_interval == 1440:
            rain = pd.read_pickle("data/transformed_data/1e_Whenuapai_rf_DAILY_nozeros.pkl").reset_index(drop=True)
            self.rain_interval = 1  # Daily

        # Read in incorrect forecast, for possible use.
        # forecast = pd.read_csv('data/raw_data/forecast_event_2or0.5.csv')

        # Index events for specified peroid
        idx = (rain['DateAndTime'] > datetime.strptime(start_date, format)) & (rain['DateAndTime'] < datetime.strptime(end_date, format))
        self.rainfall_events = rain.loc[idx, :].reset_index(drop=True)
        # self.forecast_events = forecast.loc[idx, :].reset_index(drop=True)
    
    def set_policy(self, policy=None, rLevel=1, use_forecast=False, summer_policy = None, attributes={}):
        '''
            policy: Name of the policy to use. Can be:
                    'RobustSmartTank','DetentionTank','SeasonalTank',None.
            rLevel: Robustness level to use on forecast. E.g. 2 is assuming rain event is twice that of forecast.
            use_forecast: Use fake forecast for robust smart tank.
            Attributes (specific to policy)
        '''
        self.policy = policy
        self.summer_policy = summer_policy
        self.robustness = rLevel
        self.use_forecast = use_forecast
        self.attributes = attributes

        # Setup for SeasonalTank only needed to be done at initialisation
        if self.policy == 'SeasonalTank':
            # Monthly Interpolation - all Sized Tanks
            av=[]
            bv=[]
            for i in range(12):
               av.append(self.attributes['a_vals'][i])
               bv.append(self.attributes['b_vals'][i])

            sd = [15,45,75,105,135,165,195,225,255,285,315,345] 
            x = list(range(sd[0], sd[0]+366)) # daily interpolation
            a_days = roll(interp(x, sd + [sd[0]+366], av+[av[-1]]), sd[0]-1)
            b_days = roll(interp(x, sd + [sd[0]+366], bv+[bv[-1]]), sd[0]-1)
            self.a_days = a_days
            self.b_days = b_days

            # self.maxuse = self.attributes['maxuse']

        return

    def run_simulation(self):
        ''' Run the Simulation Model '''

        print("Beginning Simulation.")
        count = 0

        for event in daterange(self.start_date, self.end_date, self):
            self.step_sim(count, event)
            count += 1

        print("Simulation Complete.")

    def step_sim(self, count, event):
        ''' Perform one time step of simulation. Works out rain in timestep,
        controlled outlet, demand splits, overflows '''
        rainfall_litres = 0
        self.data_records.loc[count, 'MainsDemand'] = 0

        # Rainfall at current time step (mm * m^2 --> 0.001 m^3 = 1L)
        if event in self.rainfall_events.values:
            rainfall_litres = self.rainfall_events[self.rainfall_events["DateAndTime"] == event]["Rainfall_mm"].iat[0] \
                    * self.rain_interval * self.Tank.area * self.Tank.harvest_ratio
            self.Tank.volume += rainfall_litres

        # Calculate Household Demand at timestep
        if self.Tank.demand_interval == 60:
            demand = self.Tank.demand_day.loc[self.Tank.demand_day.TimeOfDay == event.time(), 'DemandFactor'].values[0] \
                 * self.Tank.occupancy * self.Tank.base_demand \
                 * self.Tank.demand_month.loc[self.Tank.demand_month.Month == event.month, 'Factor'].values[0]
        elif self.Tank.demand_interval == 1440:
            demand = self.Tank.occupancy * self.Tank.base_demand \
                 * self.Tank.demand_month.loc[self.Tank.demand_month.Month == event.month, 'Factor'].values[0]

        # POLICY 1: Summer Policies to save water (water supply interests)
        prop_tank = self.demand_policy(demand, count, event)
        from_tank = demand * prop_tank
        if from_tank <= self.Tank.volume:
            # Demand can be met by tank.
            self.Tank.volume -= from_tank
            self.data_records.loc[count, 'TankDemand'] = from_tank
            self.data_records.loc[count, 'MainsDemand'] += (demand - from_tank)
        else:
            # Demand can not be met by tank, so mains water is used.
            self.data_records.loc[count, 'TankDemand'] = self.Tank.volume # uses remaining tank volume
            self.data_records.loc[count, 'MainsDemand'] += (demand - self.Tank.volume) # rest supplied by mains
            self.Tank.volume = 0

        # POLICY 2: Policies to release water earlier reduce outflowflow in storms (l/h) or (l/d)
        policy_spill = self.use_policy(event)
        self.Tank.volume -= policy_spill

        # Overflow Event - If Tank volume exceeded, need to spill excess
        # This might happen every hour, therefore liters/hour
        overflow = 0
        if self.Tank.volume > self.Tank.capacity:
            overflow = (self.Tank.volume - self.Tank.capacity)
            self.Tank.volume = self.Tank.capacity

        self.data_records.loc[count, ['DateAndTime', 'Volume', 'OverFlow', 'PolicyFlow', 'Rainfall_litres','Demand']] = [event, self.Tank.volume, overflow, policy_spill, rainfall_litres, demand]

    def demand_policy(self, demand, count, event):
        # Controls how tank is used to meet demand - policies that attempts to keep tanks full over summer and reduce peak demand.

        if self.Tank.capacity == 0:
            return 1
        else:
            tank_percent = self.Tank.volume / self.Tank.capacity

        if (self.policy not in ['SeasonalTank','XmasFill']):
            return 1

        elif self.policy == 'SeasonalTank':
       
            a = self.a_days[event.timetuple().tm_yday-1]
            b = self.b_days[event.timetuple().tm_yday-1]

            prop_tank = a*tank_percent + b
            # if prop_tank>3:
            #     print(f"a: {a}, b: {b}, tankp: {tank_percent}, maxuse: {self.maxuse}, prop_tank: {prop_tank}")
            return min(1,prop_tank)# min(self.maxuse[event.month - 1], prop_tank)

        if self.policy is 'XmasFill':
            day = event.timetuple().tm_yday
            if 358 <= day <= 365:  # between dec 24 and Dec 31
                # Historical period of LOW DEMAND in AUCKLAND - so fill tanks regardless of rain, to 100%
                fill = max(self.Tank.capacity - self.Tank.volume, 0)
                fill = fill/(24*(366- day)*60/self.Tank.demand_interval) # rate from mains to fill tank by Dec 31
                self.Tank.volume += fill
                self.data_records.loc[count, 'MainsDemand'] += fill # store that this was taken from Mains
            return 1

        else:
            print('Warning, check demand_policy implementation')

    def use_policy(self, event):
        '''Function that calls the relevant policy and passes back the # spill'''

        if self.policy in [None, 'SeasonalTank','XmasFill']:
            return 0

        if self.policy == 'DetentionTank':
            return DetentionTank(self, event)

        if self.policy == 'RobustSmartTank':
            return RobustSmartTank(self, event)


def data_verify(df):
    '''Checks that water in == water out.'''
    gap = (df.Volume.iloc[0] + df.TankDemand.iloc[0] + df.PolicyFlow.iloc[0]
           + df.Rainfall_litres.sum() - df.Volume.iloc[-1]
           - (df.TankDemand.sum() + df.OverFlow.sum() + df.PolicyFlow.sum()))
    if abs(gap) > 1E6:
        print(f"Water Gap: {round(gap,3)}")
    gap = (df.Demand.sum() - df.MainsDemand.sum() - df.TankDemand.sum())
    if abs(gap) > 1E6:
        print(f"Demand Gap: {round(gap, 3)}")
    # Check that volume is always >= 0.
    if all(df.Rainfall_litres.values >= 0):
        print('Good: No negative volume in tank')
    else:
        print('Bad: Negative volume in tank')

    # Could add a raise error statement (when running multiple sims in ipynb the outputs are not all shown so can miss potential errors)

    return


if __name__ == "__main__":
    # Use this for testing / debugging only. Repeated runs are called from main.py

    def repeat_sim(start_date, end_date, policy, attributes, interval):

        tank = Tank(capacity=15000, init_volume=10000, area=212, harvest_ratio=1)
        tank.set_demand(interval=interval, occupancy=3)
        sim = Simulation(tank, start_date=start_date, end_date=end_date)
        sim.set_policy(policy=policy, attributes=attributes)
        sim.run_simulation()
        # data_verify(sim.data_records)
        # quick_summary(sim, savefig = True, showfig = True)
        sim.data_records.to_pickle("data/temp_data/2022-08-10/"+savename+".pkl")
        del sim
        del tank

    # repeat_sim('5/1/2014', '5/1/2016', None, {}, 60)
    
    start = time.process_time()
    repeat_sim('5/1/2014', '8/1/2014', None, {}, 60)
    mid = time.process_time()
    repeat_sim('5/1/2015', '8/1/2015', None, {}, 60)
    stop = time.process_time()
    print(f'Mid {mid-start}, stop {stop-mid}')
    # df_filepaths = ['data/temp_data/none.pkl', 'data/temp_data/exp0.9.pkl']
    # compare_dfs(df_filepaths, plot=["Overflow", "Tank Volume"])



##################################################################################################################################
## Notes
# - Discrete Event used (as opposed to Adaptive Time Step with interpolation) as simulation times are reasonable.

# Tank Parameters from:
# -- Consumption of 220 l/p/day: Code of Practise for Land Development and Subdivision. (#008). Section 6.3.5.6 peak day design
# -- Avg Roof size of 212 m^2: https://aspectroofing.co.nz/useful_info/how-much-does-a-metal-roof-replacement-cost/
# -- Avg Occupancy of 2.7 : https://www.stats.govt.nz/news/new-data-shows-1-in-9-children-under-the-age-of-five-lives-in-a-multi-family-household
# -- Tank Size, taking middle option:  https://www.aucklandcouncil.govt.nz/environment/looking-after-aucklands-water/rainwater-tanks/Pages/rainwater-tank-size-calculator.aspx
# ----- All water -> 27.5k, outdoor activties -> 0.9k, indoor nondrink-> 3.7k, 6.4k for previous 2 combined.
# ---- using becas numbers -> single house 1 - 5 - 25 k. So lets go with 15 k..
