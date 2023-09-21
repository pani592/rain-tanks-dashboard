# streamlit run streamlit_app.py
# https://blog.streamlit.io/make-your-st-pyplot-interactive/

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import seaborn as sns
sns.set()
sns.set(rc={'figure.figsize':(10,8)})

from model import *

xopts_final = {}
xopts_final[1] = [0.265, 0.216, 0.109, 0.117, 0.1, 0.186, 0.128, 0.292, 0.223, 0.5, 0.324, 0.33, 0.726, 0.0, 0.191, 0.663, 0.081, 0.887, 1.0, 0.462, 0.192, 0.184, 0.0, 0.007] #10308
xopts_final[5] = [0.252, 0.19, 0.06, 0.325, 0.328, 0.228, 0.175, 0.021, 0.214, 0.0, 0.184, 0.113, 0.0, 0.076, 0.424, 0.456, 0.395, 0.716, 0.849, 0.889, 0.962, 0.526, 0.498, 0.12] #5118
xopts_final[10] = [0.262, 0.001, 0.239, 0.086, 0.245, 0.3, 0.126, 0.34, 0.145, 0.15, 0.067, 0.003, 0.027, 0.292, 0.373, 0.581, 0.748, 0.566, 0.814, 1.0, 0.887, 0.78, 0.816, 0.295] #3225
xopts_final[15] = [0.418, 0.284, 0.015, 0.222, 0.397, 0.209, 0.5, 0.108, 0.334, 0.053, 0.243, 0.049, 0.035, 0.327, 0.409, 0.59, 0.722, 0.69, 0.378, 0.94, 0.439, 0.533, 0.872, 0.382] #2399
xopts_final[20] = [0.181, 0.015, 0.2, 0.305, 0.213, 0.483, 0.249, 0.217, 0.278, 0.155, 0.356, 0.212, 0.29, 0.438, 0.452, 0.567, 0.683, 0.697, 0.717, 0.803, 0.516, 0.852, 0.525, 0.382] #1800
xopts_final[25] = [0.135, 0.437, 0.064, 0.056, 0.186, 0.107, 0.211, 0.198, 0.336, 0.131, 0.059, 0.195, 0.407, 0.422, 0.56, 0.53, 0.963, 0.9, 0.937, 0.752, 0.997, 0.706, 0.743, 0.432] #1362


def plotfig(sim):
    # Can either pass in simulation output itself, or just datarecords + timestep
    dtr = sim.data_records
    if sim.Tank.demand_interval == 60:
        timestep = 60
        n = 24
        label_use = '[L/Hr]'
    else:
        timestep = 1440
        n = 1
        label_use = '[L/Day]'

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
    #############################################################
    # Rainfall Inverted
    ax_inv = axs[0].twinx()
    ax_inv.invert_yaxis()
    ax_inv.plot(dtr.DateAndTime, dtr.Rainfall_litres, 'b', alpha=0.3, label='Rainfall On Roof '+label_use)
    ax_inv.set_ylabel('Rainfall '+label_use, size=14)
    ax_inv.legend(loc='lower right',  bbox_to_anchor= (1,-.3))
    ax_inv.grid(False)

    # Tank Volume
    axs[0].plot(dtr.DateAndTime, dtr.Volume/1000, 'k', label='Tank Volume [kL]') # can do tank level %
    axs[0].set_ylabel('Tank Volume [kL]', size=14)
    axs[0].set_ylim(ymin=0)
    axs[0].legend(loc='lower right',  bbox_to_anchor= (.25,-.3))
    axs[0].set_title('Tank Volume and Rainfall', size=14)

    #############################################################
    mains = pd.to_numeric(sliding_window_view(dtr.MainsDemand, 10).mean(axis=1))
    total = pd.to_numeric(sliding_window_view(dtr.Demand, 10).mean(axis=1))

    n = 10
    x = pd.to_datetime(dtr.DateAndTime[:-(n - 1)])
    axs[1].fill_between(x.values, mains, total,  color='g', alpha = 0.2) # fill between y1 and y0
    axs[1].fill_between(x.values, 0, mains,color='r', alpha = 0.2) # fill between y1 and y0
    axs[1].plot(x, total, 'g', label='Tank Demand')
    axs[1].plot(x, mains, 'r--', label='Mains Demand')
    axs[1].set_ylabel('Litres/day', size=14)
    axs[1].set_ylim(ymin=0)
    axs[1].legend(loc='lower right',  bbox_to_anchor= (1,-.3), ncol=2)
    axs[1].set_title('10 day Rolling Demand - Mains vs Tank', size=14)

    fig.supxlabel('Date')
    # fig.suptitle('HAT Simulation Results', size=16)
    fig.tight_layout()
    #############################################################
    plt.show()

    plt.close()

    totalmains = round(dtr.MainsDemand.sum())
    peakreduction = round((1-(mains.max()/total.max()))*100,2)

    return [fig,totalmains,peakreduction]




d_mode = st.sidebar.radio('Select Dashboard Mode', ("Standard Tank", "Drought Control"))

if d_mode == "Standard Tank":
    st.write(f"Select Parameters using the sidebar.")

    # Policy = st.sidebar.selectbox('Select Policy', (None, 'SeasonalTank'))
    Capacity = st.sidebar.selectbox('Select Tank Size [kL]', (25, 20, 15, 10, 5),index=2)
    Occupancy = st.sidebar.selectbox('Select Occupancy', (1,2,3,4,5),index=1)
    RoofArea = st.sidebar.selectbox('Select Roof Area', (100,150,200,250,300),index=1)
    StartYr = st.sidebar.selectbox('Select Start Yr (July)', range(2020,1970,-1))
    StopYr = st.sidebar.selectbox('Select End Yr (July)', range(StartYr+1,2022))
    st.write(f"Selected Parameters: Tank Capacity: {Capacity} kL | Occupancy: {Occupancy} | Roof Area: {RoofArea}m2 ")
    st.write(f"Default Parameters: Base Demand = 195 l/p/day")
    st.write("---")

    if st.sidebar.button('Run Simulation'):
        st.write('Simulation Running...')
        START_DATE = f"7/1/{StartYr}" 
        END_DATE = f"7/1/{StopYr}"
        Interval = 1440 # Has to be hourly timesteps for dashboard speed.

        tank = Tank(capacity=Capacity*1000, init_volume=0.8*Capacity*1000, area=RoofArea, harvest_ratio=1)
        tank.set_demand(Interval, Occupancy)
        sim = Simulation(tank, START_DATE, END_DATE)
        sim.set_policy(policy=None)
        sim.run_simulation()

        output = plotfig(sim) 
        st.write('Simulation Complete!')
       
        st.pyplot(output[0])
        st.write(f'Total Mains Usage: {output[1]} litres')
        st.write(f'Min Peak Reduction: {output[2]}%')
    else:
        # Placeholder Progress Bar
        st.write("Press 'Run Simulation' to view results")


if d_mode == "Drought Control":

    st.write('Example Results of Drought Control Policy')

    Capacity = st.sidebar.selectbox('Select Tank Size [kL]', (25, 20, 15, 10, 5),index=2)
    # Occupancy = st.sidebar.selectbox('Select Occupancy', (1,3,5),index=1)
    # RoofArea = st.sidebar.selectbox('Select Roof Area', (100,200,300),index=1)
    StartYr = st.sidebar.selectbox('Select Start Yr (July)', range(2020,1970,-1))
    StopYr = st.sidebar.selectbox('Select End Yr (July)', range(StartYr+1,2022))
    st.write(f"Selected Parameters: Tank Capacity: {Capacity} kL")
    st.write(f"Default Parameters: Occupancy: 3 | Roof Area: 200m2  Base Demand = 195 l/p/day")
    st.write("---")

    if st.sidebar.button('Run Simulation'):
        st.write('Simulation Running...')
        START_DATE = f"7/1/{StartYr}" 
        END_DATE = f"7/1/{StopYr}"
        Interval = 1440 # Has to be hourly timesteps for dashboard speed.

        att = {'a_vals': xopts_final[Capacity][0:12], 'b_vals': xopts_final[Capacity][12:24]}
        tank = Tank(capacity=Capacity*1000, init_volume=0.8*Capacity*1000, area=200, harvest_ratio=1)
        tank.set_demand(Interval, 3)
        sim = Simulation(tank, START_DATE, END_DATE)
        sim.set_policy(policy='SeasonalTank', attributes = att)
        sim.run_simulation()

        output = plotfig(sim) 
        st.write('Simulation Complete!')
       
        st.pyplot(output[0])
        st.write(f'Total Mains Usage: {output[1]} litres')
        st.write(f'Min Peak Reduction: {output[2]}%')

    else:
        # Placeholder Progress Bar
        st.write("Press 'Run Simulation' to view results")
