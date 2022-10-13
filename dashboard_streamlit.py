# streamlit run dashboard_streamlit.py
# https://blog.streamlit.io/make-your-st-pyplot-interactive/

# https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app
# https://docs.streamlit.io/streamlit-cloud/get-started/share-your-app#adding-viewers-from-the-app

# Would be cool to add the final figure plots to this as well for fun


import streamlit as st
import pandas as pd
import pickle as pk

from plotting import *
from model import *

xopts_final = {}
xopts_final[1] = [0.265, 0.216, 0.109, 0.117, 0.1, 0.186, 0.128, 0.292, 0.223, 0.5, 0.324, 0.33, 0.726, 0.0, 0.191, 0.663, 0.081, 0.887, 1.0, 0.462, 0.192, 0.184, 0.0, 0.007] #10308
xopts_final[5] = [0.252, 0.19, 0.06, 0.325, 0.328, 0.228, 0.175, 0.021, 0.214, 0.0, 0.184, 0.113, 0.0, 0.076, 0.424, 0.456, 0.395, 0.716, 0.849, 0.889, 0.962, 0.526, 0.498, 0.12] #5118
xopts_final[10] = [0.262, 0.001, 0.239, 0.086, 0.245, 0.3, 0.126, 0.34, 0.145, 0.15, 0.067, 0.003, 0.027, 0.292, 0.373, 0.581, 0.748, 0.566, 0.814, 1.0, 0.887, 0.78, 0.816, 0.295] #3225
xopts_final[15] = [0.418, 0.284, 0.015, 0.222, 0.397, 0.209, 0.5, 0.108, 0.334, 0.053, 0.243, 0.049, 0.035, 0.327, 0.409, 0.59, 0.722, 0.69, 0.378, 0.94, 0.439, 0.533, 0.872, 0.382] #2399
xopts_final[20] = [0.181, 0.015, 0.2, 0.305, 0.213, 0.483, 0.249, 0.217, 0.278, 0.155, 0.356, 0.212, 0.29, 0.438, 0.452, 0.567, 0.683, 0.697, 0.717, 0.803, 0.516, 0.852, 0.525, 0.382] #1800
xopts_final[25] = [0.135, 0.437, 0.064, 0.056, 0.186, 0.107, 0.211, 0.198, 0.336, 0.131, 0.059, 0.195, 0.407, 0.422, 0.56, 0.53, 0.963, 0.9, 0.937, 0.752, 0.997, 0.706, 0.743, 0.432] #1362

d_mode = st.sidebar.radio('Select Dashboard Mode', ("Run Sim", "Show Results"))

if d_mode == "Run Sim":
    st.write(f"Select Parameters using the sidebar.")

    Policy = st.sidebar.selectbox('Select Policy', (None, 'SeasonalTank'))
    Capacity = st.sidebar.selectbox('Select Tank Size [kL]', (25, 20, 15, 10, 5),index=2)
    Occupancy = st.sidebar.selectbox('Select Occupancy', (1,2,3,4,5),index=2)
    RoofArea = st.sidebar.selectbox('Select Roof Area', (100,150,200,250,300),index=2)
    StartYr = st.sidebar.selectbox('Select Start Yr (July)', range(2020,1970,-1))
    StopYr = st.sidebar.selectbox('Select End Yr (July)', range(StartYr+1,2022))
    st.write(f"Selected: | {Policy} | Tank Size: {Capacity} kL")
    st.write(f"Default Parameters: | Roof Area: {RoofArea}m2 | Base Demand = 195 l/p/day")
    st.write("---")

    if st.sidebar.button('Run Simulation'):
        st.write('Simulation Running...')
        START_DATE = f"7/1/{StartYr}" 
        END_DATE = f"7/1/{StopYr}"
        Interval = 1440 # Has to be hourly timesteps for dashboard speed.

        att = {'a_vals': xopts_final[Capacity][0:12], 'b_vals': xopts_final[Capacity][12:24]}
        tank = Tank(capacity=Capacity*1000, init_volume=0.8*Capacity*1000, area=RoofArea, harvest_ratio=1)
        tank.set_demand(Interval, Occupancy)
        sim = Simulation(tank, START_DATE, END_DATE)
        sim.set_policy(policy=Policy, attributes=att)
        sim.run_simulation()

        fig = quick_summary_report(dtr = sim.data_records, timestep = 1440) 
        st.write('Simulation Complete!')
       
        st.pyplot(fig)
    else:
        # Placeholder Progress Bar
        st.write("Press 'Run Simultation' to view results")


if d_mode == "Show Results":

    st.write('Example Results')

    @st.cache() 
    def read_data(pickle_path):
        file = open(pickle_path, 'rb')
        dtr = pk.load(file)
        file.close()

        return dtr

    pickle_path = 'data/temp_data/2022-09-02/SummerTank1.pkl'
    dtr = read_data(pickle_path)

    fig = quick_summary_report(dtr = dtr, timestep = 1440)
    st.pyplot(fig)


######################################################
# Notes
# Show key stats as well - total mains demand, total peak mains demand, % improvement in month etc.


# The largest manufactures of plastic rainwater tanks in New Zealand generally supply them in the following sizes: 
# 1,000L, 2,000L, 3,000L/3,500L, 4,000L, 5,000L 9,000L, 10,000L, 13,500L, 15,000L 25,000L, 30,000L