import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pickle
import time
import pandas as pd
import seaborn as sns
sns.set()
sns.set(rc={'figure.figsize':(10,7)})

def compare_dfs(df_filepaths, plot, showfig=True):
    '''
    Plots multiple simulation runs comparing specified aspects
    Parameters
    ----------
    df_filepaths: List of filenames, relative to the top folder, to plot. Would like
    to introduce a check for string vs list. If string then plot .pkl files in folder
    plot: Plots to produce. Can be "Tank Volume", "Outflow"..

    Note: Would like to introduce another input to specify labels. Must be for atleast 2 dfs.
    '''

    if "Overflow" in plot:
        # Plotting commands if Overflow is selected
        fig, axs = plt.subplots(nrows=len(df_filepaths), ncols=1, sharex=True,
                                sharey=True, figsize=(12, 16))
        # For each df
        for i, df_filepath in enumerate(df_filepaths):
            file = open(df_filepath, 'rb')
            df = pickle.load(file)  # load df from .pkl
            file.close()
            axs[i].plot(df.DateAndTime, (df.OverFlow + df.PolicyFlow)/60/60, 'r', label='Tank OverFlow [L/s]')
            axs[i].set_title(df_filepath)
            axs[i].set_ylabel('[Litres/Second]')

        if showfig: plt.show()

    if "Tank Volume" in plot:
        # Plotting commands if Tank Volume is selected.
        fig, axs = plt.subplots(nrows=len(df_filepaths), ncols=1, sharex=True,
                                sharey=True, figsize=(12, 16))
        for i, df_filepath in enumerate(df_filepaths):
            file = open(df_filepath, 'rb')
            df = pickle.load(file)  # load df from .pkl
            file.close()
            axs[i].plot(df.DateAndTime, df.Volume, 'k', label='Volume [L]')
            axs[i].set_title(df_filepath)
            axs[i].set_ylabel('Volume [L]')

        if showfig: plt.show()

    return

def quick_summary(sim = None, savefig = False, showfig = False, fig_title = None, savename = None, dtr = None, timestep = None):
    ''' Plotting function to make summary plot of a SINGLE model run
        Saves figures to figures/exploration/QuickSum_{timestamp}
        Returns fig
    '''

    # Can either pass in simulation output itself, or just datarecords + timestep
    if sim is not None:
        dtr = sim.data_records
        if sim.Tank.demand_interval == 60:
            timestep = 60
            n = 24
            label_use = '[L/Hr]'
        else:
            timestep = 1440
            n = 1
            label_use = '[L/Day]'
    elif dtr is not None:
        if timestep is None:
            print('If using datarecords instead of sim, please specify timestep')
            return
        elif timestep == 60:
            n = 24
            label_use = '[L/Hr]'
        elif timestep == 1440:
            n = 1
            label_use = '[L/Day]'


    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(12, 8))

    #############################################################
    # Rainfall Inverted
    ax_inv = axs[0].twinx()
    ax_inv.invert_yaxis()
    ax_inv.plot(dtr.DateAndTime, dtr.Rainfall_litres, 'b', alpha=0.2, label='Rainfall On Roof '+label_use)
    ax_inv.set_ylabel('Rainfall '+label_use)
    ax_inv.legend(loc = 1)

    # Tank Volume
    axs[0].plot(dtr.DateAndTime, dtr.Volume, 'k', label='Tank Volume [L]') # can do tank level %
    axs[0].set_ylabel('Litres')
    axs[0].set_ylim(ymin=0)
    axs[0].legend(loc = 2)
    axs[0].set_title('Tank Volume and Rainfall')

    #############################################################
    # Overflow
    axs[1].plot(dtr.DateAndTime, (dtr.OverFlow + dtr.PolicyFlow)/timestep/60, 'r')
    axs[1].set_ylabel('Litres/s')
    axs[1].set_ylim(ymin=0)
    axs[1].set_title('Tank Overflow [L/s]')

    #############################################################
    ### DAILY DEMAND MET
    # if timestep == 60:
    #     x = pd.to_datetime(dtr.DateAndTime[:-(n - 1)])
    # else:
    #     x = pd.to_datetime(dtr.DateAndTime)

    # y0 = pd.to_numeric(np.sum(sliding_window_view(dtr.Demand, n), axis=1))
    # y1 = pd.to_numeric(np.sum(sliding_window_view(dtr.MainsDemand, n), axis=1))
    # axs[2].fill_between(x.values, y1, y0,  color='g', alpha = 0.2) # fill between y1 and y0
    # axs[2].fill_between(x.values, 0, y1,color='r', alpha = 0.2) # fill between y1 and y0

    # axs[2].plot(x, y0, 'g', label='Total Demand')
    # axs[2].plot(x, y1, 'r', label='Mains Demand')
    # axs[2].set_ylabel('Litres/day')
    # axs[2].set_ylim(ymin=0)
    # axs[2].legend(loc = 1)
    # axs[2].set_title('Daily Demand - Mains vs Tank')

    # Suburb view. suburb = mains_10*(1-tank_uptake) + tdemand_10*tank_uptake
    n = 10
    tank_uptake = 0.3
    x = pd.to_datetime(dtr.DateAndTime[:-(n - 1)])
    mains_10 = sliding_window_view(dtr.MainsDemand, 10).sum(axis=1)
    tdemand_10 = sliding_window_view(dtr.Demand, 10).sum(axis=1)

    y0 = pd.to_numeric(np.sum(sliding_window_view(dtr.Demand, n), axis=1))
    y1 = pd.to_numeric(tdemand_10*(1-tank_uptake) + mains_10*tank_uptake)
    axs[2].fill_between(x.values, y1, y0,  color='g', alpha = 0.2) # fill between y1 and y0
    axs[2].fill_between(x.values, 0, y1,color='r', alpha = 0.2) # fill between y1 and y0

    axs[2].plot(x, y0, 'g', label='Total Subrubs Demand')
    axs[2].plot(x, y1, 'r', label='Suburbs Demand From Mains')
    axs[2].plot(x, np.sum(sliding_window_view(dtr.MainsDemand, n), axis=1), label='Tank Main Demand Profile')
    axs[2].set_ylabel('Litres/day')
    axs[2].set_ylim(ymin=0)
    axs[2].legend(loc = 1)
    axs[2].set_title('Daily Demand - Mains vs Tank')

    fig.supxlabel('Date')
    if fig_title is not None:
        fig.suptitle(fig_title)

    fig.tight_layout()

    #############################################################
    if savefig:
        if savename is None:
            plt.savefig("figures/exploration/QuickSum_" + time.strftime("%Y-%m-%d %H%M%S") + ".png", transparent=False)
        else:
            plt.savefig("figures/exploration/QuickSum_" + savename + ".png", transparent=False)
    if showfig:
        plt.show()

    plt.close()

    return fig


from model import *
def quick_summary_report(sim = None, dtr = None, timestep = None):
    # Can either pass in simulation output itself, or just datarecords + timestep
    if sim is not None:
        dtr = sim.data_records
        if sim.Tank.demand_interval == 60:
            timestep = 60
            n = 24
            label_use = '[L/Hr]'
        else:
            timestep = 1440
            n = 1
            label_use = '[L/Day]'
    elif dtr is not None:
        if timestep is None:
            print('If using datarecords instead of sim, please specify timestep')
            return
        elif timestep == 60:
            n = 24
            label_use = '[L/Hr]'
        elif timestep == 1440:
            n = 1
            label_use = '[L/Day]'

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False)
    #############################################################
    # Rainfall Inverted
    ax_inv = axs[0].twinx()
    ax_inv.invert_yaxis()
    ax_inv.plot(dtr.DateAndTime, dtr.Rainfall_litres, 'b', alpha=0.3, label='Rainfall On Roof '+label_use)
    ax_inv.set_ylabel('Rainfall '+label_use)
    ax_inv.legend(loc = 1)

    # Tank Volume
    axs[0].plot(dtr.DateAndTime, dtr.Volume/1000, 'k', label='Tank Volume [kL]') # can do tank level %
    axs[0].set_ylabel('Tank Volume [kL]')
    axs[0].set_ylim(ymin=0)
    axs[0].legend(loc = 2)
    axs[0].set_title('Tank Volume and Rainfall', size=12)

    #############################################################
    # Overflow
    axs[1].plot(dtr.DateAndTime, (dtr.OverFlow + dtr.PolicyFlow)/timestep/60, 'r')
    axs[1].set_ylabel('Overflow [L/s]')
    axs[1].set_ylim(ymin=0)
    axs[1].set_title('Overflow from Tank', size=12)

    #############################################################
    ### DAILY DEMAND MET
    if timestep == 60:
        x = pd.to_datetime(dtr.DateAndTime[:-(n - 1)])
    else:
        x = pd.to_datetime(dtr.DateAndTime)

    y0 = pd.to_numeric(np.sum(sliding_window_view(dtr.Demand, n), axis=1))
    y1 = pd.to_numeric(np.sum(sliding_window_view(dtr.MainsDemand, n), axis=1))
    axs[2].fill_between(x.values, y1, y0,  color='g', alpha = 0.2) # fill between y1 and y0
    axs[2].fill_between(x.values, 0, y1,color='r', alpha = 0.2) # fill between y1 and y0

    axs[2].plot(x, y0, 'g', label='Total Demand')
    axs[2].plot(x, y1, 'r', label='Mains Demand')
    axs[2].set_ylabel('Demand [L/d]')
    axs[2].set_ylim(ymin=0)
    axs[2].legend(loc = 1)
    axs[2].set_title('Daily Household Demand Met', size=12)

    # fig.supxlabel('Date')
    # fig.suptitle('HAT Simulation Results', size=16)
    fig.tight_layout()
    #############################################################
    plt.show()

    plt.close()

    return fig