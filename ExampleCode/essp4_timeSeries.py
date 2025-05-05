# Work With Time Series File For ESSP4
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Specify file name
example_file = 

# =============================================================================
#  Read in with `pandas`

df = pd.read_csv(example_file)
# Show a snippet of the beginning of the table
df.head()
# Show all the column names
df.columns

# =============================================================================
# Plot RVs and Errors

plt.figure(figsize=(8,3))
plt.xlabel('Time [eMJD]')
plt.ylabel('RV [m/s]')
# Get a list of different instruments and their indices
inst_list, inst_invs = np.unique(df['Instrument'],return_inverse=True)
# Plot RVs colored by instrument
plt.scatter(df['Time [eMJD]'],df['RV [m/s]'],
            c=inst_invs, # color by instrument
            cmap='tab10',vmin=0,vmax=10) # adjusting so familiar python colors are used
# Generate Legend
for inst in inst_list:
    plt.plot(np.nan,'o',label=inst)
# Plot error bars
plt.errorbar(df['Time [eMJD]'],df['RV [m/s]'],yerr=df['RV Err. [m/s]'],
             linestyle='None',color='k')

plt.legend(loc=2,ncol=4)

# =============================================================================
# Plot Indicators

# Function to generate masks that isolate observations from each instrument
def getInstrumentMask(df):
    inst_col = df['Instrument'].to_numpy()
    inst_list = np.unique(inst_col)
    inst_masks = {}
    for inst in inst_list:
        inst_masks[inst] = inst_col==inst
    return inst_masks

# List of indicators to plot
ind_names = ['CCF FWHM [km/s]', 'CCF Contrast', 'BIS [m/s]',
             'H-alpha Emission', 'CaII Emission']
# Generate masks to isolate observations from each instrument
inst_masks = getInstrumentMask(df)

fig, axes = plt.subplots(len(ind_names),1,figsize=(8,len(ind_names)*2.5))
for iind, ind in enumerate(ind_names):
    # Set up a subplot for each indicator
    ax = axes[iind]
    ax.set_title(ind)
    ax.set_xlabel('Time [eMJD]')
    ax.set_ylabel(ind)
    # Plot indicators from each instrument
    for inst in inst_masks.keys():
        imask = inst_masks[inst]
        # Find a unique offset for the indicator for each instrument
        ind_offset = np.nanmedian(df[ind][imask])
        ax.plot(df['Time [eMJD]'][imask],df[ind][imask]-ind_offset,'.',label=inst)
axes[0].legend(ncol=4)
fig.tight_layout()

# =============================================================================
# Rename Columns

# To rename columns, specify a dictionary like the following
#     The key should be the original column name; the value should be the new name
#     Below are just some example new column names
col_dict = {
    'Standard File Name' : 'file',
    'Time [eMJD]' : 'time',
    'RV [m/s]' : 'rv',
    'RV Err. [m/s]' : 'e_rv',
    'Exp. Time [s]' : 'exptime',
    'Airmass' : 'airmass',
    'BERV [km/s]' : 'berv',
    'Instrument' : 'inst',
    'CCF FWHM [km/s]' : 'fwhm',
    'CCF FWHM Err. [km/s]' : 'e_fwhm',
    'CCF Contrast' : 'contrast',
    'CCF Contrast Err.' : 'e_contrast',
    'BIS [m/s]' : 'bis',
    'H-alpha Emission' : 'ha',
    'CaII Emission' : 'caii'
}

df = pd.read_csv(example_file) # read in data
renamed_df = df.rename(columns=col_dict) # rename the columns
renamed_df.head() # show snippet of the beginning of the renamed table

# =============================================================================
# Read into Dictionary (instead of pandas DataFrame)

df = pd.read_csv(example_file) # read in data
data_dict = df.to_dict('list') # change to a dictionary of lists

# Note: each time series will be turned into a `list`, not a numpy array

# The keys for the new dictionary are the column names
# For example, to plot the RVs
plt.figure(figsize=(8,3))
plt.xlabel('Time [eMJD]')
plt.ylabel('RV [m/s]')
plt.errorbar(data_dict['Time [eMJD]'],data_dict['RV [m/s]'],yerr=data_dict['RV Err. [m/s]'],
             linestyle='None',marker='o',color='k')
