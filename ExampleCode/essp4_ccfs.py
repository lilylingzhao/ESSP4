# Work With Standardized CCF File For ESSP4
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import seaborn as sns

# Specify file name
# Specify where all the spectra are saved
ccfs_dir = 
# For example, to return all files in a data set:
file_list = glob(os.path.join(ccfs_dir,'*.fits'))

# Select a file at random from all files in the data set as example
file = np.random.choice(file_list)

# =============================================================================
# Open a FITS File

hdus = fits.open(file)
hdus.info() # to easily see HDU names and shapes
# Header information contained in the header of the primary HDU
hdus[0].header

# =============================================================================
# Plot CCF and Order-by-Order CCFs

fig, axes = plt.subplots(1,4,figsize=(12,3))
for iinst,inst in enumerate(['harpsn','harps','expres','neid']):
    # Set up a subplot for each instrument
    ax = axes[iinst]
    ax.set_title(inst)
    ax.set_xlabel('Velocity [km/s]')
    ax.set_ylabel('Normalized Counts')
    # Select a random file for each instrument
    file = np.random.choice(glob(os.path.join(ccfs_dir,f'*_{inst}.fits')))
    
    # Read in data
    hdus = fits.open(file)
    num_ord = len(hdus['echelle_orders'].data) # number of orders
    colors = sns.color_palette('Spectral',num_ord) # use num_ord to define a color map
    v_grid = hdus['v_grid'].data.copy() # velocity grid for all CCFs in the file
    ccf = hdus['ccf'].data.copy() # global CCF
    obo_ccf = hdus['obo_ccf'].data.copy() # order-by-order CCFs
    hdus.close()
    
    # Plot Order-by-Order CCFs
    for nord in range(num_ord):
        if np.sum(np.isfinite(obo_ccf[nord]))==0:
            # Skip CCFs with only NaN values
            continue
        ax.plot(v_grid,obo_ccf[nord]/np.nanmax(obo_ccf[nord]),color=colors[nord])
    # Plot global CCF
    ax.plot(v_grid,ccf/np.nanmax(ccf),color='k',lw=3)
fig.tight_layout()

# =============================================================================
# Resample CCF
# See https://essp-eprv.github.io/data.html#velocity_sampling for more details

fig, axes = plt.subplots(1,4,figsize=(12,3))
for iinst,inst in enumerate(['harpsn','harps','expres','neid']):
    # Set up a subplot for each instrument
    ax = axes[iinst]
    ax.set_title(inst)
    ax.set_xlabel('Velocity [km/s]')
    ax.set_ylabel('Normalized Counts')
    # Select a random file
    file = np.random.choice(glob(os.path.join(ccfs_dir,f'*_{inst}.fits')))
    
    # Read in data
    ccf_dict = resampleCCF(file)
    num_ord = len(ccf_dict['echelle_orders']) # number of orders
    colors = sns.color_palette('Spectral',num_ord) # use num_ord to define a color map
    v_grid = ccf_dict['v_grid'] # velocity grid for all CCFs in the file
    ccf = ccf_dict['ccf'] # global CCF
    obo_ccf = ccf_dict['obo_ccf'] # order-by-order CCFs
    
    # Plot Order-by-Order CCFs
    for nord in range(num_ord):
        if np.sum(np.isfinite(obo_ccf[nord]))==0:
            # Skip CCFs with only NaN values
            continue
        ax.plot(v_grid,obo_ccf[nord]/np.nanmax(obo_ccf[nord]),color=colors[nord])
    # Plot global CCF
    ax.plot(v_grid,ccf/np.nanmax(ccf),color='k',lw=3)
fig.tight_layout()