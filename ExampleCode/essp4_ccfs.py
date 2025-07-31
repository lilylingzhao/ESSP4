# Work With Standardized CCF File For ESSP4
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fitsfrom astropy.io import fits
from scipy.interpolate import interp1d
import seaborn as sns

# Specify file name
# Specify where all the data set folders (i.e. DS1) are, here saved into "essp_dir" variable
essp_dir = 
# Specify data set number
dset_num = 1
# Select a file at random from all files in the data set
file_list = glob(os.path.join(essp_dir,f'DS{dset_num}','CCFs',f'DS{dset_num}*.fits'))

# Select a file at random from all files in the data set as example
file_name = np.random.choice(file_list)

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

# =============================================================================
# Shift CCFs by Provided Offsets

def shiftCCF(ccf_x,ccf_y,rv):
    return interp1d(ccf_x-rv/1000,ccf_y,kind='cubic',bounds_error=False)(ccf_x)

# Read in offsets, which should be subtracted
offset_file = 'instrument_offsets_iccf.csv'
offset_dict = dict(zip(*np.loadtxt(offset_file),
                       delimiter=',',unpack=True,dtype=str)))
offset_dict = {key:float(val) for key,val in offset_dict.items()}

# Plot Original CCFs and Shifted CCFs
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,3))
ax1.set_title('CCFs')
ax2.set_title('Shifted CCFs')
for ax in [ax1,ax2]:
    ax.set_xlabel('Velocity [km/s]')
    ax.set_ylabel('Normalized Counts')
    
for iinst,inst in enumerate(['harpsn','harps','expres','neid']):
    # Select a random file
    file = np.random.choice(glob(os.path.join(essp_dir,f'DS{dset_num}','CCFs',f'DS{dset_num}*_{inst}.fits')))
    
    # Read in data
    hdus = fits.open(file)
    v_grid = hdus['v_grid'].data.copy() # velocity grid for all CCFs in the file
    ccf = hdus['ccf'].data.copy()
    hdus.close()
    
    # Plot Summed/Average CCF
    ax1.plot(v_grid,ccf/np.nanmax(ccf),alpha=0.5)
    # Plot Shifted CCF
    ax2.plot(v_grid,shiftCCF(v_grid,ccf/np.nanmax(ccf),offset_dict[inst]),alpha=0.5)
fig.tight_layout()