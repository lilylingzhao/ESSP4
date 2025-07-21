# Work With Standardized Spectral File For ESSP4
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Specify file name
# Specify where all the data set folders (i.e. DS1) are, here saved into "essp_dir" variable
essp_dir = 
# Specify data set number
dset_num = 1
# Select a file at random from all files in the data set
file_list = glob(os.path.join(essp_dir,f'DS{dset_num}','Spectra',f'DS{dset_num}*.fits'))
# For example, to return all files in a data set:
file_list = glob(os.path.join(spec_dir,'*.fits'))

# Select a file at random from all files in the data set as example
file = np.random.choice(file_list)

# =============================================================================
# Open a FITS File

hdus = fits.open(file)
hdus.info() # to easily see HDU names and shapes
# Header information contained in the header of the primary HDU
hdus[0].header

# =============================================================================
# Plot Continuum Normalized Spectra

# Plot Echelle Order 93 (i.e. Standard Relative Order 161-93=68)
nord = 161-93

plt.figure(figsize=(12,3))
plt.xlabel('Wavelength [A]')
plt.ylabel('Normalized Counts')
for inst in ['harpsn','harps','expres','neid']:
    # Select a random file for each instrument
    file = np.random.choice(glob(os.path.join(spec_dir,f'*_{inst}.fits')))
    hdus = fits.open(file)
    wave = hdus['wavelength'].data.copy()
    spec = hdus['flux'].data.copy()
    cont = hdus['continuum'].data.copy()
    hdus.close()
    
    plt.plot(wave[nord],spec[nord]/cont[nord],label=inst)
    hdus.close()
plt.legend(loc=4,ncol=4)

# =============================================================================
# Remove Orders With Only NaNs

def getNanMask(flux):
    return np.sum(np.isfinite(flux),axis=1)>0

# The below loop prints the shape of the resultant data array
#     before/after applying the NaN mask
for inst in ['harpsn','harps','expres','neid']:
    # Select a random file for each instrument
    file = np.random.choice(glob(os.path.join(spec_dir,f'*_{inst}.fits')))
    hdus = fits.open(file)
    # Read in the original flux values
    og_flux = hdus['flux'].data
    # Get mask and apply to og_flux array
    flux = og_flux[getNanMask(og_flux)]
    # Compare the shape of the original and masked flux arrays
    print(inst)
    print('Original Array Shape: ',og_flux.shape)
    print('Masked Array Shape: ',flux.shape)
    print('--------------------------------')
    hdus.close()

# =============================================================================
# Uncertainties
# The uncertainty, which is reported slightly differently by each instrument's DRP,
#     is approximately the square root of the original counts of each spectrum.

# Plot Echelle Order 93 (i.e. Standard Relative Order 161-93=68)
nord = 161-93

# Plot the spectra colored by their uncertainties
fig, axes = plt.subplots(4,1,figsize=(12,12))
for iinst,inst in enumerate(['harpsn','harps','expres','neid']):
    # Set up a subplot for each instrument
    ax = axes[iinst]
    ax.set_title(inst)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Counts')
    # Select a random file for each instrument
    file = np.random.choice(glob(os.path.join(spec_dir,f'*_{inst}.fits')))
    hdus = fits.open(file)
    wave = hdus['wavelength'].data.copy()
    spec = hdus['flux'].data.copy()
    unct = hdus['uncertainty'].data.copy() # read in uncertainties
    blaz = hdus['blaze'].data.copy()
    hdus.close()
    
    # Color points by uncertainty
    cbar = ax.scatter(wave[nord],spec[nord],c=unct[nord])
    fig.colorbar(cbar,ax=ax,label='Uncertainty')
fig.tight_layout()

# =============================================================================
# Plot Wavelengths Common to All Four Instruments

# Plot Echelle Order 93 (i.e. Standard Relative Order 161-93=68)
nord = 161-93

plt.figure(figsize=(12,3))
plt.xlabel('Wavelength [A]')
plt.ylabel('Normalized Counts')
for inst in ['harpsn','harps','expres','neid']:
    # Select a random file for each instrument
    file = np.random.choice(glob(os.path.join(spec_dir,f'*_{inst}.fits')))
    hdus = fits.open(file)
    wave = hdus['wavelength'].data.copy()
    spec = hdus['flux'].data.copy()
    cont = hdus['continuum'].data.copy()
    # Read in mask of common wavelengths
    cmask = hdus['common_mask'].data.astype(bool).copy()
    hdus.close()
    
    nord_mask = cmask[nord] # specify common wavelength mask for the order
    plt.plot(wave[nord][nord_mask],(spec[nord]/cont[nord])[nord_mask],label=inst)
    hdus.close()
plt.legend(loc=4,ncol=4)