import os
from glob import glob
import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy import ndimage
from skimage.draw import ellipse as circle
import pandas as pd
from tqdm import tqdm



import matplotlib.pyplot as     plt

import pandas            as     pd
import pickle
from   scipy             import ndimage
from   skimage.draw      import ellipse as circle
from   tqdm              import tqdm


# =============================================================================
# Coordniate Functions

getRsun = lambda data_dict : int(data_dict['header']['RSUN_OBS']/data_dict['header']['CDELT1']*0.99)

# Function for mu angle
def muang(data_dict,iy,ix):

    # Dimension
    Ny, Nx = data_dict['image'].shape
    Rsun = getRsun(data_dict)
    
    # Cartesian coordinates
    x = ix - Nx/2
    y = iy - Ny/2
    z = np.sqrt(Rsun**2 - x**2 - y**2)
    
    # Mu angle
    muang = np.cos(np.arctan2(np.sqrt(x**2+y**2),z))

    return muang

# Function for latitude & longitude
def latlon(data_dict,iz,iy):
    
    # Dimension
    Nz, Ny = data_dict['image'].shape
    Rsun = getRsun(data_dict)

    # Cartesian coordinates
    z = iz - Nz/2
    y = iy - Ny/2
    x = np.sqrt(Rsun**2 - y**2 - z**2)
    
    # Spherical coordinates
    lat = np.degrees(np.arctan2(np.sqrt(x**2+y**2),z)) - 90
    lon = np.degrees(np.arctan2(y,x))

    return lat, lon

def assembleSDOImages(date,data_dir,mask=True,norm=True,mag_std=8):
    ico_fits = fits.open(glob(os.path.join(data_dir,'Intensitygram_cont',f'hmi.Ic_720s.{date}*.fits'))[0])
    icf_fits = fits.open(glob(os.path.join(data_dir,'Intensitygram_flat',f'hmi.Ic_noLimbDark_720s.{date}*.fits'))[0])
    mag_fits = fits.open(glob(os.path.join(data_dir,'Magnetogram',f'hmi.M_720s.{date}*.fits'))[0])

    data_products = []
    for data_type,hdus in zip(('ico','icf','mag'),[ico_fits,icf_fits,mag_fits]):
        imag, head = np.flip(hdus[1].data.copy(),axis=1), hdus[1].header.copy()

        if mask:
            # Mask image outside R_sun
            Rsun = int(head['RSUN_OBS']/head['CDELT1']*0.99)
            xcen = int(head['CRPIX1'])
            ycen = int(head['CRPIX2'])
            circ = circle(xcen, ycen, Rsun, Rsun)
            r_mask = np.ones_like(imag, dtype=bool)
            r_mask[circ] = False
            imag[r_mask] = np.nan

        if norm:
            if data_type=='icf':
                imag /= np.nanmedian(imag)
            elif data_type=='mag':
                imag[np.abs(imag) < mag_std] = 0
        
        data_products.append(dict(zip(['image','header'],[imag,head])))
        hdus.close()

    return data_products


# =============================================================================
# Identify Active Regions

def groupLabels(imag,lim,less_than=True):
    if less_than:
        labelImage = lambda imag : ndimage.label(np.abs(imag) < lim)
    else:
        labelImage = lambda imag : ndimage.label(np.abs(imag) > lim)
    imag_lab, lab = labelImage(imag.copy())
    imag_map = imag_lab > 0
    
    # Pixel sizes of all groups
    size_pix = np.zeros(lab, dtype=int)
    pix      = np.where(imag_map)
    for i in range(len(pix[0])):
        size_pix[imag_lab[pix[0][i],pix[1][i]]-1] += 1
    
    return imag_lab, lab, size_pix

def bigGroupLabels(imag,lim,less_than=True,min_group_size=2):
    imag_lab, lab, size_pix = groupLabels(imag,lim,less_than=less_than)

    # Image copy with NaNs at positions of groups that are too small
    imag_copy = np.copy(imag)
    lab_small = np.where(size_pix < min_group_size)[0] + 1
    for i in range(len(lab_small)):
        imag_copy[imag_lab == lab_small[i]] = np.nan
    
    return groupLabels(imag_copy,lim,less_than=less_than)

def getCoordinates(imag_lab,lab):
    z = np.ones_like(lab)
    y = np.ones_like(lab)
    for i,idx in enumerate(lab):
        pix  = np.where(imag_lab == idx+1)
        z[i] = np.round(np.mean(pix[0])).astype(int)
        y[i] = np.round(np.mean(pix[1])).astype(int)

    return y,z

def getAreaAndRadius(data_dict,y,z,Rsun,size_pix):
    Asun = np.pi*Rsun**2
    area_obs = size_pix
    area_abs = area_obs/muang(data_dict,z,y)
    area_rel = area_abs/Asun
    
    radi_obs = np.sqrt(area_obs/np.pi)
    radi_abs = np.sqrt(area_abs/np.pi)
    radi_rel = radi_abs/Rsun

    names = np.concatenate([[f'{i}_{j}' for j in ['obs','abs','rel']]for i in ['area','radi']])

    return dict(zip(names,[area_obs,area_abs,area_rel,
                           radi_obs,radi_abs,radi_rel]))

def spots(icf,icf_lim=0.89):
    # Group Labels
    imag_Slab, Slab, Ssize_pix = bigGroupLabels(icf['image'],lim=icf_lim,
                                                less_than=True,min_group_size=2)

    # Coordinates
    Sy, Sz = getCoordinates(imag_Slab,range(Slab))
    
    # Area and radius
    Rsun = getRsun(icf)
    ar_dict = getAreaAndRadius(icf,Sy,Sz,Rsun,Ssize_pix)

    # Store all values
    Sdata = {}
    Sdata['size_pix' ] = Ssize_pix
    Sdata['z']         = Sz.copy()
    Sdata['y']         = Sy.copy()
    Sdata['mu_angle' ] = muang (icf,Sz,Sy)
    Sdata['latitude' ] = latlon(icf,Sz,Sy)[0]
    Sdata['longitude'] = latlon(icf,Sz,Sy)[1]
    Sdata |= ar_dict

    return imag_Slab, pd.DataFrame(Sdata)

def brights(mag,mag_lim=24):
    ### FACULAE
    imag_Flab, Flab, Fsize_pix = bigGroupLabels(mag['image'],lim=mag_lim,
                                                less_than=False,min_group_size=2)
    
    # Coordinates
    Fy, Fz = getCoordinates(imag_Flab,range(Flab))
    
    # Area and radius
    Rsun = getRsun(mag)
    Asun = np.pi*Rsun**2
    ar_dict = getAreaAndRadius(mag,Fy,Fz,Rsun,Fsize_pix)

    # Store all values
    Fdata = {}
    Fdata['size_pix' ] = Fsize_pix
    Fdata['plage']     = Fsize_pix/Asun > 20e-6 # Mark whether it's a plage or not
    Fdata['z']         = Fz.copy()
    Fdata['y']         = Fy.copy()
    Fdata['mu_angle' ] = muang (mag,Fz,Fy)
    Fdata['latitude' ] = latlon(mag,Fz,Fy)[0]
    Fdata['longitude'] = latlon(mag,Fz,Fy)[1]
    Fdata |= ar_dict

    return imag_Flab, pd.DataFrame(Fdata)