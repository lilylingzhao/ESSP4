# Fit the Continuum of Solar Data
# (I'm putting this in a separate file than data
#     to contain the EXPRES pipeline dependency)
import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.constants import c
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import pandas as pd

from data import *
from expres.cont_norm import cont_norm


# =============================================================================
# HARPS(-N) Alignment/Continuum Fitting Functions

def offsetFit(p,y,t):
    # Function that aligns orders
    return t-(y*p[0]+p[1])

def invFunc(p,wvln,spec):
    # function that unaligns orders
    spec = ((spec/np.poly1d(p[2:])(wvln))-p[1])/p[0]
    return spec

def alignHarps(wvln,spec,ret_inverse=False):
    """Align orders

    Parameters
    ----------
    wave, spec : array, floats
       wavelength and flux of orders to be aligned
    ret_inverse:
        return prameters to allow inverting from aligned orders
        to original counts

    Returns
    -------
    spec_aligned : array, float
       flux values of aligned spectrum
    inv_params : list (optional)
        parameters of the transformation to aligned spectrum
    """
    num_ord, num_pix = wvln.shape
    spec_aligned = np.full(spec.shape,np.nan)
    smax = np.nanmax(spec)
    spec_aligned[0] = spec[0].copy() + smax
    
    if ret_inverse:
        inv_params = [(1,smax,0,1)]
    for iord in range(1,num_ord):
        # Interpolate overlapping region
        wmin, wmax = np.nanmin(wvln[iord]),np.nanmax(wvln[iord-1])
        if wmax < wmin:
            break
        xarr = np.linspace(wmin,wmax,250)
        f0 = interp1d(wvln[iord-1],spec_aligned[iord-1],
                      bounds_error=False,fill_value=np.nan)
        f1 = interp1d(wvln[iord],spec[iord],
                      bounds_error=False,fill_value=np.nan)
        
        # Fit for offset and scaling
        s0, s1 = f0(xarr), f1(xarr)
        res = least_squares(offsetFit, [np.nanstd(s0-s1),np.nanmedian(s0-s1)],
                            method='lm', args=(s1,s0))
        p = res.x
        # Fit for residual slope
        lin_p = np.polyfit(xarr,s0/(s1*p[0]+p[1]),1)
        lin_fit = np.poly1d(lin_p)
        
        # Correct slope
        spec_aligned[iord] = (spec[iord]*p[0]+p[1])*lin_fit(wvln[iord])
        
        if ret_inverse:
            inv_params.append((*res.x,*lin_p))
    
    if ret_inverse:
        return spec_aligned, inv_params
    return spec_aligned

def contAligned(wvln,spec,errs,order_range=3,
                **kwargs):
    """Align consecutive orders and fit continuum

    Parameters
    ----------
    wave, spec, errs : array, floats
       wavelength, flux, and error of spectrum
    order_range : int (default: 3)
        +/- number of adjacent orders to align
        (i.e. total orders = order_range * 2 + 1
    kwargs
        optional arguments for the continuum fitting

    Returns
    -------
    cont: array, float
       resultant continuum
    """
    num_ord, num_pix = wvln.shape
    cont = np.ones_like(spec)
    
    for iord in range(num_ord):
        omin = iord-order_range if iord>order_range else 0
        omax = iord+order_range+1
        iord_aligned = iord-omin
        # I don't know why I have to do this, but I really do
        # Align spectra with each other
        spec_aligned, inv_params = alignHarps(wvln.copy()[omin:omax],spec.copy()[omin:omax],ret_inverse=True)
        cont_aligned = cont_norm(wvln.copy()[omin:omax],spec_aligned,errs.copy()[omin:omax],**kwargs)
        cont[iord] = invFunc(inv_params[iord_aligned],wvln.copy()[iord],cont_aligned[iord_aligned])
     
    return cont


# =============================================================================
# Global Continuum Fitting Function

def solarCont(file_name):
    """Derive a continuum for the standardized solar data

    Parameters
    ----------
    file_name : str
        name of standardized data file

    Returns
    -------
    cont: array, float
       continuum fit for data specified by file_name
    """
    inst = fileName2Inst(file_name)
    wvln, spec, errs, blaz = readL2(file_name)
    spec /= blaz
    num_ord, num_pix = wvln.shape
    cont = np.empty_like(wvln)
    
    if inst == 'neid':
        ord_split = 66
        # Unclear why errs needs to be divided by blaz here, but it doesn't work otherwise
        cont[:ord_split] = cont_norm(wvln[:ord_split],spec[:ord_split],
                                     (errs/blaz)[:ord_split],knot_res=50)
        cont[ord_split:] = cont_norm(wvln[ord_split:],spec[ord_split:],
                                     (errs/blaz)[ord_split:],knot_res=30)
    elif inst=='harps':
        for iord in range(num_ord):
            cont[iord] = cont_norm(wvln[iord],spec[iord],errs[iord],
                                   method='poly',deg=1 if iord<45 else 2)
    elif inst in ['harpsn','harps-n']:
        # Fit all orders to a simple linear fit (thank you blaze model!
        for iord in range(num_ord):
            cont[iord] = cont_norm(wvln[iord],spec[iord],errs[iord],method='poly',deg=1)
    elif inst == 'expres':
        hdus = fits.open(file_name)
        cont = hdus[1].data['continuum'].copy()
        hdus.close()
    else:
        assert False, print(f'Instrument name "{inst}" not recognized')
    
    return cont*blaz