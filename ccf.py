import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('./ESSP4/')
from utils import *
from data import readCCF

from expres.rv.ccf import generate_ccf_mask, order_wise_ccfs
from iCCF.meta import espdr_compute_CCF_fast

#g2_mask = '/Users/lilyzhao/Documents/ceph/CCF_Masks/ESPRESSO/ESPRESSO_G2.fits'
g2_mask = '/mnt/home/lzhao/ceph/ESPRESSO_MaskFiles/ESPRESSO_G2.fits'

# =============================================================================
# iCCF

def vactoair(wvln):
    """
    Adapted from the IDL astrolib vactoair routine
    See Exelis documentation for references & revision history
    Units of ANGSTROMS.
    """
    wvln_vac = np.copy(wvln)
    wvln_air = np.copy(wvln)
    g = np.nan_to_num(wvln_air) > 2000
    sigma2 = (1.e4 / wvln_vac[g])**2
    fact = 1. + 5.792105e-2/(238.0185e0 - sigma2) + 1.67917e-3/(57.362e0 - sigma2)
    wvln_air[g] = wvln_vac[g] / fact
    return wvln_air

def iccf(wave,spec,errs,blaz,berv,qual=None,
         vrange=20,v0=0,vspacing=.4,
         mask_file=g2_mask,
         bervmax=32,mask_width=0.5):
    
    # Need to mask out all NaNs for it to work
    m = np.isfinite(wave) & np.isfinite(spec) & np.isfinite(errs) & np.isfinite(blaz)
    if np.sum(m)==0:
        return np.full((3,len(v_grid)),np.nan)
    
    # Set the velocity grid for the CCF
    v_grid = v0 + np.arange(-vrange, vrange+vspacing*0.5, vspacing)

    # Element-wise difference in wavelength
    ll = vactoair(wave[m].copy())
    dll = np.diff(ll)
    dll = np.r_[dll, dll[-1]] # pad by repeating last element

    if qual is None:
        qual = np.zeros_like(ll)

    hdus = fits.open(mask_file)
    mask = hdus[1].data.copy()
    hdus.close()

    # Run!
    ccf_flux, ccf_error, ccf_quality = espdr_compute_CCF_fast(
        ll, dll, spec[m], errs[m], blaz[m], qual, v_grid,
        mask, berv, bervmax, mask_width=mask_width)

    return v_grid, ccf_flux, ccf_error

# =============================================================================
# EXPRES CCF

def eccf_orderwise(wvln,spec,errs,echl_ord0=161,
                   mask_file=g2_mask,npix=10,
                   vrange=20,v0=0,vspacing=.4,vwidth=None,
                   window='box',ccor=False):
    
    # Mask out <=0 and NaN values
    mask1 = np.isfinite(wvln) & np.isfinite(spec) & np.isfinite(errs)
    mask2 = (wvln>0) & (spec>0) & (errs>0)
    pix_mask = np.logical_and(mask1,mask2)
    
    # Get CCF Mask (code assumes CGS)
    vwidth = vwidth if vwidth is None else vwidth*1e5
    _, masks = generate_ccf_mask(mask_file, wvln,
                                 vspacing=vspacing*1e5,
                                 vwidth=vwidth,
                                 vrange=vrange*1e5,
                                 v0=v0*1e5)
    
    # Set the velocity grid for the CCF
    v_grid = v0 + np.arange(-vrange, vrange+vspacing*0.5, vspacing)
    # Generate the order-wise CCFs (where velocity is in CGS again)
    orders, ccfs, e_ccfs = order_wise_ccfs(masks, v_grid*1e5, wvln, spec, errs, npix=npix,
                                           window=window, pix_mask=pix_mask, ccor=ccor)
    
    # Shift orders to new starting order if specified
    orders = orders-160+echl_ord0 # 160 is the presumed initial order for EXPRES data
    
    return v_grid, ccfs, e_ccfs, orders

# =============================================================================
# Fit CCF

def Model_Gaussian(x, A, μ, σ, C):
    """
    Self-explanatory
    """
    return A * np.exp(-(x - μ)**2 / 2 / σ**2) + C

fit_opts = {'xtol':1e-8, 'gtol':1e-8, 'ftol':1e-8, 'factor':1.}
def fit_rv(v_grid, ccf, e_ccf, vrange=12, sigma_v=3, fit_opts=fit_opts,
           invert=False, opt_method='lm', model=Model_Gaussian, rv_guess=1.5):
    """Find a radial velocity by way of fitting a Gaussian to the CCF.

    Sometimes, this returns an OptimizeWarning (unable to determine covariances).
    Usually, this is because the default method of determining the correct
    ε to use for finite-difference evaluation of the Jacobian is failing.
    Either supply an analytic Jacobian, or supply a guess for ε
    via the `epsfcn` kwarg in fit_opts. e.g. epsfcn=.1 works pretty well
    for LFC diagnostic velocities.

    Parameters
    ----------
    v_grid : ndarray
        Array of velocity gridpoints at which CCF was evaluated
    ccf : ndarray
        The cross-correlation function to fit
    e_ccf : ndarray
        Error estimate of the CCF
    makeplots : bool
        Make diagnostic plots (default False)
    vrange : number
        velocity range to use in fit (to exclude noisy continuum)
    sigma_v : number
        Initial guess for velocity width of the CCF.
    fit_opts : dict
        Options for curve_fit
    invert : bool
        Whether or not the CCF was inverted
    opt_method : str
        Least squares method to pass into curve_fit
    ret_chi2 : bool
        Return the chi-squared of the fit along with the results

    Returns
    -------
    """
    # Set the guesses for the fit parameters
    offset = np.max(ccf) if not invert else np.min(ccf)
    amp = (np.min(ccf) - np.max(ccf)) * (-1 if invert else 1)
    if rv_guess is None:
        rv_guess = v_grid[np.argmin(ccf) if not invert else np.argmax(ccf)]
    p0 = [amp, rv_guess, sigma_v, offset]

    # Mask velocities outside vrange
    vmask = np.ones_like(v_grid, dtype=bool)
    if vrange is not None:
        vmask = np.abs(v_grid - rv_guess) < vrange
    vmask &= np.isfinite(ccf) & (e_ccf > 0.0)

    if np.sum(vmask) < 1:
        return np.nan, np.nan
    
    try:
        popt, pcov = curve_fit(model, v_grid[vmask],
                               ccf[vmask], p0=p0, sigma=e_ccf[vmask],
                               absolute_sigma=True, method=opt_method, **fit_opts)
    except RuntimeError:
        rv, rv_err = np.nan, np.nan
    else:
        if np.isnan(pcov[1, 1]) or pcov[1, 1] <= 0.0:
            rv, rv_err = np.nan, np.nan
        else:
            rv = popt[1]
            rv_err = np.sqrt(pcov[1, 1])
    
    return rv, rv_err


# =============================================================================
# CCF Pipeline (Spec File -> CCF File)

def ccf(spec_file,use_iccf=False,
        cont_norm=False,**kwargs):
    if os.path.basename(spec_file)==spec_file:
        spec_file = standardSpec_basename2FullPath(file)
    inst = os.path.basename(spec_file).split('_')[-1][:-5]
    
    # Read Spectral Data In
    hdus = fits.open(spec_file)
    wvln = hdus['wavelength'].data.copy()
    spec = hdus['flux'].data.copy()
    errs = hdus['uncertainty'].data.copy()
    time = hdus[0].header['mjd_utc']
    if cont_norm:
        spec /= hdus['continuum'].data.copy()
    if use_iccf:
        blaz = hdus['blaze'].data.copy()
        berv = hdus[0].header['berv']
    if inst=='neid':
        errs /= hdus['blaze'].data.copy()
    hdus.close()
    num_ord, num_pix = wvln.shape
    
    # Run CCF for All Orders
    if use_iccf:
        # Loop through all orders
        ccfs = None
        for iord in range(num_ord):
            v_grid, ccf, e_ccf = iccf(wvln[iord],spec[iord],errs[iord],
                                      blaz[iord],berv,**kwargs)
            if ccfs is None:
                ccfs, e_ccfs = np.full((2,iord,len(v_grid)),np.nan)
            ccfs[iord] = ccf.copy()
            e_ccfs[iord] = e_ccf.copy()
        orders = 161-np.arange(num_ord)
    else:
        v_grid, unordered_ccfs, unordered_e_ccfs, orders = eccf_orderwise(wvln,spec,errs,**kwargs)
        # Pad Orders as needed
        # (the EXPRES pipeline automatically removes orders with no CCF lines)
        ccfs, e_ccfs = np.full((2,num_ord,len(v_grid)),np.nan)
        for iord in range(num_ord):
            if (161-iord) in orders:
                nord = np.where((161-iord)==orders)[0][0]
                ccfs[iord] = unordered_ccfs[nord]
                e_ccfs[iord] = unordered_e_ccfs[nord]
        orders = 161-np.arange(num_ord)
    return time, v_grid, ccfs, e_ccfs, orders
    
def ccfFit(v_grid, ccfs, e_ccfs, orders, **kwargs):
    num_ord = len(ccfs)
    obo_rv, obo_rv_e = np.empty((2,num_ord))
    for iord in range(num_ord):
        if np.sum(np.isfinite(ccfs[iord]))==0:
            obo_rv[iord], obo_rv_e[iord] = np.nan, np.nan
        else:
            obo_rv[iord], obo_rv_e[iord] = fit_rv(v_grid,ccfs[iord],e_ccfs[iord], **kwargs)
    
    # Joined CCF
    ccf = np.nanmean(ccfs, axis=0)
    e_ccf = np.sqrt(np.nansum(e_ccfs**2, axis=0)) / np.sum(np.isfinite(e_ccfs), axis=0)
    obo_rv[iord], obo_rv_e[iord] = fit_rv(v_grid,ccf,e_ccf, **kwargs)
    
    ccf_rv, ccf_rv_e = fit_rv(v_grid,ccf,e_ccf, **kwargs)
    ccf_rv, ccf_rv_e = ccf_rv*1000, ccf_rv_e*1000
    
    ccf_dict = {
        'v_grid' : v_grid.copy(),
        'ccf'    : ccf.copy(),
        'e_ccf'  : e_ccf.copy(),
        'echelle_orders' : orders.copy(),
        'obo_ccf'   : ccfs.copy(),
        'obo_e_ccf' : e_ccfs.copy(),
        'obo_rv'    : obo_rv.copy(),
        'obo_e_rv'  : obo_rv_e.copy(),
    }
    
    return ccf_rv, ccf_rv_e, ccf_dict