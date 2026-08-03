# Fit the Continuum of Solar Data
import os
import numpy as np
from collections.abc import Sequence
from astropy.io import fits
from astropy.time import Time
from astropy.constants import c
from scipy.interpolate import interp1d, UnivariateSpline, LSQUnivariateSpline
from scipy.optimize import least_squares
import pandas as pd

from data import *
from tqdm.auto import trange, tqdm

# =============================================================================
# Continuum Fitting Functions Ported From EXPRES Pipeline
# (Functions preserved exactly as much as possible)

# =====================================
# splinefit.py

RESOLUTION = 300000

def spline_fit(x, y, knots=None, knot_spacing=None, n_knots=None, knot_res=None,
               k=3, clean=False, w=None,  **kwargs):
    """Wrapper for Scipy's spline-fitting functions.

    Parameters
    ----------
    x, y : ndarray
        The x and y data
    knots : ndarray or list, optional
        The positions of the knots in the b-spline
    knot_spacing : number, optional
        The spacing of knots in the b-spline. Overrides n_knots.
    n_knots : int, optional
        The number of knots to input into the b-spline. Minimum = 2.
    k : int
        The degree of the b-spline
    clean : bool
        Remove knots that have no x-values between them
    w : ndarray
        Weights corresponding to each y value
    sort : bool
        Sort x, y, and w by x before fitting the spline

    Returns
    -------
    spline : callable
        A b-spline functions that best fits the given data and parameters.
    """
    if k > 3:
        k = 3

    # Sort the unique elements of x and re-order y and w
    s = np.unique(x, return_index=True)[1]
    x = x.ravel()[s]
    y = y.ravel()[s]
    if w is not None:
        w = w.ravel()[s]

    # Use UnivariateSpline if no knot parameters are defined
    if knots is None and knot_spacing is None and n_knots is None and knot_res is None:
        return UnivariateSpline(x, y, k=k, w=w, **kwargs)

    # Define the knots of the b-spline
    if knots is None:
        if knot_spacing is not None:
            n_knots = int((np.max(x) - np.min(x)) // knot_spacing + 1)
        if n_knots is not None:
            knots = np.linspace(np.min(x), np.max(x), n_knots)[1:-1]
        elif knot_res is not None:
            knots = resolution_knots(np.min(x), np.max(x), knot_res)[1:-1]

    # Remove knots that have no x-values between them
    if clean:
        for i in np.argwhere(np.diff(x) > np.max(np.diff(knots))):
            knots = knots[(knots <= x[i]) | (knots >= x[i+1])]

    return LSQUnivariateSpline(x, y, knots, k=k, w=w, **kwargs)


def resolution_knots(min_x, max_x, res=RESOLUTION):
    """Create a set of knots equally spaced by resolution (log-space)

    Derivation:
    w1 = w0 + dw
    R = w / dw
    w1 = w0 + w0 / R = w0 (1 + 1/R)
    ln(w1) = ln(w0) + ln(1 + 1/R)
    """
    log_knots = np.arange(np.log(min_x), np.log(max_x), np.log(1 + 1/res))
    return np.exp(log_knots)

# =====================================
# fit_reject.py

def fit_reject(xdata, ydata, yerr, init_mask=None, cut=3.0, signed_cut=False,
               ret_mask=False, n=None, **kwargs):
    """Fit the data while iteratively rejecting outliers

    Both `cut` and `signed_cut` can be iterables, meaning mulutiple "levels"
    of outlier can be rejected. Therefore, since `mask` updates in each `cuts`
    iteration, once an outlier is rejected in one iteration, it cannot later
    become a non-outlier, so choose these cuts wisely.

    Parameters
    ----------
    xdata : ndarray
        The x data
    ydata : ndarray
        The y data
    yerr : ndarray
        The errors corresponding to the ydata
    init_mask : ndarray
        An initial mask of pixels to avoid while fitting
    cut : float or list(float)
        The standard deviation cut(s) when masking outliers
    signed_cut : bool or list(bool)
        Whether or not each cut is signed. When False, both positive and
        negative outliers are rejected. A list must correspond with a list
        of cuts.
    ret_mask : bool
        Return the mask along with the fit continuum
    method : str ('poly' or 'spline')
        The fitting method to use
    n : int, optional
        Maximum number of values to mask in each iteration. If None, rejected
        all assumed outliers in each iteration.

    Returns
    -------
    fit : callable
        The best fit for the data
    mask : ndarray, optional
        The mask of outliers. Only returned if ret_mask is True.
    """
    # Make sure cut and signed_cut are both iterables
    if not isinstance(cut, Sequence):
        cut = [cut]
    if len(cut) < 1:
        raise ValueError("`cut` must have at least one value")

    if not isinstance(signed_cut, Sequence):
        signed_cut = [signed_cut]
    if len(signed_cut) == 1:
        signed_cut *= len(cut)
    elif len(signed_cut) != len(cut):
        raise ValueError("`signed_cut` must have the same length as `cut`")

    # Set the mask for intial outliers
    if init_mask is None:
        init_mask = np.ones_like(xdata, dtype=bool)
    init_mask &= np.isfinite(xdata) & np.isfinite(ydata) & (np.nan_to_num(yerr) > 0.0)

    # Instantiate the outputs
    fit = None
    mask = np.copy(init_mask)

    # Iterate through each cut
    for c, s in zip(cut, signed_cut):
        if n is None:
            fit, mask = _fit_reject(xdata, ydata, yerr, mask,
                                    cut=c, signed_cut=s, **kwargs)
        else:
            fit, mask = _fit_reject_n(xdata, ydata, yerr, mask, n=n,
                                      cut=c, signed_cut=s, **kwargs)

    if ret_mask:
        return fit, mask
    return fit


def _fit_reject(xdata, ydata, yerr, init_mask, cut=3.0, signed_cut=False,
                rtol=1e-5, atol=0.0, max_iter=1000, **kwargs):
    """Fit data and reject all valid outliers in each iteration

    Any value (not in init_mask) can return as a non-outlier after being
    rejected, therefore, tolerances for breaking from the loop are necessary
    to avoid infinite recursion

    Parameters
    ----------
    xdata : ndarray
        The x data
    ydata : ndarray
        The y data
    yerr : ndarray
        The errors corresponding to the ydata
    init_mask : ndarray
        An initial mask of pixels to avoid while fitting
    cut : float or list(float)
        The standard deviation cut(s) when masking outliers
    signed_cut : bool or list(bool)
        Whether or not each cut is signed. When False, both positive and
        negative outliers are rejected. A list must correspond with a list
        of cuts.
    rtol : float
        Relative tolerance (based on init_mask) for the iterations to stop.
        For example, if there are 10,000 valid values in ydata and rtol is
        1e-3, then less than 10 values need to be rejected in an iteration
        for the loop to stop.
    atol : scalar
        Absolute tolerance for the iterations to stop. For example, if atol
        is 10 and rtol is 0.0, then less than 10 values need to be rejected
        in an iteration for the loop to stop.
    max_iter : int
        Maximum number of iterations to loop over before forcing the loop to
        stop.

    Returns
    -------
    fit : callable
        The best fit function for the data
    mask : ndarray
        Boolean array where False corresponds to masked outliers
    """
    tol = rtol * np.sum(init_mask) + atol
    mask = np.copy(init_mask)

    # Iterate until all outliers are removed
    itr = 1
    while np.sum(mask) > 1:

        fit, resid = _fit_data(xdata, ydata, yerr, mask, **kwargs)

        if not signed_cut and cut > 0:
            new_mask = init_mask & (np.abs(resid) < np.abs(cut))
        elif cut < 0:
            new_mask = init_mask & (resid > -np.abs(cut))
        elif cut > 0:
            new_mask = init_mask & (resid < np.abs(cut))
        else:
            raise ValueError("`cut` must be nonzero")

        if np.sum(mask ^ new_mask) <= tol:
            break

        if itr >= max_iter:
            break

        mask = np.copy(new_mask)
        itr += 1
    else:
        raise RuntimeError("Too many pixels were rejected while fitting the data")

    return fit, mask


def _fit_reject_n(xdata, ydata, yerr, init_mask, cut=3.0, signed_cut=False,
                  n=1, **kwargs):
    """Fit data and reject the n largest outliers in each iteration

    Any outlier rejected in a given iteration CANNOT later become a
    non-outlier, os make sure `n` is not too large here

    Parameters
    ----------
    xdata : ndarray
        The x data
    ydata : ndarray
        The y data
    yerr : ndarray
        The errors corresponding to the ydata
    init_mask : ndarray
        An initial mask of pixels to avoid while fitting
    cut : float or list(float)
        The standard deviation cut(s) when masking outliers
    signed_cut : bool or list(bool)
        Whether or not each cut is signed. When False, both positive and
        negative outliers are rejected. A list must correspond with a list
        of cuts.
    n : int
        Maximum number of outliers to reject in each iteration. A larger `n`
        means the algorithm will run faster, but it is more likely that
        non-outlier values will be rejected.

    Returns
    -------
    fit : callable
        The best fit function for the data
    mask : ndarray
        Boolean array where False corresponds to masked outliers
    """
    mask = np.copy(init_mask)

    # Iterate until all outliers are removed
    while np.sum(mask) > 1:

        fit, resid = _fit_data(xdata, ydata, yerr, mask, **kwargs)

        # Once a value is rejected, it can not come back
        resid *= mask

        if not signed_cut and cut > 0:
            if np.all(np.abs(resid) < np.abs(cut)):
                break
            cut_n = max(np.abs(cut), np.partition(np.abs(resid), -n)[-n])
            mask[np.abs(resid) >= cut_n] = False

        elif cut < 0:
            if np.all(resid > cut):
                break
            cut_n = min(-np.abs(cut), np.partition(resid, n-1)[n-1])
            mask[resid <= cut_n] = False

        elif cut > 0:
            if np.all(resid < cut):
                break
            cut_n = max(np.abs(cut), np.partition(resid, -n)[-n])
            mask[resid >= cut_n] = False

        else:
            raise ValueError("`cut` must be nonzero")

    else:
        raise RuntimeError("Too many pixels were rejected while fitting the data")

    return fit, mask


def _fit_data(xdata, ydata, yerr, mask, method='spline', deg=3, knot_res=None, **kwargs):
    """Fit a function to the data and calculate the residual

    Parameters
    ----------
    xdata : ndarray
        The x data
    ydata : ndarray
        The y data
    yerr : ndarray
        The errors corresponding to the ydata
    mask : ndarray
        Outlier mask corresponding to ydata
    method : str ('poly' or 'spline')
        The fitting function to use
    deg : int
        Degree of the fitting function

    Returns
    -------
    fit : callable
        The best fit function for the data
    resid : ndarray
        Residual array corresponding to `(ydata - fit(xdata)) / yerr`
    mask : ndarray
        Boolean array where False corresponds to masked outliers"""
    if method not in ['poly', 'spline']:
        raise ValueError("method must be `poly` or `spline`")

    if method == 'poly':
        fit = np.poly1d(np.polyfit(xdata[mask], ydata[mask], w=1/yerr[mask],
                                   deg=deg, **kwargs))
    else:
        fit = spline_fit(xdata[mask], ydata[mask], w=1/yerr[mask],
                         k=deg, knot_res=knot_res, **kwargs)

    # import matplotlib.pyplot as plt
    # plt.errorbar(xdata, ydata, yerr, fmt='.', ms=1, lw=1, c='C0', zorder=0)
    # plt.plot(xdata[~mask], ydata[~mask], '.', ms=1, c='r', zorder=1)
    # plt.plot(xdata[mask], fit(xdata[mask]), c='C1', lw=1, zorder=2)
    # plt.show()

    resid = np.nan_to_num((ydata - fit(xdata)) / yerr)

    return fit, resid

# =====================================
# cont_norm.py

METHOD = 'spline'
CUT = -1.5
DEG = 3
RES = 100

BAD_REGIONS = ((6868, 6883),  # oxygen B-band
               (7593, 7675))  # oxygen A-band

def _cont_norm(wvln, spec, errs, mask, method=METHOD, cut=CUT, deg=DEG, knot_res=RES,
               debug=False, **kwargs):
    """Internal function to approximate the continuum of a spectrum

    Parameters
    ----------
    wvln : 1D ndarray
        The x-values (e.g. wavelengths) corresponding to spec
    spec : 1D ndarray
        The spectral values of the order
    errs : 1D ndarray
        The absolute errors of spec
    mask : 1D ndarray
        An initial mask of pixels to avoid while fitting the order
    method : str
        Fitting function for the continuum (see fit_reject(...))
    cut : float
        The standard deviation cut when masking absorption lines from the fit
    deg : int
        The polynomial degree for the continuum fit
    res : scalar
        Resolution of B-spline knots, if using this method
    debug : bool
        Show debug plots

    Returns
    -------
    cont : 1D ndarray
        The continuum for the order
    mask : 1D ndarray
        The pixels masked by the continuum fitter.
        Only returned if ret_mask is True.
    """
    try:
        fit, mask = fit_reject(wvln, spec, errs, init_mask=mask, ret_mask=True,
                               cut=cut, method=method, deg=deg, knot_res=knot_res, **kwargs)
    except RuntimeError:
        tqdm.write(f'Too many pixels were masked in continuum normalization')
        cont = np.ones_like(wvln)
        mask = np.zeros_like(wvln, dtype=bool)
    else:
        cont = fit(wvln)

    cont[np.isnan(spec)] = np.nan

    if debug:
        plot_cont_norm(wvln, spec, errs, cont, mask, deg)

    return cont, mask

def cont_norm(wvln, spec, errs, mask=None, ret_mask=False, **kwargs):
    """Continuum normalize a spectrum by rejecting absorption (or emission) lines.

    Parameters
    ----------
    wvln : 1D ndarray
        The wavelengths corresponding to spec
    spec : 1D ndarray
        The spectral values of the order
    errs : 1D ndarray
        The absolute errors of spec
    mask : 1D ndarray
        An initial mask of values to avoid while fitting the continuum
    ret_mask : bool
        Return the mask along with the fit continuum

    Returns
    -------
    cont : 1D ndarray
        The continuum for the order
    mask : 1D ndarray, optional
        The pixels masked by the continuum fitter.
        Only returned if ret_mask is True.
    """
    if mask is None:
        mask = np.ones_like(spec, dtype=bool)

    # Mask out known deep line regions in the spectrum
    for l, r in BAD_REGIONS:
        mask &= (wvln < l) | (wvln > r)

    # Calculate the continuum
    cont, mask = _cont_norm(wvln, spec, errs, mask, **kwargs)

    if ret_mask:
        return cont, mask
    return cont


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
            cont[iord] = cont_norm(wvln[iord],spec[iord],errs[iord],
                                   method='poly',deg=1 if iord<=4 else 2)
    elif inst == 'expres':
        hdus = fits.open(file_name)
        cont = hdus[1].data['continuum'].copy()
        hdus.close()
    else:
        assert False, print(f'Instrument name "{inst}" not recognized')
    
    return cont*blaz