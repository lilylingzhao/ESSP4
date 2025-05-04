# Derive indicators for all Solar Data
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.constants import c
from scipy.optimize import curve_fit
from scipy import interpolate

from utils import unpadOrders

# =============================================================================
# Gaussian Fitting Functions

def gaussFunc(x, A, mu, sig, m , b):
    model = A*np.exp(-.5*((x-mu)/sig)**2)+m*x+b
    return model

def findGaussp0(x,y,invert=False):
    """
    Generate initial guesses for a Gaussian fit to a given CCF
    """
    A = y.max()-y.min() * (-1 if invert else 1)
    sig = 10.*np.nanmedian(np.diff(x)) # A rough estimate scaled by the spread
    mu = x[np.argmin(y)]+np.pi/1000 # Find position of lowest CCF w/ perturbation
    m = float((y[-1]-y[0])/len(y)) # Average slope between edge points
    b = np.nanmedian(y) # Average offset
    
    return [A, mu, sig, m, b]

def gaussFit(x,y,yerr=None,invert=True,**kwargs):
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        finite_mask &= np.isfinite(yerr)
    
    x, y = x[finite_mask], y[finite_mask]
    if yerr is not None:
        yerr = yerr[finite_mask]
    
    p0 = findGaussp0(x,y,invert=invert)
    try:
        popt, pcov = curve_fit(gaussFunc, x, y, p0=p0, sigma=yerr, method='lm',
                               **kwargs)
    except ValueError:
        assert False

    # Sometimes sigma is negative and that can't be good
    # Adding bounds to curve_fit messes up more things than it fixes
    popt[2] = abs(popt[2])

    return popt, pcov


# =============================================================================
# CCF FWHM/Contrast

def ccfFwhmContrast(x,y,yerr):
    # Fit to Gaussian
    popt, pcov = gaussFit(x,y,yerr,invert=True)

    # Calculate Wanted Values
    fwhm, e_fwhm = 2*np.sqrt(2*np.log(2))*popt[2], 2*np.sqrt(2*np.log(2))*np.sqrt(pcov[2,2])
    contrast, e_contrast = popt[0], np.sqrt(pcov[0,0])
    
    return (fwhm, e_fwhm), (contrast, e_contrast)


# =============================================================================
# CCF BIS

def getBisector(x,y,yerr,
                num_y_pts=100,x_pts_res=1e-3):
    """ Find emission in a specified line

    Parameters
    ----------
    x,y,yerr : float array (shape: num_vel)
        CCF x, y, and error values
    num_y_pts  : int (default: 100)
        Number of y points (i.e., resolution of bisector)
    x_pts_res : float, units of x (default: 0.1 mm/s assuming x is in cm/s)
        Velocity resolution of x grid
        (i.e., precision of returned bisector)
    
    Returns
    -------
    bis_x, bis_y : float array
        x and y positions of line bisector
    """
    
    # Fit CCF to Guassian
    popt, pcov = gaussFit(x,y,yerr,invert=True)
    G_A, G_mu, G_sig, G_m, G_b = popt
    
    # Bisector vertical points depending on specified resolution
    bis_y = np.linspace(y.min(),(G_m*(x[np.argmin(y)])+G_b),num_y_pts+2)[1:-1]
    bis_l, bis_r = np.full((2,num_y_pts),np.nan)
    
    # Interpolate the CCF
    if yerr is None:
        weights = None
    else:
        weights = 1/yerr
    f = interpolate.UnivariateSpline(x,y,w=weights,k=3)
    x_lft = np.arange(G_mu-4*G_sig,G_mu,x_pts_res)
    y_lft = f(x_lft)
    x_rgt = np.arange(G_mu,G_mu+4*G_sig,x_pts_res)
    y_rgt = f(x_rgt)
    
    # Find Closest x Value at Each bis_y
    for i,y_want in enumerate(bis_y):
        bis_l[i] = x_lft[np.argmin(abs(y_lft-y_want))]
        bis_r[i] = x_rgt[np.argmin(abs(y_rgt-y_want))]
    
    return (bis_l+bis_r)/2, bis_y

def findBIS(ccf_x,ccf_y,ccf_e,vt=(.6,.9),vb=(.1,.4),**kwargs):
    """ Find bisector inverse slope (BIS)

    Parameters
    ----------
    bis_x : float array
        x values of bisector
    vt : float (<1)
        percentile range of bisector line depth for "top"
    vb : float (<1)
        percentile range of bisector line depth for "bottom"
    Classic ranges
        Classic bis: vt = (.6,.9), vb = (.1,.4)
        bis+: vt = (.8,.9), vb = (.1,.2)
        bis-: vt = (.6,.7), vb = (.3,.4)
    
    Returns
    -------
    mean of top - mean of bottom
    """
    bis_x, bis_y = getBisector(ccf_x,ccf_y,ccf_e,**kwargs)
    
    # Separate out the specified percentiles
    num_point = len(bis_x)
    top_mask, bot_mask = np.zeros((2,num_point),dtype=bool)
    top_mask[int(num_point*vt[0]):int(num_point*vt[1])] = True
    bot_mask[int(num_point*vb[0]):int(num_point*vb[1])] = True
    
    # Don't include zero values
    top_mask = np.logical_and(top_mask, bis_x!=0)
    bot_mask = np.logical_and(bot_mask, bis_x!=0)
    
    # Find BIS
    bis_value = np.nanmean(bis_x[top_mask]) - np.nanmean(bis_x[bot_mask])
    return bis_value


# =============================================================================
# Line Emission

def lineEmission(wvln, spec, core=6564.6, fit_width=.5, intp_width=0.5, errs=None, **kwargs):
    # Remove All Nan Orders
    wvln, spec = unpadOrders(wvln), unpadOrders(spec)
    if errs is not None:
        errs = unpadOrders(errs)
    
    # Find Relevant Order
    num_ord, num_pix = wvln.shape
    ord_min = np.nanmin(wvln,axis=1)
    ord_max = np.nanmax(wvln,axis=1)
    ord_list = np.arange(num_ord)[(ord_min<(core-fit_width)) & (ord_max>(core+fit_width))]

    emis = np.zeros_like(ord_list,dtype=float)
    x_intp = np.linspace(core-intp_width/2,core+intp_width/2)
    for inord,nord in enumerate(ord_list):
        nord_mask = (np.abs(wvln[nord]-core)<(fit_width/2)) & np.isfinite(spec[nord])
        if errs is not None:
            weights = 1/errs[nord][nord_mask]
            cont_errs = errs[nord] # Where is this used?
        else:
            weights = None
            cont_errs = np.sqrt(spec[nord])
        f = interpolate.UnivariateSpline(wvln[nord,nord_mask],spec[nord,nord_mask],
                                         w=weights,k=3,**kwargs)
        emis[inord] = np.min(f(x_intp))

    return emis
        

# =============================================================================
# Calcium HK

def triangle(x,line_cent,line_width=2.18,y_max=2):
    m = y_max/(line_width/2)
    b1 = y_max-m*line_cent
    b2 = y_max+m*line_cent
    
    y = np.zeros_like(x)
    y[x<line_cent]  =  m*x[x<line_cent]+b1
    y[x>=line_cent] = -m*x[x>=line_cent]+b2
    y[y<0] = 0
    return y

# Hard Coded Values
ca_h, ca_k = 3969.6, 3934.7
vlims=(3950,3956) # These (and below) established in 241121_indicatorPipeline
rlims=(3996,4006)
cont_perct=(.75,.95)

def caHK(wvln, spec, errs):
    
    num_ord, num_pix = wvln.shape
    
    wvln[np.isnan(spec)] = np.nan
    ord_min = np.nanmin(wvln,axis=1)
    ord_max = np.nanmax(wvln,axis=1)
    
    # Get Continuum Values
    cont_values = np.empty(2,dtype=float)
    for icont,(cmin,cmax) in enumerate([rlims,vlims]):
        core = (cmin+cmax)/2
        width = (cmax-cmin)
        ord_list = np.arange(num_ord)[(ord_min<(core-width)) & (ord_max>(core+width))]

        cval_ord = np.empty(len(ord_list),dtype=float)
        for iord,nord in enumerate(ord_list):
            cmask = (wvln[nord]>cmin) & (wvln[nord]<cmax)
            min_ind, max_ind = np.array(np.sum(cmask)*np.array(cont_perct),dtype=int)
            cval_ord[iord] = np.nanmedian(np.sort((spec[nord])[cmask])[min_ind:max_ind])
        cont_values[icont] = np.nanmedian(cval_ord) * (1.16 if icont==1 else 1) # I really forget where this factor comes from

    # Get Emission
    core_values = np.empty(2,dtype=float)
    for icore,core in enumerate([ca_h,ca_k]):
        ord_list = np.arange(num_ord)[(ord_min<(core-5)) & (ord_max>(core+5))]
        
        eval_ord = np.empty(len(ord_list),dtype=float)
        snr_ord = np.empty(len(ord_list),dtype=float)
        for iord,nord in enumerate(ord_list):
            m = np.isfinite(wvln[nord]) & np.isfinite(spec[nord])
            eval_ord[iord] = np.average(spec[nord][m],weights=triangle(wvln[nord][m],core))
            snr_ord[iord] = np.median((spec/errs)[nord][m])
        core_values[icore] = eval_ord[np.argmax(snr_ord)] # use value from highest SNR Order
    
    return np.sum(core_values)/np.sum(cont_values)