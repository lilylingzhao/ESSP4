# Inject Doppler Shifts into Spectra

import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.constants import c, M_sun, M_earth, M_jup
from astropy.timeseries import LombScargle
import astropy.units as u
from scipy.signal import medfilt
from tqdm import tqdm
import pandas as pd

#import rebound
#from spock import FeatureClassifier

from kepler import getRV, getRV_K, getMfromK

from utils import solar_dir, ceph_dir

wave_diff_log = 1.47149502e-06 # Determined in 250331_telluricWavelength.ipynb
wave_min, wave_max = 3787, 11053
y_shift = 0.002

default_tell_file = os.path.join(solar_dir,'telluricMask.fits')
default_tapas_file = os.path.join(ceph_dir,'250425_telluric','kpno_telluric_16.0_51.0.fits')

# =============================================================================
# Mask Out Tellurics

def getSeleniteModel(file_list):
    """Collect SELENITE models from list of files

    Parameters
    ----------
    file_list: array, str
        list of EXPRES files containing SELENITE models
    
    Returns
    -------
    wave, tell, blaz : array, float
        Arrays of shape [len(file_list), number of echelle orders, number of pixels]
        Collected wavelength, SELENITE model, and flat (to get SNR) from across the
            specified EXPRES files
    """
    num_obs = len(file_list)
    
    # Do we have full paths or just file names
    full_path = os.path.basename(file_list[0])!=file_list[0]
    
    # Open an example file
    example_file = file_list[0] if full_path else os.path.join(solar_dir,'EXPRES',file_list[0])
    hdus = fits.open(example_file)
    tell = hdus[1].data['tellurics'].copy()
    hdus.close()
    num_ord, num_pix = tell.shape
    
    # Load in wavelengths, telluric models, and blaze functions
    waves, tells, blazs = np.empty((3,num_obs,num_ord,num_pix))
    for ifile,file in enumerate(file_list):
        hdus = fits.open(file if full_path else os.path.join(solar_dir,'EXPRES',file_list[0]))
        waves[ifile] = hdus[1].data['wavelength'].copy()
        tells[ifile] = hdus[1].data['tellurics'].copy()
        blazs[ifile] = hdus[1].data['blaze'].copy()
        hdus.close()
    
    # Get median wavelength value for each pixel
    if num_obs>1:
        wave = np.nanmedian(waves,axis=0)
        # Get pixel-by-pixel tellluric mask and smooth with median filter
        tell = np.nanmin(tells,axis=0)
        # Get minimum blaze value for each pixel
        blaz = np.nanmin(blazs,axis=0)
    else:
        wave, tell, blaz = waves[0], tells[0], blazs[0]
    
    return wave, tell, blaz

def makeSeleniteTelluricMask(wave,tell,blaz,
    tell_wave=None,wave_diff_log=1.47149502e-06*.75,
    tell_cut=0.99,smoothing_window=9):
    """Generate a telluric mask

    Parameters
    ----------
    wave, tell, blaz : array, float
        Arrays of shape [len(file_list), number of echelle orders, number of pixels]
        Collected wavelength, SELENITE model, and flat (to get SNR) from across the
            specified EXPRES files
    tell_wave : array, float
        Wavelength array for which to save telluric mask
        (i.e. tuned to be finer than any wavelength grid on which the mask will be evaluated)
    wave_diff : float
        Wavelength resolution of tell_wave to be constructed of tell_wave is not specified
    tell_cut : float
        Cutoff for tellurics
        i.e., tell values less than tell_cut will be considered telluric lines
        Recall that SELENTIE models are inherently normalized
    smoothing_window : int
        Window for the median filter that smoothes the telluric mask
        Should be tuned to remove features that are less than the instrument resolution
            (and therefore unphysical)

    Returns
    -------
    tell_wave, tell_mask : array, float/bool
        Wavelength grid and a mask of which wavelengths fall
            within telluric lines as defined by SELENITE models
    """
    num_ord, num_pix = wave.shape
    
    # Get pixel-by-pixel tellluric mask
    tell_mask = (tell<tell_cut).astype(float)
    
    ### Establish uniform wavelength grid if not given
    if tell_wave is None:
        x_shift = y_shift/wave_diff_log
        tell_wave = np.exp(np.arange(np.log(wave_min-x_shift),np.log(wave_max-x_shift),wave_diff_log))+x_shift
    
    # Get Telluric Mask Value for Highest SNR Across Wavelength Grid
    tell_mask_arr = np.zeros((num_ord,len(tell_wave)))
    tell_blaz_arr = np.zeros_like(tell_mask_arr,dtype=float)
    for iord in range(num_ord): # is there a way to numpy this?
        # Interpolate telluric mask onto uniform wavelength grid
        # Smooth order's telluric mask
        tell_mask_arr[iord] = np.round(np.interp(tell_wave,wave[iord],
                                                 np.ceil(medfilt(tell_mask[iord],smoothing_window)),
                                                 left=0,right=0)) # False where there is no data
        # Interpolate rough SNR of telluric model onto uniform wavelength grid
        tell_blaz_arr[iord] = np.interp(tell_wave,wave[iord],blaz[iord],
                                        left=0,right=0) # 0 SNR where there is no data
    # Find max blaze for each uniform wavelength value
    tell_blaz_max = np.nanmax(tell_blaz_arr,axis=0)
    # Mask out any telluric mask values from lower SNR pixels
    for iord in range(num_ord):
        tell_mask_arr[iord][tell_blaz_arr[iord]<tell_blaz_max] = np.nan

    # Get Telluric Mask
    return tell_wave, np.nanmax(tell_mask_arr,axis=0).astype(bool)


def getTapasModel(wave,transmission,
    tell_wave=None,wave_diff_log=1.47149502e-06*.75,
    tell_cut=0.7,derv_cut=1000,smoothing_window=9):
    
    # Get Telluric Derivative
    diff_t = np.diff(transmission)
    delta_T = np.nanmedian([[np.nan,*diff_t],[*diff_t,np.nan]],axis=0)
    diff_l = np.diff(wave)
    delta_l = np.nanmedian([[np.nan,*diff_l],[*diff_l,np.nan]],axis=0)/wave
    tell_derv = delta_T/delta_l
    
    mask = (np.abs(tell_derv)>derv_cut) | (transmission<tell_cut)
    mask = np.ceil(medfilt(mask.astype(int),smoothing_window)).astype(bool)
    
    if tell_wave is None:
        x_shift = y_shift/wave_diff_log
        tell_wave = np.exp(np.arange(np.log(wave_min-x_shift),np.log(wave_max-x_shift),wave_diff_log))+x_shift
    
    tell_mask = np.round(np.interp(tell_wave,wave,mask,left=0,right=0)).astype(bool)
    return tell_wave, tell_mask

# =============================================================================
# Injecting Doppler Shifts into Wavelengths

# Read Planet File Into Parameter Dictionary
def getParamDict(file_name):
    """Change parameter file into standard dictionary
    (format used by stability code)

    Parameters
    ----------
    file_name : str
        CSV file containing parameters of the system for
            which to derive RVs

    Returns
    -------
    parameter dictionary
    """
    return pd.read_csv(file_name).to_dict('list')

# Get RV Time Series for System
getRV_argNames = {'K [m/s]':'K','mass_pl [earth]':'Mpl','P [d]':'p',
                  'e':'e','w [rad]':'w','t0 [eMJD]':'t0','i [rad]':'i'}
def getRvTimeSeries(times,param_file,host_mass=1):
    """Calculate an RV time series given a system defined
    in param_file.

    Parameters
    ----------
    times: array, float
        time stamps for the RV time series
    param_file : str
        CSV file containing parameters of the system for
            which to derive RVs
        Requires ESSP format for parameters: 'K [m/s]', 'P [d]', 't0 [eMJD]', 'e', 'w [deg]', 'i [deg]'
        defaults exist for eccentricity (0) and inclination (pi/2 [rad])
    host_mass : float
        Mass of host star in units of solar masses

    Returns
    -------
    rv_v : array, float
        RVs at specified times for the system specified by param_file
    """
    param_df = pd.read_csv(param_file).set_index('planet')
    num_pln = len(param_df)
    
    # Fill in eccentricity values if not given
    if 'e' not in param_df.columns:
        param_df['e'] = 0
    # Fill in inclinations and convert degress to radians
    if 'i [deg]' not in param_df.columns:
        param_df['i [rad]'] = np.pi/2
    else:
        param_df['i [rad]'] = param_df['i [deg]']*np.pi/180
    param_df['w [rad]'] = param_df['w [deg]']*np.pi/180
    
    param_list = param_df.rename(columns=getRV_argNames).to_dict('records')
    # Generate RV Curve
    rv_v = np.zeros_like(times)
    use_k = 'K [m/s]' in param_df.columns
    if not use_k:
        assert 'mass_pl [earth]' in param_df.columns
    for ipln in range(num_pln):
        pln_dict = param_list[ipln]
        if use_k:
            rv_v += getRV_K(times,**{param:pln_dict[param] for param in ['K','p','e','w','t0','i']})
        else:
            rv_v += getRV(times,host_mass,
                          **{param:pln_dict[param] for param in ['Mpl','p','e','w','t0','i']})
    
    return rv_v

def getTellMask(wave,tell_file=default_tell_file):
    """Derive a telluric mask with same shape as wave

    Parameters
    ----------
    wave : array, float
        wavelength over which to derive telluric mask
    tell_file : string
        File containing telluric information
        (can be generated using mKTelluricMask.py)

    Returns
    -------
    tell_mask : array, bool
        Mask with same dimensions as wave
        True for wavelengths where there are tellurics
    """
    # Read in telluric mask
    hdus = fits.open(tell_file)
    tell_wave = hdus['wavelength'].data.copy()
    tell_vals = hdus['telluric_mask'].data.copy().astype(bool)
    hdus.close()
    
    # Interpolate onto wavelength in question
    tell_mask = np.ceil(np.interp(wave,tell_wave,tell_vals)).astype(bool)
    return tell_mask

def injectPlanet(rv,data_dict,tell_file=default_tell_file):
    """Inject specified RV into standard data_dict info

    Parameters
    ----------
    rv : float
        Velocity to Doppler shift data_dict by in units of m/s
    data_dict : dictionary
        Standard data_dict (created in mkDataSet.py)
    tell_file : string
        File containing telluric information
        (can be generated using mKTelluricMask.py)

    Returns
    -------
    tell_mask : array, bool
        Mask with same dimensions as data_dict info
        True where there are tellurics
    data_dict : dictionary
        Same data_dict as input, but with NaNs where there are tellurics
        for the wavelengths, flux, and uncertainty
    """
    shifted_dict = data_dict.copy()
    wave = data_dict['wavelength'].copy()/np.exp(np.arctanh(rv/c.to(u.m/u.s).value))
    tell_mask = getTellMask(wave,tell_file=tell_file)
    
    shifted_dict['wavelength'] = wave.copy()
    for key in ['flux','uncertainty']:
        shifted_dict[key][tell_mask] = np.nan
    
    shifted_dict['telluric_mask'] = tell_mask.astype(int)
    
    return tell_mask, shifted_dict


# =============================================================================
# Detectability Check

# Check for Detectability Given Instrument Errors and Time Sampling
# NOT SURE IF I'M DOING RIGHT BY THE MULTI-PLANET
def getDetectability(times,param_file,errs,host_mass=1,num_tests=1500,
                     min_freq=None,max_freq=None,perct_peak=(.95,1.05)):
    rv_t, rv_V = getRvTimeSeries(times,pd.read_csv(param_file),host_mass=host_mass)
    
    pvals = np.zeros(num_pln)
    for _ in range(num_tests):
        # Generate representative error
        noise = np.random.randn(len(rv_t))*errs
        
        # Calculate white-noise periodogram for comparison
        wn = LombScargle(rv_t, noise)
        wnFreq, wnPowr = wn.autopower(minimum_frequency=min_freq,
                                      maximum_frequency=max_freq)
        wnPerds = 1/wnFreq.copy()
        wn_peak = np.max(wnPowr)
        
        # Calculate periodogram for simulated RV curve + noise
        ls = LombScargle(rv_t, rv_v+noise, dy=errs)
        for ipln in range(num_pln):
            pln_period = param_dict['period'][ipln]
            pln_freqs =  1/np.linspace(perct_peak[0]*pln_period,perct_peak[1]*pln_period,9)
            pln_peak = np.max(ls.power(pln_freqs))

            # Count whether or not planet periodicity is less significant than
            #     a peak in the white noise only periocogram
            pvals[ipln] += float(pln_peak <= wn_peak)
    
    # Return false alarm percentage
    return pvals/num_tests


# =============================================================================
# Stability Check using SPOCK
"""
def getStability(param_dict,host_mass=1):
    Use SPOCK code to estimate the probability that the 
    specified system is dynamically stable.

    Parameters
    ----------
    param_dict : dict
        dictionary with orbital parameters for each planet
        Requires ESSP format for parameters: 'K [m/s]', 'P [d]', 't0 [eMJD]', 'e', 'w [deg]', 'i [deg]'
        defaults exist for eccentricity (0) and inclination (pi/2 [rad])
    host_mass : float [solar masses]
        mass of host star

    Returns
    -------
    stability_probability : float
        percent chance the system is dynamically stable
    
    # Initialize REBOUND object
    sim = rebound.Simulation()
    sim.add(m=host_mass*M_sun.value) # host star in solar masses
    
    # Add Planets
    num_pln = len(param_dict['K [m/s]'])
    for ipln in range(num_pln):
        ecc = param_dict['e'][ipln] if 'e' in param_dict.keys() else 0
        inc = param_dict['i [deg]'][ipln]/180*np.pi if 'i [deg]' in param_dict.keys() else np.pi/2
        mass_pl = getMfromK(K=param_dict['K [m/s]'][ipln],Mstar=host_mass,
                            p=param_dict['P [d]'][ipln],e=ecc,i=inc)
        
        sim.add(m=mass_pl,
                P=param_dict['P [d]'][ipln],
                e=ecc, inc=inc, # all degrees should be given in radians
                omega=param_dict['w [deg]'][ipln]/180*np.pi,
                T=param_dict['t0 [eMJD]'][ipln])
        sim.move_to_com()
    
    # Get Probability of Stability
    stability_probability = FeatureClassifier().predict_stable(sim)

    return stability_probability
"""