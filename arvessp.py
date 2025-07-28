# ARVE CCFs/LBL RVs
# (requires a separate environment)
import arve
import os
from glob import glob
import numpy as np
from astropy.io import fits
import pickle

from utils import *

# =============================================================================
# CCF

default_arve_mask = default_mask_file.replace('.fits','.csv')

def makeArveMask(fits_file,save_file,overwrite=False):
    """Re-write FITS mask (ESPRESSO style) into CSV for ARVe

    Parameters
    ----------
    fits_file : str
        File name of ESPRESSO FITS mask
    save_file : str
        Where to save CSV file

    Returns
    -------
    """
    assert os.path.isfile(fits_file), print('Cannot find specified mask fits file')
    if os.path.isfile(save_file) and not overwrite:
        return

    hdus = fits.open(fits_file)
    wave   = hdus[1].data['lambda'  ].astype('float64')
    weight = hdus[1].data['contrast'].astype('float64')**2
    hdus.close()
    # Need to add wave_l and wave_u columns,
    #     which are required by ARVE
    mask_df = pd.DataFrame({
        'wave'  : wave.copy(),
        'wave_l': wave.copy(),
        'wave_u': wave.copy(),
        'weight': weight.copy()
    }).to_csv(save_file,index=False)
    
    return mask_df

def arveCCF(data,instrument='essp',medium_data='vac',
    mask_file=default_arve_mask,medium_mask='air',
    vrange=20,v0=0,vspacing=.4,save_file=''):
    """Get ARVE CCFs for specified data
    specified either as a directory of FITS files or 
    list of wavelength, flux, and errors

    Parameters
    ----------
    data
        str  : directory containing FITs files to incorporate
        list : wvln, spec, err of spectra to CCF
    instrument : str (default: essp)
        specify data format of input files (all lowercase)
        expres, harps, harps-n, neid, and essp are all relevant options
    medium_data : str [air, vac]
        if input data given, specify if wavelengths are in air or vac
    masks_file, medium_mask : str
        CSV file of CCF mask and whether wavlenghts are 'air' or 'vac'
    vrange, v0, vspacing : floats
        specy velcity grid of CCF

    Returns
    -------
    ccf_dict
        Info about the observations ('files','times'),
        the CCFs ('v_grid','obo_ccf','obo_e_ccf'),
        and the ARVE-fitted RVS ('obo_rv','obo_e_rv')
    """
    # Initiate ARVE object
    arve_inst = arve.ARVE()
    # Specify the target star
    arve_inst.star.target = 'Sun'
    # Get the stellar parameters
    # (used to shift the spectra by the systematic velocity, which is 0 for the Sun)
    arve_inst.star.get_stellar_parameters()
    
    # Add data to ARVE in one of two ways
    if type(data)==str:
        # Add data by pointing to directory with spectra
        arve_inst.data.add_data(path=data, instrument=instrument,
            format='s2d', extension='fits', berv_corrected=True)
    elif type(data)==list:
        wvln, spec, errs = [np.array([i]).copy() for i in data]
        wvln = wvln[0]
        # Fill in data products
        arve_inst.data.add_data(wave_val=wvln,flux_val=spec,flux_err=errs, # Essential
                                medium=medium_data,format='s2d', # might be helpful
                                resolution=1e5,instrument='custom') # probably not useful
        arve_inst.data.spec['files'] = ['']
        
    # Get auxiliary data (such as CCF mask lines)
    arve_inst.data.get_aux_data(path_mask=mask_file,      # path to the CSV mask
                                medium_mask=medium_mask)   # medium of the mask wavelengths (if different than the spectrum wavelengths, the mask wavelengths will be converted to same medium as the spectrum)
    
    # Compute CCF RVs
    arve_inst.data.compute_vrad_ccf(weight_name='weight',    # same name as in the ESPRESSO mask converted to CSV
                                    exclude_tellurics=False, # turned off becasue ESPRESSO already has a cleaned mask
                                    ccf_err_scale=True,      # scale the CCF errors so that the RV error becomes independent of the velocity step size of the CCF
                                    vrad_grid=[v0-vrange,v0+vrange,vspacing]) # start, stop and step velocity for the CCF calculation, in km/s

    if len(arve_inst.data.spec['files'])==1:
        ccf_dict = {
            'v_grid' : arve_inst.data.ccf['ccf_vrad'].copy(),
            'obo_ccf'   : arve_inst.data.ccf['ccf_val'][0].copy(),
            'obo_e_ccf' : arve_inst.data.ccf['ccf_err'][0].copy(),
            'obo_rv'    : arve_inst.data.vrad['vrad_val_ord'][0].copy()*1000,
            'obo_e_rv'  : arve_inst.data.vrad['vrad_err_ord'][0].copy()*1000,
        }
    else:
        ccf_dict = {
            'files'  : arve_inst.data.spec['files'],
            'times'  : arve_inst.data.time['time_val'],
            'v_grid' : arve_inst.data.ccf['ccf_vrad'].copy(),
            'obo_ccf'   : arve_inst.data.ccf['ccf_val'].copy(),
            'obo_e_ccf' : arve_inst.data.ccf['ccf_err'].copy(),
            'obo_rv'    : arve_inst.data.vrad['vrad_val_ord'].copy()*1000,
            'obo_e_rv'  : arve_inst.data.vrad['vrad_err_ord'].copy()*1000,
        }

    # The RVs might have issues because of the default velocity grid
    # This code will save the CCFs produce
    # We will combine and fit the CCFs w/ the code in ccf.py

    if len(save_file)>0:
        pickle.dump(arve_inst,open(save_file,'wb'))

    return ccf_dict

# =============================================================================
# LBL

def arveLBL(data,instrument='essp',medium_data='vac'):
    """Get ARVE line-by-line RVs for specified data
    specified either as a directory of FITS files or 
    list of wavelength, flux, and errors

    Parameters
    ----------
    data
        str  : directory containing FITs files to incorporate
        list : wvln, spec, err of spectra to CCF
    instrument : str (default: essp)
        specify data format of input files (all lowercase)
        expres, harps, harps-n, neid, and essp are all relevant options
    medium_data : str [air, vac]
        if input data given, specify if wavelengths are in air or vac

    Returns
    -------
    lbl_dict
        Info about the observations ('files','times'),
        and the LBL RVS ('obo_rv','obo_e_rv')
    """
    # Initiate ARVE object
    arve_inst = arve.ARVE()
    # Specify the target star
    arve_inst.star.target = 'Sun'
    # Get the stellar parameters
    # (used to shift the spectra by the systematic velocity, which is 0 for the Sun)
    arve_inst.star.get_stellar_parameters()
    
    # Add data by pointing to directory with spectra
    arve_inst.data.add_data(path=data, instrument=instrument,
        format='s2d', extension='fits', berv_corrected=True)
    
    # Get auxiliary data
    arve_inst.data.get_aux_data()

    # Compute reference spectrum (how lines will get determined)
    arve_inst.data.compute_spec_reference()
    
    # Compute LBL RVs
    # "CaHK" criterium excludes the Ca H&K lines and lines in their vicinity 
    arve_inst.data.compute_vrad_lbl(criteria=['crit_CaHK'])

    pickle.dump(arve_inst,open(f'./Arve/Data/250722_{instrument}_lbl_arveInst.pkl','wb'))

    num_ord = arve_inst.data.spec['N_ord']
    lbl_dict = {
        'files' : arve_inst.data.spec['files'],
        'times' : arve_inst.data.time['time_val'].copy(),
        'rv'    : arve_inst.data.vrad['vrad_val'].copy()*1000,
        'e_rv'  : arve_inst.data.vrad['vrad_err'].copy()*1000,
        'lbl_rv'   : [arve_inst.data.vrad['vrad_val_lbl'][nord][:,:,0].T for nord in range(num_ord)],
        'lbl_e_rv' : [arve_inst.data.vrad['vrad_err_lbl'][nord][:,:,0].T for nord in range(num_ord)],
        'lines' : arve_inst.data.aux_data['mask'], # dictionary of DFs
        'tmp_wave' : arve_inst.data.aux_data['spec']['wave'], # I'm honestly not sure what these are
        'tmp_spec' : arve_inst.data.aux_data['spec']['flux'],
        'ref_wave' : arve_inst.data.spec_reference['wave_val'], # I think this is used to determine the lines?
        'ref_spec' : arve_inst.data.spec_reference['flux_val'],
    }
    if 'vrad_mask_lbl' in arve_inst.data.vrad.keys():
        lbl_dict['outlier_mask'] = arve_inst.data.vrad["vrad_mask_lbl"][nord][:,:,0].T
    
    return lbl_dict