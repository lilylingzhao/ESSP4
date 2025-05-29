# Selecting and Standardizing Data
import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.constants import c
import pandas as pd
from utils import *

# =============================================================================
# File Selection

airm_cutoff = lambda inst: 1.8 if inst == "harps" else 1.4

def downSelectTimes(data_df, num_obs=3, airmass_cutoff=None, day_spread_max=2):
    """Select a subset of solar observations from full timeseries

    Parameters
    ----------
    data_df : pandas.DataFrame
        data frame of the solar data set to be downselected
        presumed to be in the standard format of the ESSP 3 data release
    num_obs : int
        desired number of observations per day
    airmass_cutoff : number
        ignore observations with greater airmass than this number
        default (defined in airm_cutoff) is 1.4, 1.8 for HARPS
    day_spread_max : number [hours]
        stop considering observations this many hours away from the focal point

    Returns
    -------
    obs_mask : list (bool)
        mask of selected observations (corresponding to data_df dimensions)
    """
    # Gather Needed Info
    times, airms = data_df.loc[:, ["Time [MJD]", "Airmass"]].to_numpy().T

    # Make cut in airmass
    if airmass_cutoff is None:
        inst = fileName2Inst(data_df["File Name"].to_numpy()[0])
        airmass_cutoff = airm_cutoff(inst)
    airm_mask = airms < airmass_cutoff

    # Determine Unique Days with Good Observations
    tint = times.astype(int)
    unq_dint = np.unique(tint[airm_mask])

    # For each day
    obs_mask = np.zeros(len(data_df), dtype=bool)
    for iday, dint in enumerate(unq_dint):
        day_mask = tint == dint
        # Pick an observation with high airmass at random
        focal_time = np.random.choice(times[day_mask][airm_mask[day_mask]])
        # Find the num_obs closest observations
        closest_indices = np.argsort(abs(times - focal_time))[:int(num_obs)]
        day_times = times[closest_indices]
        # Remove any observations further away than the specified max spread
        close_enough = abs(day_times - focal_time) < (day_spread_max / 24.0)
        obs_mask[closest_indices[close_enough]] = True

    return obs_mask

def getFilesToAverage(data_df, target_expt=5.4*60, num_obs=3, airmass_cutoff=None, day_spread_max=2):
    """Define files to average to hit a target exposure time

    Parameters
    ----------
    data_df : pandas.DataFrame
        data frame of the solar data set to be downselected
        presumed to be in the standard format of the ESSP 3 data release
    target_expt : float [sec] (default: solar nu_max)
        target exposure time; observations will be averaged to
        this exposure time or above
    num_obs : int
        desired number of observations per day
    airmass_cutoff : number
        ignore observations with greater airmass than this number
        default (defined in airm_cutoff) is 1.4, 1.8 for HARPS
    day_spread_max : number [hours]
        stop considering observations this many hours away from the focal point

    Returns
    -------
    obs_list : list (int)
        files to average into single observations, specified by observation number
        i.e. averaging the first and second half of four observations would return
        [1,1,2,2]
    """
    # Gather Needed Info
    target_expt_days = target_expt/60/60/24
    times, airms, expts = data_df.loc[:, ["Time [MJD]", "Airmass","Exp. Time [s]"]].to_numpy().T

    # Make cut in airmass
    if airmass_cutoff is None:
        inst = fileName2Inst(data_df["File Name"][0])
        airmass_cutoff = airm_cutoff(inst)
    airm_mask = airms < airmass_cutoff

    # Determine Unique Days with Good Observations
    tint = times.astype(int)
    unq_dint = np.unique(tint[airm_mask])

    # For each day
    obs_list = np.zeros(len(data_df), dtype=int)
    nobs = 0
    for iday, dint in enumerate(unq_dint):
        day_mask = tint == dint
        # Pick an observation time with high airmass at random
        focal_time = np.random.choice(times[day_mask][airm_mask[day_mask]])
        # Back up the time by half of num_obs x target_expt
        focal_time -= (num_obs*target_expt_days)/2
        # Starting adding observation based on this time
        iobs0 = np.argmin(np.abs(times-focal_time))
        iobs1 = iobs0
        for _ in range(num_obs):
            time0, time1 = times[iobs0], times[iobs1]
            while (time1-time0)<(target_expt_days) and iobs1<(len(times)-1):
                iobs1 += 1
                time1 = times[iobs1]
            # Label these observations as part of the next observation
            nobs += 1
            obs_list[iobs0:iobs1] = nobs
            # Reset
            iobs0 = iobs1
            
            # Stop if we're hitting any observations further
            #     away than the specified max spread
            if (time1-time0) > (day_spread_max / 24.0):
                break

    return obs_list

def getExpresNeidTimes(df_dict,num_obs,closest_neid=True,
                       valid_mode=None,num_valid=None,
                       **kwargs):
    """Define files to average to hit a target exposure time

    Parameters
    ----------
    df_dict : dictionary of pandas.DataFrame objects
        must at least contain keys for 'expres' and 'neid'
        values should be data frame of the solar data set to be downselected
        presumed to be in the standard format of the ESSP 3 data release
    closest_neid : bool (default: True)
        if True, return the closest NEID observation to each EXPRES observation
        if False, return the average of all NEID observations that fall within
            the time span of an EXPRES observation 
    valid_mode/num_valid : varied
        arguments for selecting validation set
        see `selectValidationSet` for more info
    kwargs : varied
        additional arguments for `downSelectTimes`

    Returns
    -------
    obs_df, vld_df : pandas.DataFrame
        Data frame specifying the training and validation sets
        Is this...what I want to return?
    """

    # EXPRES observations are longer so we'll anchor NEID observations to those
    expres_df = df_dict['expres']
    expres_df['Instrument'] = 'expres'
    obs_mask = downSelectTimes(expres_df,num_obs=num_obs,**kwargs)
    if valid_mode is not None:
        train_mask = selectValidationSet(expres_df[obs_mask], valid_mode, num_valid)
    else:
        train_mask = np.ones_like(np.sum(obs_mask),dtype=bool)
    expres_obs_df = expres_df[obs_mask].reset_index()
    expres_obs_df['Training'] = train_mask
    e_time, e_expt, e_train = expres_obs_df.loc[:, ["Time [MJD]", "Exp. Time [s]", "Training"]].to_numpy().T

    neid_df = df_dict['neid']
    neid_df['Instrument'] = 'neid'
    neid_df['Training'] = False
    n_time = neid_df['Time [MJD]']
    n_list = np.zeros(len(neid_df),dtype=int)
    n_train = np.zeros(len(neid_df),dtype=bool)
    for it,t in enumerate(e_time): # Match NEID observations to EXPRES Time stamps
        t_diff = np.abs(n_time-t)
        expt_days = e_expt[it]/60/60/24 # convert exposure time to units of day
        if closest_neid:
            n_list[np.argmin(t_diff)] = True
            n_train[np.argmin(t_diff)] = e_train[it]
        else:
            # This will play into the averaging of observations later, which we are not currently prepared to deal with
            n_list[t_diff<(expt_days/2)] = it
            n_train[t_diff<(expt_days/2)] = e_train[it]
    neid_df['Training'] = n_train
    
    if closest_neid:
        n_mask = n_list.astype(bool)
        neid_obs_df = neid_df[n_list.astype(bool)].reset_index()
    else:
        print('Missing Code still')
        return
        neid_obs_df = averageFiles(neid_df,n_list)
    obs_df = pd.concat([expres_obs_df,neid_obs_df]).sort_values(by='Time [MJD]').reset_index()

    return obs_df

def selectValidationSet(data_df, mode, validation_amount):
    """Define validation observations from specified observations

    Parameters
    ----------
    data_df : pandas.DataFrame
        data frame of the solar data set to be downselected
        presumed to be in the standard format of the ESSP 3 data release
    mode : str ['NperDay','NthDay','Nweek','Npercent','NthMinute]
        specify the type of validation separation
        Npercent : random drop N percent of observations
        NperDay : pick N observations per day as validation observations
        Nperiodic : drop observations on a periodicity of N w/ units of days
                    within M time (validation_amount takes the form (N,M))
            e.g., (3,1) drops every obs within a day of every 3rd day
                  (5.4/60/24,1/60/24) drops obs every v_max within a minute
        (note: N is specified by validation_amount)
    validation_amount : float
        specify whatever quantity is relevant for the chosen mode

    Returns
    -------
    train_mask, valid_mask : boolean list
        mask of training/validation (corresponding to data_df dimensions)
    """
    times = data_df["Time [MJD]"].to_numpy().T
    num_obs = len(times)

    # Isolate validation points from the selected observations
    train_mask = np.zeros(num_obs, dtype=bool)
    obs_indx = np.arange(num_obs)
    if mode.lower() == "nperday":
        tint = times.astype(int)
        unq_day, day_count = np.unique(tint, return_counts=True)
        num_training = (np.max(day_count) - validation_amount)  # we want to keep this constant instead of valid
        for iday, dint in enumerate(unq_day):
            day_mask = tint == dint
            if day_count[iday]<=num_training:
                train_mask[obs_indx[day_mask]] = True
            else:
                # Separate out validation points
                day_train_indices = np.random.choice(np.arange(np.sum(day_mask),dtype=int),num_training,replace=False)
                train_mask[obs_indx[day_mask][day_train_indices]] = True
    elif mode.lower() == "nperiodic":
        period, width = validation_amount
        drop_times = np.arange(min(times), max(times), period)
        train_mask = np.array([min(abs(t - drop_times)) > width/2 for t in times])
    else:  # assuming Npercent
        assert (validation_amount > 0) & (validation_amount < 1)
        train_mask[np.random.choice(np.arange(num_obs),int(np.ceil(num_obs*(1-validation_amount))),replace=False)] = True
    
    return train_mask


# =============================================================================
# Spectral File Standardization

def readL2(file_name,pad_orders=False):
    """Return spectrum from any solar observation file

    Parameters
    ----------
    file_name : str
        name of file to open
    pad_orders : bool
        if true, standardize order index and echelle order
        (i.e. artificially make order 0 correspond to echelle order 161 for all instruments)

    Returns
    -------
    wave, spec, errs, blaz : array, floats
       wavelength, flux, error, and blaze of spectrum from file_name
        
    """
    inst = fileName2Inst(file_name)
    if os.path.basename(file_name)==file_name:
        # Change from file name to full file pathe
        inst_dir = instrument_nickname2Fullname(inst)
        if inst in ['harps','harpsn'] and 'BLAZE' in file_name:
            inst_dir += '_wBlaze'
        file_name = os.path.join(solar_dir,inst_dir,file_name)
    
    if inst in ['harps','harpsn','harpn','harps-n']: # HARPS or HARPS-North
        
        # Get HDUS w/ and w/o Blaze
        if '_wBlaze' in file_name:
            hdus = fits.open(getHarpsNoBlazeFile(file_name))
            blaz_hdus = fits.open(file_name)
        else:
            hdus = fits.open(file_name)
            blaz_hdus = fits.open(getHarpsBlazeFile(file_name))
        
        wave = hdus['wavedata_vac_bary'].data.copy()
        spec = blaz_hdus['scidata'].data.copy()
        spec_woB = hdus['scidata'].data.copy()
        errs = blaz_hdus['errdata'].data.copy()
        
        blaz = spec/spec_woB
        
        blaz_hdus.close()
        hdus.close()
    elif inst=='neid': # NEID
        hdus = fits.open(file_name)
        # The first 12 and last 4 orders of the NEID data are never any good
        # I'm going to remove an additional 13 orders from the red
        #     they're so ravaged by tellurics or low SNR it's just not worth it
        wave = hdus['SCIWAVE'].data[12:-17].copy()
        # Correct each order for the barycentric correction
        for iord in range(len(wave)):
            nord = iord+12
            real_nord = str(173-nord) if nord<=73 else '0'+str(173-nord)
            berv_ms = hdus[0].header[f'SSBRV{real_nord}']*1000 # m/s
            wave[iord] /= np.exp(np.arctanh(-berv_ms/c.value))
        blaz = hdus['SCIBLAZE'].data[12:-17].copy()
        spec = hdus['SCIFLUX'].data[12:-17].copy()/blaz
        errs = np.sqrt(hdus['SCIVAR'].data[12:-17].copy())
        hdus.close()
    else: # EXPRES Pipeline Format
        hdus = fits.open(file_name)
        wave = hdus[1].data['bary_wavelength'].copy()
        cont = hdus[1].data['continuum'].copy()
        blaz = hdus[1].data['blaze'].copy()
        spec = hdus[1].data['spectrum'].copy()
        errs = hdus[1].data['uncertainty'].copy()*blaz
        wave[np.isnan(spec)] = np.nan
        #tell = hdus[1].data['tellurics'].copy()
        hdus.close()
    
    if pad_orders:
        wave, spec, errs, blaz = [padOrders(drp_arr.copy(),inst) for drp_arr in [wave,spec,errs,blaz]]
    
    return wave, spec, errs, blaz

# Keyword mapping
key_map_df = pd.read_csv(os.path.join(essp4_dir,'header_map.csv')).set_index('keyword')
# Functions for adjusting keywords
def dms2deg(dms):
    d,m,s,direction = dms.split(' ')
    deg = float(d) + float(m)/60 + float(s)/(60*60);
    if direction=='E' or direction=='S':
        deg *= -1
    return deg
key_funcs = {
    'data-set' : (['all'],
                  lambda file_name : os.path.basename(file_name).split('.')[0]),
    'date'     : (['all'],
                  lambda file_name : Time.now().fits),
    'mjd_utc'  : (['harps','harpsn','neid'],
                  lambda jd : Time(jd,format='jd').mjd),
    'jd_utc'   : (['expres'],
                  lambda mjd : Time(mjd,format='mjd').jd),
    'obslon'   : (['harpsn'],
                  lambda geolon : dms2deg(geolon)),
    'obslat'   : (['harpsn'],
                  lambda geolat : dms2deg(geolat)),
}

def standardizeHeader(file_name,standard_name):
    """Read header and change to standardized version

    Parameters
    ----------
    file_name : str
        name of file to open

    Returns
    -------
    head : standardized header
        
    """
    head = fits.Header()
    inst = fileName2Inst(file_name)
    inst_idx, inst_key = f'{inst}_idx', f'{inst}_key'
    
    hdus = fits.open(file_name)
    for key in key_map_df.index:
        if key=='file_name':
            hdus[key] = (file_name,'Name of the FITS file')
            continue
        
        key_inst_idx = int(key_map_df.at[key,inst_idx])
        key_inst_key = key_map_df.at[key,inst_key]
        if key_inst_idx == -1: # that means we entered a static value
            value = key_inst_key
        elif key_inst_key=='file_name':
            value = standard_name
        elif inst.lower()=='neid' and key.lower()=='berv':
            # NEID BERV is a special case
            # Need to average over all the order-by-order values
            ord_list = range(52,174)
            value = np.nanmedian([hdus[0].header[f'SSBRV0{nord}' if nord<100 else f'SSBRV{nord}'] for nord in ord_list])
        else:
            value = hdus[key_inst_idx].header[key_inst_key]
        
        # If value needs to be adjusted somehow
        if key in key_funcs.keys():
            inst_list, kfunc = key_funcs[key]
            if (inst in inst_list) or ('all' in inst_list):
                value = kfunc(value)
        
        # Save value to header dictionary
        head[key] = (value,key_map_df.at[key,'Comment'])
    hdus.close()
    
    return head

# =============================================================================
# CCF File Standardization

def readCCF(file_name,standard=False):
    """Return CCF from any solar observation file

    Parameters
    ----------
    file_name : str
        name of file to open
    
    WE NEED A NEW FUNCTION TO PAD CCF ORDERS
    pad_orders : bool
        if true, standardize order index and echelle order
        (i.e. artificially make order 0 correspond to echelle order 161 for all instruments)

    Returns
    -------
    wave, spec, errs, blaz : array, floats
       wavelength, flux, error, and blaze of spectrum from file_name
        
    """
    inst = 'expres' if standard else fileName2Inst(file_name)
      
    hdus = fits.open(file_name)
    if standard:
        time_mjd = hdus[0].header['time']
        ccf_x = hdus['v_grid'].data.copy()
        ccf_y = hdus['ccf'].data.copy()
        ccf_e = hdus['e_ccf'].data.copy()
        ccf_rv = hdus[0].header['rv']
        ccf_rv_e = hdus[0].header['e_rv']
        # Order-By-Order
        ccf_obo_y = hdus['obo_ccf'].data.copy()
        ccf_obo_e = hdus['obo_e_ccf'].data.copy()
        ccf_obo_rv = hdus['obo_rv'].data.copy()
        ccf_obo_rv_e = hdus['obo_e_rv'].data.copy()
        ccf_obo_o = hdus['echelle_orders'].data.copy()
    elif inst == 'harps': # HARPS
        time_mjd = hdus[0].header['MJD-OBS']
        num_ord, num_vel = hdus[0].data.shape
        v0 = hdus[0].header['CRVAL1']
        vstep = hdus[0].header['CDELT1'] # km/s
        ccf_x = np.arange(v0,v0+vstep*num_vel,vstep)
        ccf_y = np.sum(hdus[0].data.copy(),axis=0)
        ccf_e = np.full_like(ccf_y,np.nan)
        #ccf_rv = hdus [0].header['HIERARCH ESO DRS CCF RV']*1000 # m/s; no drift correction
        ccf_rv = hdus [0].header['HIERARCH ESO DRS CCF RVC']*1000 # m/s; drift corrected
        ccf_rv_e = hdus[0].header['HIERARCH ESO DRS DVRMS'] # m/s
        # Order-By-Order
        ccf_obo_y = hdus[0].data.copy()
        ccf_obo_e = np.sqrt(ccf_obo_y)
        ccf_obo_rv = np.full(num_ord,np.nan)
        ccf_obo_rv_e = np.full(num_ord,np.nan)
        ccf_obo_o = 160-np.delete(np.arange(num_ord+1),45)
    elif inst in ['harpsn','harpn','harps-n']: # HARPS-North
        time_mjd = Time(hdus[0].header['DATE-OBS']).mjd
        num_ord, num_vel = hdus[1].data.shape
        v0 = hdus[0].header['HIERARCH TNG RV START']
        vstep = hdus[0].header['HIERARCH TNG RV STEP'] # probably also km/s?
        ccf_x = np.arange(v0,v0+vstep*num_vel,vstep)
        # Apply a flux corection?: HIERARCH TNG QC ORDER[1-69] FLUX CORR
        ccf_y = np.sum(hdus[1].data.copy(),axis=0)
        ccf_e = np.sqrt(np.sum(hdus[2].data.copy()**2,axis=0))
        ccf_rv = hdus[0].header['HIERARCH TNG QC CCF RV']*1000 # km/s -> m/s
        ccf_rv_e = hdus[0].header['HIERARCH TNG QC CCF RV ERROR'] # m/s?
        # Order-By-Order
        ccf_obo_y = hdus[1].data.copy()
        ccf_obo_e = hdus[2].data.copy()
        ccf_obo_rv = np.full(num_ord,np.nan)
        ccf_obo_rv_e = np.full(num_ord,np.nan)
        ccf_obo_o = 157-np.arange(num_ord)
    elif inst=='neid': # NEID
        # The first 12 and last 4 orders of the NEID data are never any good
        try:
            time_mjd = Time(hdus[0].header['DATE-OBS']).mjd
        except:
            time_mjd = np.nan
        ccf_head, ccf_data = hdus['ccfs'].header, hdus['ccfs'].data[12:-4].copy()
        num_ord, num_vel = ccf_data.shape
        v0 = ccf_head['CCFSTART']
        vstep = ccf_head['CCFSTEP'] # km/s? 
        ccf_x = np.arange(v0,v0+vstep*num_vel,vstep)
        # Apply a flux corection?: CCFWT057=                  0.0 / Flux weighting factor for order 57 
        # Apply a flux corection?: SCWT057 =  / Solar spectrum weighting factor for order 57
        ccf_y = np.sum(ccf_data,axis=0)
        ccf_e = np.full_like(ccf_y,np.nan)
        ccf_rv = ccf_head['CCFRVSUM'] # m/s
        ccf_rv_e = ccf_head['DVRMSSUM']*1000 # km/s -> m/s; Estimated RV uncertainty for all orders (km/s)
        #ccf_rv_e = hdus[0].header['DVRMSMOD']*1000 # km/s -> m/s; Estimated RV uncertainty for weighted orders (km/s)
        # Order-By-Order
        ccf_obo_y = ccf_data.copy()
        ccf_obo_e = np.full_like(ccf_obo_y,np.nan)
        ccf_obo_o = 161-np.arange(num_ord)
        ccf_obo_rv = np.full(num_ord,np.nan)
        key_list = list(hdus[12].header.keys())
        for iord,nord in enumerate(ccf_obo_o):
            nord = str(nord)
            while len(nord)<3:
                nord = '0'+nord
            key = f'CCRV{nord}'
            if key in key_list:
                ccf_obo_rv[iord] = hdus[12].header[key]*1000
        ccf_obo_rv_e = np.full(num_ord,np.nan)
    else: # EXPRES Pipeline Format
        time_mjd = hdus[0].header['MJD']
        ccf_x = hdus[1].data['V_grid'].copy()/100/1000
        ccf_y = hdus[1].data['ccf'].copy()
        ccf_e = hdus[1].data['e_ccf'].copy()
        ccf_rv = hdus[0].header['V']/100
        ccf_rv_e = hdus[0].header['E_V']/100
        # Order-By-Order
        ccf_obo_y = hdus[2].data['ccfs'].copy()
        ccf_obo_e = hdus[2].data['errs'].copy()
        ccf_obo_rv = hdus[2].data['v'].copy()/100
        ccf_obo_rv_e = hdus[2].data['e_v'].copy()/100
        ccf_obo_o = hdus[2].data['orders'].copy()
    hdus.close()
    
    ccf_dict = {
        'time'   : time_mjd,
        'v_grid' : ccf_x.copy(),
        'ccf'    : ccf_y.copy(),
        'e_ccf'  : ccf_e.copy(),
        'rv'     : ccf_rv,
        'e_rv'   : ccf_rv_e,
        'orders'   : ccf_obo_o.copy(),
        'obo_ccf'    : ccf_obo_y.copy(),
        'obo_e_ccf'  : ccf_obo_e.copy(),
        'obo_rv'     : ccf_obo_rv.copy(),
        'obo_e_rv'   : ccf_obo_rv_e.copy(),
    }
    
    return ccf_dict
