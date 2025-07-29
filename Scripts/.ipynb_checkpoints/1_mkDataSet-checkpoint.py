# Generate Data Set with Given Parameters
import argparse
from tqdm import tqdm

import sys
sys.path.append('/Users/lilyzhao/Documents/Employment/ESSP/4SolarTests/ESSP4/')
from utils import solar_dir, essp_dir, instruments, mon_min, offset_dict
from data import *
from solarContinuum import solarCont
from planetInjection import getRvTimeSeries, injectPlanet

def main():
    
    parser = argparse.ArgumentParser(description='Generate an ESSP IV Data Set')
    
    parser.add_argument('-d','--data-set',type=int,
                        help='Number of the data set to use in saved files.')
    
    # Selecting observations
    parser.add_argument('-n','--num-obs',type=int,default=3,
                        help='Number of observations to select per night')
    parser.add_argument('-o','--num-day',type=int,default=60,
                        help='Number of days to select per data set')
    parser.add_argument('-x','--target-expt',type=float,default=0,
                        help='Target exposure time when identifying observations to average')

    # Isolating validation observations
    parser.add_argument('-v','--validation-mode',type=str,default='',
                        help='Method for determining validation observations')
    parser.add_argument('--validation-amount',type=int,default=0, #!!! won't work for Nperiodic !!!
                        help='Amount(s) relevant to how validation observations will be determined')
    
    # Injecting planets
    parser.add_argument('-planet-file',type=str,default='',
                        help='File containing planet parameters to be injected into the data set')
    
    args = parser.parse_args()
    
    # Define Data Set Name
    data_set_num = int(args.data_set)
    data_set_name = f'DS{data_set_num}'
    
    # Set Random Seed for Data Set
    np.random.seed(data_set_num)
    
    # Generate random time offset for data set
    time0 = Time('2021-04-27').mjd+np.random.rand()
    
    ### Select Observations for Data Set
    ds_df = getObs(data_set_name,
                   num_obs=int(args.num_obs),target_expt=float(args.target_expt),
                   validation_mode=args.validation_mode,
                   validation_amount=int(args.validation_amount),
                   time0=time0)
    
    ### Make Directory for Data Set if it DNE
    dset_dir = os.path.join(essp_dir,'Training',data_set_name)
    dir_list = ['Spectra','CCFs']
    if not os.path.isdir(dset_dir):
        for dir_name in dir_list:
            os.makedirs(os.path.join(dset_dir,dir_name))
    if np.sum(~ds_df['Training'])>0:
        vlid_dir = os.path.join(essp_dir,'Validation',data_set_name)
        if not os.path.isdir(vlid_dir):
            for dir_name in dir_list:
                os.makedirs(os.path.join(vlid_dir,dir_name))
    
    ### Determine Requested Doppler Shift
    if len(args.planet_file)>0:
        inject_rv = getRvTimeSeries(ds_df['Time [MJD]'],args.planet_file,host_mass=1)
    else:
        inject_rv = np.zeros(len(ds_df))
    
    ### Generate and Save Standardized Files
    minax_df = pd.read_csv(os.path.join(essp4_dir,'order_wminax.csv'))
    for ifile,file in enumerate(tqdm(ds_df['File Name'],desc=data_set_name)):
        # Get Full Path for Original Files
        inst_name = ds_df.at[ifile,'Instrument']
        file_name = spec_basename2FullPath(file)
        save_file = os.path.join(dset_dir if ds_df.at[ifile,'Training'] else vlid_dir,
                                 'Spectra',ds_df.at[ifile,'Standard File Name'])
        
        # Read in original data
        wave, spec, errs, blaz = readL2(file_name)
        
        # Fit continuum
        cont = solarCont(file_name)
        
        # Determine mask of wavelengths present in all instruments
        wave = padOrders(wave,inst_name)
        common_mask = np.zeros_like(wave,dtype=bool)
        for iord in range(len(wave)):
            common_mask[iord] = (wave[iord]>=minax_df.at[iord,'Min']) & (wave[iord]<=minax_df.at[iord,'Max'])
        
        # Standardized Headers
        head = standardizeHeader(file_name,os.path.basename(save_file))
        # Artificially Offset Time
        head['mjd_utc'] -= (mon_min-time0)
        head['jd_utc']  -= (mon_min-time0)
        
        ### Assemble Spectral Data
        data_dict = {
            'wavelength'  : wave.copy(),
            'flux'        : padOrders(spec,inst_name),
            'uncertainty' : padOrders(errs,inst_name),
            'continuum'   : padOrders(cont,inst_name),
            'blaze'       : padOrders(blaz,inst_name),
            'common_mask' : common_mask.copy().astype(int),
        }
        
        ### Inject Planet
        tell_mask, data_dict = injectPlanet(inject_rv[ifile],data_dict.copy())
        
        ### Save FITS File
        hdu = fits.PrimaryHDU(data=None,header=head)
        hdu_list = [hdu]
        for key in data_dict.keys():
            hdu_list.append(fits.ImageHDU(data=data_dict[key],name=key))
        fits.HDUList(hdu_list).writeto(save_file,overwrite=True)
        tqdm.write(save_file)

def padIobs(iobs):
    # We want all iobs numbers to be three characters long
    iobs = str(iobs)
    while len(iobs)<3:
        iobs = '0'+iobs
    return iobs

def getStandardFiles(ds_df,data_set):
    files = []
    # We'll number traning and validation observations separately
    tiobs, viobs = 1, 1
    train_mask = ds_df['Training'].to_numpy()
    inst_list = ds_df['Instrument'].to_numpy()
    for i in range(len(ds_df)):
        inst = inst_list[i]
        if train_mask[i]:
            files.append(f'{data_set}.{padIobs(tiobs)}_spec_{inst}.fits')
            tiobs += 1
        else:
            files.append(f'{data_set}.{padIobs(viobs)}_v_spec_{inst}.fits')
            viobs += 1
    return files

def getObs(data_set,num_obs,num_day=None,target_expt=0,
           validation_mode='',validation_amount=None,time0=0):
    ### Select Observations for Data Set
    df_list = []
    for iinst,inst in enumerate(instruments):
        inst_df = pd.read_csv(os.path.join(solar_dir,f'{inst}_drp.csv'))
        inst_df['RV [m/s]'] = inst_df['RV [m/s]']-offset_dict[inst]
        
        # Select Observations
        if target_expt==0: # no averaging of observations needed
            obs_mask = downSelectTimes(inst_df,num_obs=num_obs)
        else:
            obs_indx = getFilesToAverage(inst_df, target_expt=args.target_expt,num_obs=tot_obs)
            #!!!!# INSERT CODE TO AVERAGE OBSERVATIONS HERE ONCE WRITTEN
            return
        
        inst_df['Instrument'] = instruments[iinst]
        inst_ds_df = inst_df[obs_mask].copy()
        
        # Define Training/Validation Sets
        if len(validation_mode)>0:
            # For periodic validation, we mask after combining data from all instruments
            if validation_mode.lower() != 'nperiodic':
                train_mask = selectValidationSet(inst_ds_df,
                                                 mode=validation_mode,
                                                 validation_amount=validation_amount)
        else:
            train_mask = np.ones(np.sum(obs_mask),dtype=bool)
        
        inst_ds_df['Training'] = train_mask.copy()
        
        df_list.append(inst_ds_df)
    
    ### Generate File Describing Data Set Files
    ds_df = pd.concat(df_list).sort_values(by='Time [MJD]').reset_index() # combine info from all four instruments
    # Down Select Days if Specified
    if num_day is not None:
        tint = ds_df['Time [MJD]'].to_numpy().astype(int)
        unq_day = np.unique(tint)
        if num_day>len(unq_day):
            # You want more days than there are,
            #     so keep them all!
            day_mask = np.ones(len(ds_df),dtype=bool)
        else:
            day_mask = np.zeros(len(ds_df),dtype=bool)
            days_to_keep = np.random.choice(unq_day,num_day)
            for day in days_to_keep:
                day_mask[tint==day] = True
        ds_df = ds_df[day_mask].reset_index()
    # Add standard file names
    ds_df['Standard File Name'] = getStandardFiles(ds_df,data_set)
    # Define Periodic T/V Set Here so Data From All Instruments are Masked in Same Way
    if (validation_mode.lower() == 'nperiodic'):
        ds_df['Training'] = selectValidationSet(inst_df[obs_mask],
                                                mode=validation_mode,
                                                validation_amount=validation_amount)
    
    for key in ['index','Quality','Start Time [MJD]']:
        if key in ds_df.columns:
            del ds_df[key]
    # Adjust Times
    if time0!=0:
        ds_df['time0'] = time0
        ds_df['Time [eMJD]'] = ds_df['Time [MJD]']-mon_min+time0
    
    # Save Summary File
    ds_df.to_csv(os.path.join(essp_dir,f'{data_set}.csv'),index=False)
    
    return ds_df

if __name__ == '__main__':
    import sys
    sys.exit(main())