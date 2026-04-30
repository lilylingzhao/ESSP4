# Gather Needed SDO Files/Values for ESSP

import os
from glob import glob
from multiprocessing import Pool, cpu_count

import sys
sys.path.append('../')
from utils import sdo_dir, mon_min, mon_max
from sdoobs import *

overwrite = False

# How to check if a day is fullly downloaded/processed
# (change if what we're downloading changes)
tot_nobs_perDay = 120
last_tstamp = 234800

def processDay(ymd,overwrite=False):
    save_file = os.path.join(sdo_dir,f'{ymd}_sdoValues.csv')
    
    ### Check if day is already processed
    if os.path.isfile(save_file) and not overwrite:
        save_df = pd.read_csv(save_file)
        num_processed = len(save_df)
        if num_processed==tot_nobs_perDay:
            # All obs for day processed, exit
            return
    else:
        save_df = None
        num_processed = 0
    
    ### Get Values Across Day
    imap_list = glob(os.path.join(sdo_dir,'Flatcont',f'{ymd}.*_flatcont.fits'))
    tstamp_list = [os.path.basename(f)[7:13] for f in imap_list]
    for t in tstamp_list:
        # Check if this time stamp has already been processed
        if save_df is not None and f'{ymd}.{t}' in save_df['obs_name'] and not overwrite:
            # check that the files are deleted
            t_file_list = glob(os.path.join(sdo_dir,'*',f'{ymd}.*_*.fits'))
            if len(t_file_list)>0:
                for file in t_file_list:
                    os.remove(file)
            continue

        # Generate SDO_Obs Object
        sdo_obs = SDO_Obs(ymd,t)
        sdo_obs.getSolAsterValues()
        sdo_obs.getHaywoodValues()

        # Save Values
        save_df = sdo_obs.save(save_file,existing_df=save_df,
                               save_dir=sdo_dir,save_regions=True)
        print(f'({len(save_df)}/{len(tstamp_list)}) {save_df.iloc[-1]["obs_name"]} saved at {Time.now().isot}')

        # Delete Files after Values are Saved
        for ftype in type_list:
            os.remove(getSdoFileName(ymd,t,ftype,full_path=True))

def poolingFunc(mjd):
    ymd = mjd2ymd(mjd)
    print('-'*10,f'PROCESSING {ymd}',Time.now().isot,'-'*10)
    processDay(ymd)
    print('='*10,f'ENDING {ymd}',Time.now().isot,'='*10)

if __name__=='__main__':
    # Run Through MJDs w/ Downloaded Files
    file_list = glob(os.path.join(sdo_dir,'Flatcont','21*.2*_flatcont.fits'))
    # Continue checking for new downloaded files
    # (This isn't the most efficient, but is the best
    #  way w/o requring all files are downloaded before
    #  any processing is done)
    while len(file_list)>0:
        ### Understand What Days Still Need to Be Processed
        # Collect ymd of all downloaded files
        ymd_list = np.unique([os.path.basename(f).split('.')[0] for f in file_list])
        mjd_list = []
        for ymd in ymd_list: # For each day
            save_file = os.path.join(sdo_dir,f'{ymd}_sdoValues.csv')
            # If the save file doesn't exist, definitely process it
            if not os.path.isfile(save_file):
                mjd_list.append(ymd2mjd(ymd))
            else:
                # If the save file has less than the expected number of rows, process it
                if len(pd.read_csv(save_file))<tot_nobs_perDay:
                    mjd_list.append(ymd2mjd(ymd))
        
        print('-'*20,'YMD LIST','-'*20)
        print(ymd_list)
        print('-'*40)
        
        n_workers = min(cpu_count()-2,6)
        with Pool(processes=n_workers) as pool:
            pool.map(poolingFunc, mjd_list)

        # Get most up to date list of downloaded files
        file_list = glob(os.path.join(sdo_dir,'Flatcont','21*.23*_flatcont.fits'))
        