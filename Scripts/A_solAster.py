# Gather Needed SDO Files/Values for ESSP

import os
from glob import glob
from multiprocessing import Pool, cpu_count

import sys
sys.path.append('../')
from utils import sdo_dir, mon_min, mon_max
from sdoobs import *

overwrite = False

leadingZeros = lambda val : ('0'+str(val))[-2:]

save_file = os.path.join(sdo_dir,'260422_sdoValues.csv')
save_df = pd.read_csv(save_file) if os.path.isfile(save_file) else None

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
    
    ### Download Needed Data
    if overwrite: # Re-download All
        downloadDay(f'21{imon}{iday}',save_dir=sdo_dir,
                    cadence=12*u.minute,overwrite=overwrite,
                    e_mail='lilylingzhao@uchicago.edu')
    else:
        # Check if any files are already downloaded/processed
        num_sdo_file = len(glob(getSdoFileName(ymd,'*',type_list[-1],full_path=True,sdo_dir=sdo_dir)))
        if (num_processed+num_sdo_file) < tot_nobs_perDay:
            # There are files that are not (1) processed or (2) downloaded
            downloadDay(f'21{imon}{iday}',save_dir=sdo_dir,
                        cadence=12*u.minute,overwrite=overwrite,
                        skip_tstamps=save_df['tstamp'],
                        e_mail='lilylingzhao@uchicago.edu')
    
    # Get Values Across Day
    imap_list = glob(os.path.join(sdo_dir,'Flatcont',f'{ymd}.*_flatcont.fits'))
    tstamp_list = [os.path.basename(f)[7:13] for f in imap_list]
    for t in tstamp_list:
        if save_df is not None and f'{ymd}.{t}' in save_df['obs_name'] and not overwrite:
            # These observations have already been processed
            # Add a check for NaN values here? (I don't think it's necessary)

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

        # Delete Files if Values are Saved
        for ftype in type_list:
            os.remove(getSdoFileName(ymd,t,ftype,full_path=True))

def poolingFunc(mjd):
    yy,m,d = Time(mjd,format='mjd').isot.split('T')[0].split('-')
    ymd = yy[2:]+m+d
    processDay(ymd)

if __name__=='__main__':
    mjd_list = list(range(int(mon_min-2),int(mon_max+3)))
    n_workers = min(cpu_count(),8)
    with Pool(processes=n_workers) as pool:
        pool.map(poolingFunc, mjd_list)