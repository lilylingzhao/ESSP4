# Patch Missing SDO Values

import os
from glob import glob
from multiprocessing import Pool, cpu_count

import sys
sys.path.append('./ESSP4/')
from utils import sdo_dir, mon_min, mon_max
from sdoobs import *

# List of Days to Check
mjd_list = list(range(int(mon_min-2),int(mon_max+3)))
# List of Time Stamps to Check For
tstamp_list = np.concatenate([[int(h*1e4+m*1e2) for m in np.arange(0,60,12)] for h in range(24)])

# List of Files that DNE
dne_list_file = os.path.join(sdo_dir,'SDO_DNE.txt')
if os.path.isfile(dne_list_file):
    dne_files = list(np.genfromtxt(dne_list_file, dtype=str))
else:
    dne_files = []

# Download Files For Specific Time Stamp
def downloadTstamp(ymd,tstamp):
    # Narrow Down to Just Time Stamp of Interest
    tstamp = leadingZeros(tstamp)
    hr = tstamp[:2]
    mn = tstamp[2:4]
    min_hr = int(hr)+int(mn)/60
    max_hr = int(hr)+(int(mn)+1)/60

    # Querry for just that one obs
    num_files = downloadObs(ymd,save_dir=sdo_dir,
                            min_hr=min_hr,max_hr=max_hr,
                            cadence=12*u.minute)
    return num_files

# For Each Date
for mjd in mjd_list:
    ymd = mjd2ymd(mjd)
    save_file = os.path.join(sdo_dir,f'{ymd}_sdoValues.csv')
    if not os.path.isfile(save_file):
        continue

    ### Read in Already Saved Data
    save_df = pd.read_csv(save_file)
    save_df.drop_duplicates(subset=['tstamp'], keep='last', inplace=True)
    if len(save_df)==len(tstamp_list):
        # Has all time stamps we could want, continue
        continue

    ### Get List of Missing Time Stamps
    tstamp_miss = np.setdiff1d(tstamp_list,save_df['tstamp'])
    
    ### Download Files for Missing Time Stamps
    for t in tstamp_miss:
        t = leadingZeros(t)
        # Check if we already have all 4 needed files
        if len(glob(os.path.join(sdo_dir,'*',f'{ymd}.{t}*')))==4:
            continue
        # Check if we already know this observation DNE
        if f'{ymd}.{t}' in dne_files:
            continue
        
        # Download Files
        num_files = downloadTstamp(ymd,t)
        # If no files existed, add to bad list
        if 0 in num_files:
            dne_files.append(f'{ymd}.{t}')
            tstamp_miss = np.delete(tstamp_miss,np.where(tstamp_miss==t)[0][0])
    # Save updated list of observations that DNE
    np.savetxt(dne_list_file,dne_files,fmt='%s')

    ### Process Re-Downloaded Files for Day
    for t in tqdm(tstamp_miss,desc=ymd):
        
        # Generate SDO_Obs Object
        sdo_obs = SDO_Obs(ymd,t)
        sdo_obs.getSolAsterValues()
        sdo_obs.getHaywoodValues()
    
        # Save Values
        save_df = sdo_obs.save(save_file,existing_df=save_df,
                               save_dir=sdo_dir,save_regions=True)
    
        # Delete Files if Values are Saved
        for ftype in type_list:
            os.remove(getSdoFileName(ymd,t,ftype,full_path=True))
    
        # Re-order and Save CSV
        save_df.sort_values(by='date_saved',inplace=True)
        save_df.drop_duplicates(subset=['tstamp'],keep='last',inplace=True)
        save_df.to_csv(save_file,index=False)