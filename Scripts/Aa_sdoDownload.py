# Download SDO Files

import os
from glob import glob

import sys
sys.path.append('../')
from utils import sdo_dir, mon_min, mon_max
from sdoobs import *

mjd_list = list(range(int(mon_min-2),int(mon_max+3)))

for mjd in mjd_list:
    ymd = mjd2ymd(mjd)

    ### Check if day has already been processed
    save_file = os.path.join(sdo_dir,f'{ymd}_sdoValues.csv')
    if os.path.isfile(save_file):
        continue
    
    ### Check if we already have 120 files from this day
    # Flatcont is the last of the four file types to be downloaded
    if len(glob(os.path.join(sdo_dir,'Flatcont',f'{ymd}.*_flatcont.fits')))==120:
        continue

    ### Download all relevant 
    downloadObs(ymd,save_dir=sdo_dir,
                cadence=12*u.minute,
                e_mail=sdo_email)