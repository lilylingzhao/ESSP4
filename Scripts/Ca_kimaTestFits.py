import os
from glob import glob
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
from utils import *
from kimafit import *

overwrite = False

local_kima_dir = '/Users/lilyzhao/Documents/ceph/ESSP/ESSP4/Signals/KimaFits'

dset_list = range(1,num_dset+1)
script_dir = os.getcwd()

# Directory for temporary, kima-formated data files
scratch_dir = os.path.join(kima_dir,'Scratch_Ca')
os.makedirs(scratch_dir, exist_ok=True)

# (There's probably a smarter way to do this)
dir_names = ['ErrLevel','ErrDRP','RedNoise']
for save_dir in [kima_dir,local_kima_dir]:
    for dname in dir_names:
        final_dir = os.path.join(save_dir,dname)
        if not os.path.isdir(final_dir):
            os.makedirs(scratch_dir)
    

# =============================================================================
# White Noise Tests

# =====================================
# Error Levels

errs_list = np.round(np.arange(0.1,0.36,0.05),2)
for ierr, err in enumerate(errs_list):
    err_int = int(err*100)
    for dset in dset_list:
        save_file = os.path.join(kima_dir,'ErrLevel',f'DS{dset}_e{err_int}.csv')
        if os.path.isfile(save_file) and not overwrite:
            continue

        data_file = genKimaTestData(dset,kima_data_dir=scratch_dir,
                                    err=err,rand_seed=dset)

        os.chdir(scratch_dir)
        model, res = kimaFit(data_file,max_npl=3,
                             trend_deg=kima_trend_dict[dset],
                             t_baseline=1,
                             steps=100_000,num_threads=4,print_thin=2e4)
        df = posteriorSampleDataFrame(os.path.join(scratch_dir,'posterior_sample.txt'),
                                      save_file=save_file)
        df.to_csv(os.path.join(local_kima_dir,'ErrLevel',f'DS{dset}_e{err_int}.csv'),index=False)
        os.chdir(script_dir)

# =====================================
# DRP Noise
# (Only doing one realization of wn,
#  can show the posteriors of all wn realizations
#  are similar at referee request?)

for dset in dset_list:
    save_file = os.path.join(kima_dir,'ErrDRP',f'DS{dset}_drp.csv')
    if os.path.isfile(save_file) and not overwrite:
        continue

    data_file = genKimaTestData(dset,kima_data_dir=scratch_dir,
                                err=None,rand_seed=dset)

    os.chdir(scratch_dir)
    model, res = kimaFit(data_file,max_npl=3,
                         trend_deg=kima_trend_dict[dset],
                         t_baseline=1,
                         steps=100_000,num_threads=4,print_thin=2e4)
    df = posteriorSampleDataFrame(os.path.join(scratch_dir,'posterior_sample.txt'),
                                  save_file=save_file)
    df.to_csv(os.path.join(local_kima_dir,'ErrDRP',f'DS{dset}_drp.csv'),index=False)
    os.chdir(script_dir)

# =============================================================================
# Red Noise Tests
# CSV files of all CCF RVs generated in `250525_rvComparison.ipynb`

# essp4 - distributed data (w/ planets)
# drp - original drp RVs
# iCCF - iCCF on original spectra
# essp - full ESSP pipeline, but no planets (a better version of iCCF, it just doesn't work good yet)
test_name_list = ['essp4','drp','iCCF','essp']

for dset in dset_list:
    v_key_list = [f'ds{dset}','RV [m/s]']#,'iCCF']#,'ESSP']
    e_key_list = [f'e_ds{dset}','RV Err. [m/s]']#,'e_iCCF']#,'e_ESSP']
    for ikey,(v_key, e_key) in enumerate(zip(v_key_list,e_key_list)):
        save_file = os.path.join(kima_dir,'RedNoise',f'DS{dset}_{test_name_list[ikey]}.csv')
        if os.path.isfile(save_file) and not overwrite:
            continue
        
        dset_file = os.path.join(essp_dir,'CCFs',f'DS{dset}_allCCFs.csv')
        file_list = genKimaDataFiles(dset_file,save_dir=scratch_dir,
                                     t_key='Time [MJD]',v_key=v_key,e_key=e_key,
                                     separate_insts=True)

        os.chdir(scratch_dir)
        model, res = kimaFit(file_list,max_npl=3,
                             trend_deg=kima_trend_dict[dset],
                             t_baseline=1,
                             steps=100_000,num_threads=4,print_thin=2e4)
        df = posteriorSampleDataFrame(os.path.join(scratch_dir,'posterior_sample.txt'),
                                      save_file=save_file)
        df.to_csv(os.path.join(local_kima_dir,'RedNoise',f'DS{dset}_{test_name_list[ikey]}.csv'),index=False)
        os.chdir(script_dir)