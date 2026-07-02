# Fits debugged and results combined in `260608_kimaFitSubmissions.ipynb`

import os
import shutil
from glob import glob
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import sys
sys.path.append('../')
from utils import *
from kimafit import *
from kepler import getRV_K

overwrite = False

local_kima_dir = '/Users/lilyzhao/Documents/ceph/ESSP/ESSP4/Signals/KimaFits'

script_dir = os.getcwd()

# Define trend degree
kima_trend_dict = {dset:3 if dset in [4,8] else 0 for dset in range(1,10)}

# =============================================================================
# Fit Cleaned RVs For All Submissions

def getKeys(sgnl_df,meth):
    col_names = sgnl_df.columns.to_numpy()
    v_key = meth+'-RV_C'
    if v_key not in col_names:
        v_key = None
    
    e_key = meth+'-eRV_C'
    # If no errors given, default to DRP errors
    if (e_key not in col_names) or (np.sum(np.isnan(sgnl_df[e_key]))==len(sgnl_df)):
        e_key = 'DRP RV Err. [m/s]'
    return v_key, e_key

def fitSubmission(dset,meth,overwrite=overwrite):
    # Place to Save Final Result
    save_dir = os.path.join(kima_dir,f'DS{dset}')
    save_file = os.path.join(save_dir,f'DS{dset}_{meth}.csv')
    if os.path.isfile(save_file) and not overwrite:
        return
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir.replace(kima_dir,local_kima_dir))
    
    # Directory to Save Kima-Related Files
    method_scratch_dir = os.path.join(kima_dir,f'Scratch_{dset}_{meth}')
    if not os.path.isdir(method_scratch_dir):
        os.makedirs(method_scratch_dir)
    
    # Read in CSV that Collects All Results
    sgnl_df_file = os.path.join(sgnl_dir,f'DS{dset}_signals.csv')
    sgnl_df = pd.read_csv(sgnl_df_file)
    
    # Generate Kima-formated data files
    v_key, e_key = getKeys(sgnl_df,meth)
    if v_key is None:
        shutil.rmtree(method_scratch_dir)
        return
    file_list = genKimaDataFiles(sgnl_df_file,save_dir=method_scratch_dir,
                                 t_key='Time [eMJD]',v_key=v_key,e_key=e_key,
                                 separate_insts=True)
    if file_list is None or file_list==[]:
        shutil.rmtree(method_scratch_dir)
        return

    # Change to Directory to Save Kima-Related Files
    os.chdir(method_scratch_dir)
    # Run the Model
    model, res = kimaFit(file_list,max_npl=3,
                         trend_deg=kima_trend_dict[dset],
                         t_baseline=1,
                         steps=100_000,num_threads=4,print_thin=2e4)
    # Get DataFrame to Store Posteriors
    df = posteriorSampleDataFrame(os.path.join(method_scratch_dir,'posterior_sample.txt'),
                                  save_file=save_file)
    # (Save Local Copy for Convenience)
    df.to_csv(save_file.replace(kima_dir,local_kima_dir),index=False)
    
    # Change back to the base directory with the Kima Script
    os.chdir(script_dir)
    # Remove the Method Specific Kima Directory
    shutil.rmtree(method_scratch_dir)

if __name__=='__main__':
    dset_meth_list = itertools.product(dset_list,meth_list)

    n_workers = min(cpu_count()-2,6)
    with Pool(processes=n_workers) as pool:
        pool.starmap(fitSubmission, dset_meth_list)