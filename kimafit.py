# Planet Fitting Pipeline w/ Kima
# https://www.kima.science/docs/
# https://github.com/kima-org/kima
import os
from glob import glob
import numpy as np
import pandas as pd

import kima
from kima import RVData, RVmodel, distributions

from utils import *

essp_dir = '/Users/lilyzhao/Documents/ceph/ESSP_Solar/ESSP4'
kima_dir = os.path.join(essp_dir,'KimaFits')

# Specify a scratch directory to store temporary, Kima-standard data files
scratch_dir = os.path.join(kima_dir,'Scratch')
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

### STILL NEED TO FIGURE OUT SOMETHING TO DO IF ERRORS AREN'T RETURNED FOR THE SUBMITTED RVS

def getKimaDataFiles(data_file,
                     save_dir=scratch_dir,
                     submission=True,
                     separate_insts=True):
    df = pd.read_csv(data_file)
    t_key = 'Time [eMJD]'
    v_key, e_key = ['RV_C','eRV_C'] if submission else ['RV [m/s]','RV Err. [m/s]']

    if separate_insts:
        file_list = []
        for inst in instruments:
            save_file = os.path.join(save_dir,f'RVData_{inst}.rdb')
            m_inst = df['Instrument']==inst
            np.savetxt(save_file, df.loc[m_inst,[t_key,v_key,e_key]].to_numpy(),
                       header='time rv rv_err', comments='', fmt='%f %f %5.3f')
            file_list.append(save_file)
    else:
        save_file = os.path.join(save_dir,f'RVData.rdb')
        np.savetxt(save_file, df.loc[:,[t_key,v_key,e_key]].to_numpy(),
                   header='time rv rv_err', comments='', fmt='%f %f %5.3f')
        file_list = save_file        
    return file_list

def kimFit(data_file,save_file,
           max_npl=3,prior_dict={},
           steps=100_000,num_threads=8):
    file_list = getKimaDataFiles(data_file)

    # Read in Data
    D = RVData(files, skip=1)

    # Initialize Model
    model = RVmodel(fix=False, npmax=max_npl, data=D)

    # Change Priors
    if len(prior_dict)>0:
        print("?")
    
    # Run the Model
    kima.run(model, steps=steps, num_threads=num_threads, print_thin=200)

    # Load the Result
    res = kima.load_results(model)
    res.save_pickle(filename=save_file)

    return model, res