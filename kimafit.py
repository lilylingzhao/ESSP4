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
from planetInjection import getRvTimeSeries

# Define trend degree
kima_trend_dict = {dset:3 if dset in [4,8] else 0 for dset in range(1,10)}

# =============================================================================
# Kima Fitting Files

def binByDay(t,v,e):
    tint = t.astype(int)
    tbin, vbin, ebin = np.zeros((3,len(tint)))
    for i,it in enumerate(tint):
        m = tint==it
        tbin[i] = np.median(t[m])
        vbin[i] = np.average(v[m],weights=1/e[m])
        ebin[i] = np.sqrt(1/np.sum(1/e[m]**2))
    return tbin, vbin, ebin

def genKimaDataFiles(essp_data_file,save_dir,
                     t_key='Time [eMJD]',v_key='RV [m/s]',e_key='RV Err. [m/s]',
                     separate_insts=True,bin_by_day=False):
    df = pd.read_csv(essp_data_file)
    m = df[v_key].notna() & df[e_key].notna()
    if np.sum(m)==0:
        return []
    df = df.copy()[m]

    if separate_insts:
        file_list = []
        for inst in instruments:
            save_file = os.path.join(save_dir,f'RVData_{inst}.rdb')
            m_inst = df['Instrument']==inst
            if np.sum(m_inst)==0:
                continue
            time, rvel, errs = df.loc[m_inst,[t_key,v_key,e_key]].to_numpy().T
            if bin_by_day:
                time, rvel, errs = binByDay(time, rvel, errs)
            np.savetxt(save_file, np.array([time, rvel, errs]).T,
                       header='time rv rv_err', comments='', fmt='%f %f %5.3f')
            file_list.append(save_file)
    else:
        save_file = os.path.join(save_dir,f'RVData.rdb')
        time, rvel, errs = df.loc[:,[t_key,v_key,e_key]].to_numpy().T
        if bin_by_day:
            tall, vall, eall = [],[],[]
            for inst in instruments:
                m = df['Instrument']==inst
                t,v,e = binByDay(time[m], rvel[m], errs[m])
                tall.append(t)
                vall.append(v)
                eall.append(e)
            tall, vall, eall = np.concatenate(tall), np.concatenate(vall), np.concatenate(eall)
            tsort = np.argsort(tall)
            time, rvel, errs = tall[tsort], vall[tsort], eall[tsort]
        np.savetxt(save_file, np.array([time, rvel, errs]).T,
                   header='time rv rv_err', comments='', fmt='%f %f %5.3f')
        file_list = save_file        
    return file_list

def kimaFit(files,save_file=None,
            max_npl=3,prior_dict={},
            trend_deg=3,t_baseline=1,
            steps=100_000,num_threads=4,print_thin=200,
            **kwargs):
    D = RVData(files, skip=1)

    # Initialize Model
    model = RVmodel(fix=False, npmax=max_npl, data=D, **kwargs)
    # Set Kumaraswamy bound for eccentricity
    model.conditional.eprior = kima.distributions.Kumaraswamy(0.8, 3)
    # Change period bound
    model.conditional.Pprior = distributions.LogUniform(2, t_baseline*D.get_timespan())
    if trend_deg>0: # introduce a polynomial fit
        model.trend = True
        model.degree = trend_deg

    # Change Priors
    if len(prior_dict)>0:
        # I don't think this will actually work for most priors
        for key in prior_dict.keys():
            setattr(model.conditional,key,prior_dict[key])
    
    # Run the Model
    kima.run(model, steps=int(steps), num_threads=int(num_threads), print_thin=int(print_thin))

    # Load the Result
    res = kima.load_results(model)
    if save_file is not None:
        res.save_pickle(filename=save_file)

    return model, res

def posteriorSampleDataFrame(file,save_file=None):
    # Get Column Names
    f = open(file,'r')
    col_names = f.readline().lstrip('#').strip().split()
    f.close()
    # Combine into Data Frame
    df = pd.read_csv(file, sep=r'\s+', comment='#', names=col_names)
    if save_file is not None:
        df.to_csv(save_file,index=False)
    return df


# =============================================================================
# Functions for Testing Kima Performance

def genKimaTestData(dset,kima_data_dir,err=None,rand_seed=None):
    # Generate Data
    dset_df = pd.read_csv(os.path.join(data_dir,'Training',f'DS{dset}',f'DS{dset}_timeSeries.csv'))
    t = dset_df['Time [eMJD]']
    # Generate RVs
    param_file = os.path.join(essp_dir,'SuperSecretPlanets',f'dataset_seq_{dset}.csv')
    v = getRvTimeSeries(t,param_file,host_mass=1)
    # Add Errors
    if err is None: # Use given errors
        e = dset_df['RV Err. [m/s]']
    else:
        e = np.zeros_like(v)+err
    np.random.seed(rand_seed)
    v += np.random.randn(len(t))*e
    
    # Write to File
    data_file = os.path.join(kima_data_dir,'testCase.rdb')
    np.savetxt(data_file, list(zip(t,v,e)),
               header='time rv rv_err', comments='', fmt='%f %f %5.3f')
    return data_file


# =============================================================================
# Functions for Collecting Kima Results

kimaOrb_2essp = {'K':'K [m/s]','P':'P [d]','ecc':'e','w':'w [deg]','phi':'phi [deg]'}
smryVals = ['mean','std','median','neg_sigma','pos_sigma']

def getKimaValDict(smry_df,key):
    essp_key = kimaOrb_2essp[key[:-1]] if key[:-1] in kimaOrb_2essp.keys() else key
    key_dict = {}
    for val in smryVals:
        key_dict[essp_key + f' {val}'] = smry_df[key][val]
    return key_dict

def getSysDf(posteriors_df):
    # Prep DF of Summary Values
    smry_df = posteriors_df.describe(percentiles=[0.158,0.5,0.841]).T.rename(columns={'50%':'median'})
    smry_df['neg_sigma'] = smry_df['median']-smry_df['15.8%']
    smry_df['pos_sigma'] = smry_df['84.1%']-smry_df['median']
    smry_df = smry_df.T

    # Organize into a Single DF for System
    plnt_list = []
    for npln in range(3):
        plnt_dict = {'planet':[*'bcd'][npln]}
        for kima_key in smry_df.columns:
            if kima_key[:-1] in kimaOrb_2essp.keys(): # Is a planet parameter
                if int(kima_key[-1])!=npln:
                    continue
            plnt_dict |= getKimaValDict(smry_df,kima_key)
        plnt_list.append(plnt_dict)
    sys_df = pd.DataFrame(plnt_list)
    sys_df['P_sort'] = np.argsort(sys_df['P [d] mean'])

    return sys_df