#!python
import os
import shutil
from glob import glob
import numpy as np
from astropy.time import Time
import pandas as pd
from tqdm import tqdm

# Replace with local pointing to box download
essp_data_dir = '/Volumes/Hasbrouck/ceph/ESSP_Solar/4_DataSets'
box_dir = '/Users/lilyzhao/Documents/ceph/ESSP_Solar/ESSP4/Box/Submissions'

save_dir = '/Users/lilyzhao/Documents/ceph/ESSP_Solar/ESSP4/Submissions'

# Remove Any Existing Submissions Folder
# (There's no reason not to start from scratch, right?)
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
# Re-make Submissions Folder
os.makedirs(save_dir)


# =============================================================================
# Scaffolding Functions

# Make team folder and method sub-folders
def makeTree(team_name,method_list,sub_folders=['Results']):
    if type(method_list)==str:
        method_list = [method_list]
    for method in method_list:
        for sub in sub_folders:
            os.makedirs(os.path.join(save_dir,team_name,method,sub))

def addFileNames(results_file,t_tol=5/60/60/24):
    # 5 second tolerance in case BJD is used
    
    results_df = pd.read_csv(results_file)
    assert 'Standard File Name' not in results_df.columns

    ds = os.path.basename(results_file).split('_')[0]
    ts_df = pd.read_csv(os.path.join(essp_data_dir,'Training',ds,f'{ds}_timeSeries.csv'))
    ds_columns = ['Time [eMJD]','Standard File Name']
    if 'Instrument' not in results_df.columns:
        ds_columns.append('Instrument')
    filed_df = pd.merge_asof(ts_df.loc[:,ds_columns],
                             results_df.sort_values(by='Time [eMJD]'), on='Time [eMJD]',
                             direction='nearest',tolerance=t_tol)
    return filed_df


# =============================================================================
# AustinGeneva

team_name = 'AustinGeneva'
method = 'CNNFitter'
# Make Necessary File Structure
makeTree(team_name,method,sub_folders=['Results','PlanetFit'])

# Move files from individual DS folders to CNN folder
file_list = glob(os.path.join(box_dir,'AustinGeneva','DS*','*_results.csv'))
for og_file in file_list:
    # Replace "Austin_Geneva" with "AustinGeneva"
    # Replace "CNN_Filter" with "CNNFilter"
    file_baseName = os.path.basename(og_file).replace("Austin_Geneva","AustinGeneva").replace("CNN_Fitter","CNNFitter")
    file = os.path.join(save_dir,team_name,method,'Results',file_baseName)
    shutil.copy(og_file,file)

    # Rename "Time (eMJD)" column to "Time [eMJD]"
    df = pd.read_csv(file).rename(columns={'Time (eMJD)':'Time [eMJD]'})
    # Remove "Ind.[U]" column, which is just the instrument
    del df['Ind.[U]']
    df.to_csv(file,index=False)

    # Add File Names to Results Table
    addFileNames(file).to_csv(file,index=False)

# Combine planet signals for each data set
for ds in range(1,10):
    file_list = glob(os.path.join(box_dir,team_name,f'DS{ds}','*_signal*.csv'))
    if len(file_list)==0:
        continue
    planet_df = pd.concat([pd.read_csv(f) for f in file_list])
    planet_df['e'] = 0
    planet_df['w [deg]'] = 0

    save_file = os.path.join(save_dir,team_name,method,'PlanetFit',
                             f'DS{ds}_{team_name}_{method}_planetFit.csv')
    planet_df.to_csv(save_file,index=False)

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,'AustinGeneva','*.*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))


# =============================================================================
# DTUPadovaPSU

og_team_name = 'DTU-Padova-PSU'
team_name = 'DTUPadovaPSU'
# Method Renaming
method_dict = {}
for mc in ['Single','Multiple']:
    for ind_num in [2,4,5]:
        method_dict[f'emcee_{mc.lower()}_{ind_num}_activity_indi'] = f'emcee{mc}{ind_num}actInd'
    method_dict[f'emcee_{mc.lower()}_ccfs'] = f'emcee{mc}{ind_num}ccfInd'
for fiesta_num in [2,3]:
    method_dict[f'emcee_multiple_{fiesta_num}modes'] = f'emceeMultiple{fiesta_num}fiesta'
# Make Necessary File Structure
makeTree(team_name,list(method_dict.values()),sub_folders=['Results','PlanetFit','Hyperparameters'])

# Move files:
#     0) don't bother if file is empty
#     1) unique sub-folder for each method (even sub-classes of methods)
#     2) separate out results, planetFit, and hyperparameter files
file_list = sorted(glob(os.path.join(box_dir,og_team_name,'*','*.csv')))
for og_file in file_list:
    if len(pd.read_csv(og_file))==0:
        continue
    
    base_name = os.path.basename(og_file)
    name_parts = base_name.split('_')
    # Get the New Method Name
    method_name = method_dict['_'.join(name_parts[2:-1])]
    # Get the Type of Result
    file_type = name_parts[-1][:-4]
    # Assemble into new directory and file name
    file = os.path.join(save_dir,team_name,method_name,file_type.capitalize(),
                        f'{name_parts[0]}_{team_name}_{method_name}_{file_type}.csv')
    # Copy File Over
    shutil.copy(og_file,file)

# Add File Names to Results Tables
file_list = glob(os.path.join(save_dir,team_name,'*','Results','*_results.csv'))
for file in file_list:
    addFileNames(file).to_csv(file,index=False)

# Move HTML Files
file_list = sorted(glob(os.path.join(box_dir,og_team_name,'*','*.html')))
for og_file in file_list:
    dir_name, base_name = og_file.split('/')[-2:]
    dir_name = dir_name.split('_')[-1].capitalize()
    file = f'emcee{dir_name}Comparison_{base_name[17:]}'
    # Copy File Over
    shutil.copy(og_file,os.path.join(save_dir,team_name,file))

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,og_team_name,'*.*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))


# =============================================================================
# TeamLSD

og_team_name = 'TeamLSD'
team_name = 'LSD'
method = 'MMLSD'
# Make Necessary File Structure
makeTree(team_name,method,sub_folders=['Results','PlanetFit','PlanetFitPosteriors'])

for ds in range(1,10):
    # Copy Over Planet Fit Files
    shutil.copy(os.path.join(box_dir,og_team_name,'Planet_Parameters',
                             f'DS{ds}_TeamLSD_MMLSD+TWEAKS.csv'),
                os.path.join(save_dir,team_name,method,'PlanetFit',
                             f'DS{ds}_{team_name}_{method}_planetFit.csv'))

    # Copy Over Planet Posterior File Files as CSV
    pos_file = os.path.join(box_dir,og_team_name,'Planet_fit_posteriors',
                             f'DS{ds}_posterior_sample.txt')
    new_pos_file = os.path.join(save_dir,team_name,method,'PlanetFitPosteriors',
                             f'DS{ds}_{team_name}_{method}_planetFitPosteriors.txt')
    pos_col_names = open(pos_file,'r').read().splitlines()[0][3:].split('   ')
    pd.read_table(pos_file, sep=r'\s+', comment='#',
                  names=pos_col_names).to_csv(new_pos_file,index=False)
    
    # Copy Over Results Files
    file = os.path.join(save_dir,team_name,method,'Results',f'DS{ds}_{team_name}_{method}_results.csv')
    shutil.copy(os.path.join(box_dir,og_team_name,'Results',f'ESSP_DS{ds}_Combined.csv'),file)
    
    # Rename "Time [MJD]" column to "Time [eMJD]"
    df = pd.read_csv(file).rename(columns={'Time [MJD]':'Time [eMJD]'})
    # Remove 'instrum' column
    del df['instrum']
    df.to_csv(file,index=False)
    # Add File Names
    addFileNames(file).to_csv(file,index=False)

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,og_team_name,'*.*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))
shutil.copy(os.path.join(box_dir,og_team_name,'Method_details','MM_LSD_description.pdf'),
            os.path.join(save_dir,team_name,f'{team_name}_{method}.pdf'))


# =============================================================================
# WisconsinPennStateChicago

team_name = 'WisconsinPennStateChicago'
method_list = ['baseline', 'gauss', 'gaussPlusPCA']
# Make Necessary File Structure
makeTree(team_name,method_list)

file_list = sorted(glob(os.path.join(box_dir,team_name,'[1-9]_*.csv')))
for og_file in file_list:
    base_name = os.path.basename(og_file)
    method = base_name.split('_')[-2]

    file = os.path.join(save_dir,team_name,method,'Results',f'DS{base_name}')
    shutil.copy(og_file,file)

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,team_name,f'{team_name}*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))