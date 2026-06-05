#!python
import os
import shutil
from glob import glob
import numpy as np
from astropy.time import Time
import pandas as pd
from tqdm import tqdm

# Replace with local pointing to box download
essp_data_dir = '/Users/lilyzhao/Documents/ceph/ESSP/ESSP4/Data'
box_dir = '/Users/lilyzhao/Documents/ceph/ESSP/ESSP4/Box/Submissions'

save_dir = '/Users/lilyzhao/Documents/ceph/ESSP/ESSP4/Submissions'

# Remove Any Existing Submissions Folder
# (There's no reason not to start from scratch, right?)
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
# Re-make Submissions Folder
os.makedirs(save_dir)

# =============================================================================
# Scaffolding Functions

# Make team folder and method sub-folders
def makeTree(team_name,method_list,sub_folders=['results']):
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

def addPlanetIndexing(planet_df):
    # Add Planet Names if Needed
    if 'planet' not in planet_df:    
        planet_df.insert(0,'planet',[*'bcdefghijklmnop'[:len(planet_df)]])

    # Index by Period
    P_sort = planet_df['P [d]']
    planet_df.insert(planet_df.columns.get_loc('P [d]')+1,'P_sort',
                     len(planet_df)-np.argsort(P_sort)-1)
    return planet_df

# =============================================================================
# AustinGeneva

og_team_name = 'Austin_Geneva'
team_name = 'AustinGeneva'
print(Time.now().isot.split('T')[-1],team_name)
method = 'CNNFitter'
# Make Necessary File Structure
makeTree(team_name,method,sub_folders=['results','planetFit'])

# Move files from individual DS folders to CNN folder
file_list = glob(os.path.join(box_dir,'AustinGeneva','DS*','*_results.csv'))
for og_file in file_list:
    # Replace "Austin_Geneva" with "AustinGeneva"
    # Replace "CNN_Filter" with "CNNFilter"
    file_baseName = os.path.basename(og_file).replace(og_team_name,team_name).replace("CNN_Fitter","CNNFitter")
    file = os.path.join(save_dir,team_name,method,'results',file_baseName)
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

    save_file = os.path.join(save_dir,team_name,method,'planetFit',
                             f'DS{ds}_{team_name}_{method}_planetFit.csv')
    addPlanetIndexing(planet_df).to_csv(save_file,index=False)

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,'AustinGeneva','*.*'))
for og_file in file_list:
    if 'members' in og_file:
        new_base_file = f'{team_name}_members.csv'
    else:
        new_base_file = os.path.basename(og_file).replace(og_team_name,team_name).replace("CNN_Filter","CNNFitter")
    shutil.copy(og_file,os.path.join(save_dir,team_name,new_base_file))


# =============================================================================
# DTUPadovaPSU

og_team_name = 'DTU-Padova-PSU'
team_name = 'DTUPadovaPSU'
print(Time.now().isot.split('T')[-1],team_name)
# Method Renaming
method_dict = {}
for mc in ['Single','Multiple']:
    for ind_num in [2,4,5]:
        method_dict[f'emcee_{mc.lower()}_{ind_num}_activity_indi'] = f'emcee{mc}{ind_num}actInd'
    method_dict[f'emcee_{mc.lower()}_ccfs'] = f'emcee{mc}{ind_num}ccfInd'
for fiesta_num in [2,3]:
    method_dict[f'emcee_multiple_{fiesta_num}modes'] = f'emceeMultiple{fiesta_num}fiesta'
# Make Necessary File Structure
makeTree(team_name,list(method_dict.values()),sub_folders=['results','planetFit','hyperparameters'])

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
    file = os.path.join(save_dir,team_name,method_name,file_type,
                        f'{name_parts[0]}_{team_name}_{method_name}_{file_type}.csv')
    
    if file_type=='results':
        # Add File Names to Results Tables
        addFileNames(og_file).to_csv(file,index=False)
    elif file_type=='planetFit':
        # Add Planet Indexing
        addPlanetIndexing(pd.read_csv(og_file)).to_csv(file,index=False)
    elif file_type=='hyperparameters':
        # Add units to period values (to mirror MALTED results)
        df = pd.read_csv(og_file)
        df = df.rename(columns={key:key+' [d]' for key in ['Prot','Pdec']})
        df.to_csv(file,index=False)
    else:
        # Copy File Over, No Changes Needed
        shutil.copy(og_file,file)

# Move HTML Files
file_list = sorted(glob(os.path.join(box_dir,og_team_name,'*','*.html')))
for og_file in file_list:
    dir_name, base_name = og_file.split('/')[-2:]
    dir_name = dir_name.split('_')[-1]
    file = f'emcee{dir_name}Comparison_{base_name[17:]}'
    # Copy File Over
    shutil.copy(og_file,os.path.join(save_dir,team_name,file))

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,og_team_name,'*.*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)).replace(og_team_name,team_name))


# =============================================================================
# GP

team_name = 'MALTED'
print(Time.now().isot.split('T')[-1],team_name)
method = '1dGP'
ind_list = ['CaII','Contrast','FWHM','Ha','BIS']
ind_dict = dict(zip(ind_list,['CAII','CONT','FWHM','HA','BIS'])) # map to original
fit_list = ['Circular','Keplerian']
signals = [*'bcd']
getGPMethod = lambda ind, fit, sig : f'{method}{ind}{fit}{sig}'
method_list = np.array([[[getGPMethod(i,f,s) for i in ind_list] for f in fit_list]for s in signals]).flatten()
types_of_subs = ['results',
                 'planetFit','planetFitPosteriors',
                 'hyperparameters','hyperparametersPosteriors']
# Make Necessary File Structure
makeTree(team_name,method_list,sub_folders=types_of_subs)

for dset in range(1,10):
    for fit in fit_list:
        for ind in ind_dict.keys():
            for sig in signals:
                method_name = getGPMethod(ind,fit,sig)
                for sub in types_of_subs:
                    # Generate the original file names
                    og_file_name = f'DS{dset}_{team_name}_1dGP-{ind_dict[ind]}-{fit.lower()}_{sub}.csv'
                    og_file = os.path.join(box_dir,'GP',f'signal_{sig}',fit.lower(),og_file_name)
                    if not os.path.isfile(og_file):
                        print(f'Missing: {method_name} {sub}')
                        continue
                    # Generate reorganized file name
                    file_name = f'DS{dset}_{team_name}_{method_name}_{sub}.csv'
                    file = os.path.join(save_dir,team_name,method_name,sub,file_name)

                    if sub in ['planetFitPosteriors','hyperparameters','hyperparametersPosteriors']:
                        # Just move, no edits necessary!
                        shutil.copy(og_file,file)
                    elif sub=='planetFit':
                        # Add P index
                        addPlanetIndexing(pd.read_csv(og_file)).to_csv(file,index=False)
                    else:
                        assert sub=='results'
                        # Add file names and sort by file names
                        df = addFileNames(og_file)
                        # Make instruments lower case
                        df['Instrument'] = [i.lower() for i in df['Instrument']]
                        # Remove units from all columns
                        cols2skip = ['Standard File Name','Time [eMJD]','Instrument']
                        col_dict = {c:c if c in cols2skip else c.split(' ')[0] for c in df.columns}
                        df = df.rename(columns=col_dict)
                        df.to_csv(file,index=False)

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,'GP',f'{team_name}*'))
file_list.append(os.path.join(box_dir,'GP','README'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))


# =============================================================================
# GrazIWF

team_name = 'GrazIWF'
print(Time.now().isot.split('T')[-1],team_name)
method = 'breakpoint'
# Make Necessary File Structure
makeTree(team_name,method,sub_folders=['results','planetFit','planetFitPosteriors'])

file_list = sorted(glob(os.path.join(box_dir,team_name,'RESULTS','DS[1-9]_*.csv')))
for og_file in file_list:
    base_name = os.path.basename(og_file)
    file_type = base_name.split('_')[-1][:-4]
    file = os.path.join(save_dir,team_name,method,file_type,base_name.replace('Breakpoint','breakpoint'))

    # Remove the '#' from the beginning of each file
    # (in absolutely the dumbest way imaginable)
    df = pd.read_csv(og_file)
    col = df.columns
    df = df.rename(columns={col[0]:col[0][1:]})

    if file_type=='results':
        # Add Instrument column
        df['Instrument'] = [f.split('_')[-1].split('.')[0] for f in df['Standard File Name']]
    elif file_type=='planetFit':
        # Add Planet Indexing
        df = addPlanetIndexing(df)
    df.to_csv(file,index=False)

# Copy Auxiliary Files
file_list = [*glob(os.path.join(box_dir,team_name,'RESULTS',f'{team_name}*')),
             *glob(os.path.join(box_dir,team_name,f'{team_name}*'))]
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))


# =============================================================================
# LSD

og_team_name = 'TeamLSD'
team_name = 'LSD'
print(Time.now().isot.split('T')[-1],team_name)
method = 'MMLSD'
# Make Necessary File Structure
makeTree(team_name,method,sub_folders=['results','planetFit','planetFitPosteriors'])

for ds in range(1,10):
    # Copy Over Planet Fit Files
    fit_file = os.path.join(box_dir,og_team_name,'Planet_Parameters',
                            f'DS{ds}_TeamLSD_MMLSD+TWEAKS.csv')
    if type(pd.read_csv(fit_file)['K [m/s]'].to_list()[0])!=float:
        # When there exists a fit the type is string

        # Rewrite error to separate column
        df = pd.read_csv(fit_file)
        for col in df.columns[:-1]:
            for i in df.index:
                val = df[col][i]
                if type(val)!=str:
                    continue
                if '±' in val:
                    val, err = val.split('±')
                    df.at[i,col] = float(val.strip())
                    if 'e'+col not in df.columns:
                        df.insert(df.columns.get_loc(col)+1,'e'+col,np.nan)
                    df.at[i,'e'+col] = float(err.strip())
                elif 'to' in val:
                    # Get rid of the "N to N" entries
                    # Add original range to notes
                    note = df.at[i,'Notes']
                    df.at[i,'Notes'] = note + ('' if len(note)==0 else '; ') + col + ' is ' + val
                    # Add float values to columns
                    val_range = [float(i) for i in val.split(' to ')]
                    df.at[i,col] = np.sum(val_range)/2
                    df.at[i,'e'+col] = -np.diff(val_range)
        planet_fit_file = os.path.join(save_dir,team_name,method,'planetFit',
                              f'DS{ds}_{team_name}_{method}_planetFit.csv')
        # Add Planet Indexing
        addPlanetIndexing(df).to_csv(planet_fit_file,index=False)

    # Copy Over Planet Posterior File Files as CSV
    pos_file = os.path.join(box_dir,og_team_name,'Planet_fit_posteriors',
                             f'DS{ds}_posterior_sample.txt')
    new_pos_file = os.path.join(save_dir,team_name,method,'planetFitPosteriors',
                             f'DS{ds}_{team_name}_{method}_planetFitPosteriors.csv')
    pos_col_names = open(pos_file,'r').read().splitlines()[0][3:].split('   ')
    pd.read_table(pos_file, sep=r'\s+', comment='#',
                  names=pos_col_names).to_csv(new_pos_file,index=False)
    
    # Copy Over Results Files
    file = os.path.join(save_dir,team_name,method,'Results',f'DS{ds}_{team_name}_{method}_results.csv')
    shutil.copy(os.path.join(box_dir,og_team_name,'results',f'ESSP_DS{ds}_Combined.csv'),file)
    
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
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)).replace(og_team_name,team_name))
shutil.copy(os.path.join(box_dir,og_team_name,'Method_details','MM_LSD_description.pdf'),
            os.path.join(save_dir,team_name,f'{team_name}_{method}.pdf'))


# =============================================================================
# Oxford

team_name = 'Oxford'
print(Time.now().isot.split('T')[-1],team_name)
method = 'DCPCA'
makeTree(team_name,method,sub_folders=['results','planetFit','planetFitPosteriors'])

file_list = sorted(glob(os.path.join(box_dir,team_name,'*','*.csv')))
for og_file in file_list:
    if len(pd.read_csv(og_file))==0:
        continue
    
    base_name = os.path.basename(og_file).replace('planetfit','planetFit').replace('oxford','Oxford')
    file_type = base_name.split('_')[-1][:-4]
    file = os.path.join(save_dir,team_name,method,file_type,base_name)
    
    if file_type=='results':
        # Add File Names to Results Tables
        addFileNames(og_file).to_csv(file,index=False)
    elif file_type=='planetFit':
        # Add Planet Indexing
        addPlanetIndexing(pd.read_csv(og_file)).to_csv(file,index=False)
    else:
        # Copy File Over, No Changes Needed
        shutil.copy(og_file,file)

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,team_name,f'{team_name}*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))


# =============================================================================
# PSU

team_name = 'PSU'
print(Time.now().isot.split('T')[-1],team_name)
method = 'Scalpels'
makeTree(team_name,method,sub_folders=['results','planetFit'])

for ds in range(1,10):
    for sub in ['results','planetFit']:
        file_name = f'DS{ds}_{team_name}_{method}_{sub}.csv'
        og_file = os.path.join(box_dir,team_name,method,file_name)
        if not os.path.isfile(og_file):
            print(f'Missing: {og_file_name}')
            continue
        file = os.path.join(save_dir,team_name,method,sub,file_name)

        if sub=='results':
            # Add file names and sort by file names
            df = addFileNames(og_file)
            df.to_csv(file,index=False)
        else:
            assert sub=='planetFit'
            df_list = []
            pln_df_all = pd.read_csv(og_file)
            for inst in ['harpsn','harps','expres','neid']:
                df_list.append(addPlanetIndexing(pln_df_all[pln_df_all['Instrument']==inst]))
            pd.concat(df_list).to_csv(file,index=False)


# =============================================================================
# Sidera

team_name = 'Sidera'
print(Time.now().isot.split('T')[-1],team_name)
method_list = ['FDACmean','FDACmeanXDM']
makeTree(team_name,method_list)

for ds in range(1,10):
    for method in method_list:
        og_file = os.path.join(box_dir,team_name,f'DS{ds}',f'DS{ds}_{team_name}_{method}_results.csv')
        og_file = og_file.replace('XDM','+XDM')
        file = os.path.join(save_dir,team_name,method,'results',f'DS{ds}_{team_name}_{method}_results.csv')
        shutil.copy(og_file,file)

        # Rename "Time [MJD]" to "Time [eMJD]"
        df = pd.read_csv(file,index_col=0).rename(columns={'Time [MJD]':'Time [eMJD]'})
        
        df.to_csv(file,index=False)
    
        # Add File Names to Results Table
        addFileNames(file).to_csv(file,index=False)

# Save Trend Files
file_list = glob(os.path.join(box_dir,team_name,'DS[1-9]','DS[1-9]_trends.csv'))
for trend_file in file_list:
    shutil.copy(trend_file,os.path.join(save_dir,team_name,os.path.basename(trend_file)))

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,team_name,f'{team_name}*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))

# =============================================================================
# WisconsinPennStateChicago

team_name = 'WisconsinPennStateChicago'
print(Time.now().isot.split('T')[-1],team_name)
method_list = ['baseline', 'gauss', 'gaussPlusPCA']
# Make Necessary File Structure
makeTree(team_name,method_list)

file_list = sorted(glob(os.path.join(box_dir,team_name,'[1-9]_*.csv')))
for og_file in file_list:
    base_name = os.path.basename(og_file)
    method = base_name.split('_')[-2]

    file = os.path.join(save_dir,team_name,method,'results',f'DS{base_name}')
    shutil.copy(og_file,file)

# Copy Auxiliary Files
file_list = glob(os.path.join(box_dir,team_name,f'{team_name}*'))
for og_file in file_list:
    shutil.copy(og_file,os.path.join(save_dir,team_name,os.path.basename(og_file)))