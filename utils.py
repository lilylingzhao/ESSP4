# Nonsense code I don't want to be copy/pasting
#     to the top of every notebook

import os
import numpy as np
from astropy.time import Time
import pandas as pd
import seaborn as sns

ceph_dir_local = '/Users/lilyzhao/Documents/ceph/'
ceph_dir = '/Volumes/Hasbrouck/ceph/'
solar_dir = os.path.join(ceph_dir,'ESSP_Solar',)
mask_dir = os.path.join(ceph_dir,'CCF_Masks','ESPRESSO')
essp_dir = os.path.join(solar_dir,'4_DataSets')
essp4_dir = '/Users/lilyzhao/Documents/Employment/ESSP/4SolarTests/ESSP4/'

# Default CCF Mask
default_mask_file = os.path.join(mask_dir,'NEID_G2_telluricAdjusted.fits')
#default_mask_file = os.path.join(mask_dir,'ESPRESSO_G2.fits')

# =============================================================================
# Useful Variables

# List of Instrument Names (and Grown Up Names)
instruments = ['harpsn','harps','expres','neid']
inst_names = ['HARPS-N','HARPS','EXPRES','NEID']
num_inst = len(instruments)

ts_dict = {inst:pd.read_csv(os.path.join(solar_dir,f'{inst}_drp.csv')).loc[:,
           ['Time [MJD]','RV [m/s]','RV Err. [m/s]']].to_numpy().T for inst in instruments}

# Instrument Colors a la ESSP #
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(int(r*256), int(g*256), int(b*256))
# JuliaGraphics Johnson Color Scheme
rgbs = [(0.627,0.055,0.0),
        (0.0,0.525,0.659),
        (0.965,0.761,0.0),
        (0.816,0.306,0.0),
       ]
inst_cols = sns.color_palette(rgbs)

# Time Range of Shared Data
mon_min, mon_max = Time('2021-03-23').mjd,Time('2021-06-23').mjd
unq_day = np.arange(mon_min,mon_max).astype(int)
def mmdd2unqmjd(mmdd):
    mm, dd = mm[:2], dd[2:]
    return int(Time(f'2021-{mm}-{dd}').mjd)

# Offset For Each Instrument Using Binned, Overlap Regions
# Calculated in genAlignment.ipnyb
drp_offset = {
    'neid': -648.2956247730958,
    'expres': -651.2321534973294,
    'harps': -0.16456447894142912,
    'harpsn': -1.4949798091404143
}

# Calculated for iCCF DRP RVs in 250729_checkIccf.ipynb
iccf_offset = {
    'neid': -79.735444958673,
    'expres': -88.60501568816387,
    #'harps': 557.5480840931398,
    'harps' : 551.6919999732906, # special edit from 250730_harpsDS
    'harpsn': 546.3609640230954
}

pd.DataFrame.from_dict(iccf_offset,orient='index',columns=['offset']).to_csv(
    os.path.join(solar_dir,'instrument_offsets_iccf.csv'),header=False)

# =============================================================================
# Instrument/File Names

def instrument_nickname2Fullname(inst):
    matched_name = np.array(inst_names)[np.array(instruments)==inst]
    assert len(matched_name)==1
    return matched_name[0]

def instrument_fullname2Nickname(inst):
    matched_name = np.array(instruments)[np.array(inst_names)==inst]
    assert len(matched_name)==1
    return matched_name[0]

def fileName2Inst(file_name):
    file_name = os.path.basename(file_name)
    if file_name[:2] == 'DS':
        return 'essp'
    elif file_name[:4] == "Sun_":
        return "expres"
    elif file_name[:7] == "neidL2_":
        return "neid"
    else:  # HARPS or HARPS-N
        inst = file_name.replace('r.','').split(".")[0].lower()
        return inst if inst == "harps" else "harpsn"

def getHarpsBlazeFile(file_name):
    dir_list = file_name.split('/')
    blaze_file = os.path.join('/',*dir_list[:-2],dir_list[-2]+'_wBlaze',
                              dir_list[-1].replace('S2D_A','S2D_BLAZE_A'))
    return blaze_file

def getHarpsNoBlazeFile(file_name):
    dir_list = file_name.split('/')
    blaze_file = os.path.join('/',*dir_list[:-2],dir_list[-2][:-7],
                              dir_list[-1].replace('S2D_BLAZE_A','S2D_A'))
    return blaze_file

def spec_basename2FullPath(file_name):
    inst = fileName2Inst(file_name)
    inst_fullName = instrument_nickname2Fullname(inst)
    return os.path.join(solar_dir,'Spectra',inst_fullName + ('_wBlaze' if 'BLAZE' in file_name else ''),
                        file_name)

def spec_spec2ccf(spec_file):
    inst = fileName2Inst(spec_file)
    
    if os.path.basename(spec_file)==spec_file:
        spec_file = spec_basename2FullPath(spec_file)
    
    if inst=='neid':
        ccf_file = spec_file
    elif inst=='expres':
        ccf_file = os.path.join(solar_dir,'CCFs','EXPRES',os.path.basename(spec_file))
    else:
        base_file = os.path.basename(spec_file).replace('_BLAZE','')
        if inst=='harpsn':
            base_file = base_file.replace('_S2D_A','_CCF_A')
        else:
            base_file = base_file[2:].replace('_S2D_A','_ccf_G2_A')
        
        ccf_file = os.path.join(solar_dir,'CCFs',instrument_nickname2Fullname(inst),base_file)
    return ccf_file

def standardSpec_basename2FullPath(file_name):
    name_part_list = os.path.basename(file_name).split('_')
    dset_name = name_part_list[0].split('.')[0]
    full_name = os.path.join(essp_dir,
                             'Validation' if 'v' in name_part_list else 'Training',
                              dset_name,'Spectra',os.path.basename(file_name))
    
    return full_name

def standardSpec_spec2ccf(spec_file):
    return spec_file.replace('Spectra','CCFs').replace('_spec_','_ccfs_')

def standardFile_file2inst(file):
    return file.split('_')[-1].split('.')[0]

# =============================================================================
# Standardizing Relative/Echelle Order

max_nord = 106 # I don't think we need to pad redder orders?

def padOrders(og_arr,inst):
    num_ord,num_pix = og_arr.shape
    if inst=='harps':
        # HARPS is missing order 45 for some reason
        pad_arr = np.insert(og_arr,45,np.full(num_pix,np.nan),axis=0)
        # Pad bluer orders
        pad_arr = np.insert(pad_arr,0,np.full(num_pix,np.nan),axis=0)
    elif inst in ['harpsn','harps-n']:
        pad_arr = np.insert(og_arr,0,np.full((4,num_pix),np.nan),axis=0)
    elif inst=='expres':
        pad_arr = np.insert(og_arr,0,np.full(num_pix,np.nan),axis=0)
    elif inst=='neid':
        pad_arr = og_arr.copy()
    else:
        assert False, print(f'Instrument name "{inst}" not recognized')
    return pad_arr

def unpadOrders(og_arr):
    isord = np.sum(np.isfinite(og_arr),axis=1)>0
    return og_arr[isord]