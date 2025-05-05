# Nonsense code I don't want to be copy/pasting
#     to the top of every notebook

import os
import numpy as np
from astropy.time import Time
import pandas as pd
import seaborn as sns

ceph_dir = '/Users/lilyzhao/Documents/ceph/'
solar_dir = os.path.join(ceph_dir,'Solar')
mask_dir = os.path.join(ceph_dir,'CCF_Masks','ESPRESSO')
essp4_dir = '/Users/lilyzhao/Documents/Employment/ESSP/4SolarTests/ESSP4/'

ceph_dir = '/mnt/home/lzhao/ceph/'
solar_dir = os.path.join(ceph_dir,'SolarData')
mask_dir = os.path.join(ceph_dir,'ESPRESSO_MaskFiles')
essp4_dir = '/mnt/home/lzhao/SolarComparison/ESSP4/'

# =============================================================================
# Useful Variables

# List of Instrument Names (and Grown Up Names)
instruments = ['harpsn','harps','expres','neid']
inst_names = ['HARPS-N','HARPS','EXPRES','NEID']
num_inst = len(instruments)

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
mon_min, mon_max = Time('2021-05-25').mjd,Time('2021-06-23').mjd
unq_day = np.arange(mon_min,mon_max).astype(int)
def mmdd2unqmjd(mmdd):
    mm, dd = mm[:2], dd[2:]
    return int(Time(f'2021-{mm}-{dd}').mjd)

# Offset For Each Instrument Using Binned, Overlap Regions
# Calculated in 250417_instOffset.ipynb
offset_dict = {
    'expres': -1.9809117141404522,
    'neid': -1.108487803337379,
    'harps': -0.26701724421038897,
    'harpsn': 100.22080891122873
}

offset_dict_essp = {
    'expres': -61.310874674375846,
    'neid': -52.04676213938201,
    'harps': 575.8334149163857,
    'harpsn': 565.6070617312148
}

# Edit for DS0 (still debugging why this is necessary)
offset_dict_essp = {
    'expres': -61.53249302289472+9.651874674375843,
    'neid': -51.914834793447966+9.698762139382012,
    'harps': 575.0338367796065+0.6205850836142872,
    'harpsn': 565.0542930652725+0.7579382687852103
}
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
    if file_name[:4] == "Sun_":
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
    return os.path.join(solar_dir,inst_fullName + ('_wBlaze' if 'BLAZE' in file_name else ''),
                        file_name)

def standardSpec_basename2FullPath(file_name):
    name_part_list = os.path.basename(file_name).split('_')
    dset_name = name_part_list[0].split('.')[0]
    full_name = os.path.join(solar_dir,'DataSets',
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