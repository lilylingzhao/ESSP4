# Generate Data Set with Given Parameters
import os
from glob import glob
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../')
from utils import solar_dir, standardSpec_basename2FullPath, standardSpec_spec2ccf
from indicators import *

def main():
    
    parser = argparse.ArgumentParser(description='Generate indicators for ESSP IV data')
    
    # Specify data set(s) for which to run Indicators
    parser.add_argument('-d','--data-set',nargs='*',default=[],
                        help='Numbers of data sets to run CCFs for')
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing time series files")
    
    args = parser.parse_args()
    
    # Gather Data Sets
    if not args.data_set:
        dset_list = glob(os.path.join(solar_dir,'DataSets','DS*.csv'))
    else:
        dset_list = [os.path.join(solar_dir,'DataSets',f'DS{dset_num}.csv') for dset_num in args.data_set]
    
    for dset in dset_list:
        dset_name = os.path.basename(dset)[:-4]
        train_file = os.path.join(os.path.dirname(dset),'Training',dset_name,f'{dset_name}_timeSeries.csv')
        valid_file = train_file.replace('Training','Validation')
        if os.path.isfile(train_file) and os.path.isfile(valid_file) and not args.overwrite:
            continue
        
        # Read in Data Set Info
        ds_df = pd.read_csv(dset).set_index('Standard File Name')
        for col in cols_final: # Add columns that don't yet exist
            if col not in ds_df.columns:
                ds_df[col] = np.nan
        
        # Loop through list of standard spectrum files
        for ifile, file in enumerate(tqdm(ds_df.index,desc=f'{dset_name} Indicators')):
            
            # Get spectra related indicators
            spec_file = standardSpec_basename2FullPath(file)
            hdus = fits.open(spec_file)
            wvln = hdus['wavelength'].data.copy()
            spec = (hdus['flux'].data/hdus['continuum'].data).copy()
            errs = hdus['uncertainty'].data.copy()
            hdus.close()
            # H-alpha Emission
            try:
                ds_df.at[file,'H-alpha Emission'] = lineEmission(wvln, spec, errs=errs)
            except:
                tqdm.write(f'Failed to get H-alpha for: {file}')
            # S Index
            try:
                ds_df.at[file,'S Index'] = caHK(wvln, spec, errs)
            except:
                tqdm.write(f'Failed to get S Index for: {file}')
            
            # Get CCF related indicators
            ccf_file = standardSpec_spec2ccf(spec_file)
            hdus = fits.open(ccf_file)
            ccf_x = hdus['v_grid'].data.copy()
            ccf_y = hdus['ccf'].data.copy()
            ccf_e = hdus['e_ccf'].data.copy()
            hdus.close()
            # FWHM and Contrast
            try:
                (fwhm, e_fwhm), (contrast, e_contrast) = ccfFwhmContrast(ccf_x,ccf_y,ccf_e)
                if not np.isfinite(e_fwhm):
                    e_fwhm = np.nan
                if not np.isfinite(e_contrast):
                    e_contrast = np.nan
                ds_df.at[file,'CCF FWHM [m/s]'], ds_df.at[file,'CCF FWHM Err. [m/s]'] = fwhm, e_fwhm
                ds_df.at[file,'CCF Contrast [m/s]'], ds_df.at[file,'CCF Contrast Err. [m/s]'] = contrast, e_contrast
            except:
                tqdm.write(f'Failed to fit CCF to Gaussian for: {file}')
            # BIS
            try:
                ds_df.at[file,'BIS [m/s]'] = findBIS(ccf_x,ccf_y,ccf_e)*1000
            except:
                tqdm.write(f'Failed to get CCF Bisector for: {file}')
        
        # Remove Columns not specified in cols final
        train_mask = ds_df['Training'].to_numpy().copy()
        for col in ds_df.columns:
            if col not in cols_final:
                del ds_df[col]
        
        # Save info for T/V separately (while re-ordering columns)
        ds_df[cols_final][train_mask].sort_values(by='Time [eMJD]').to_csv(train_file)
        ds_df[cols_final][~train_mask].sort_values(by='Time [eMJD]').to_csv(valid_file)

cols_final = ['Time [eMJD]','RV [m/s]', 'RV Err. [m/s]',
              'Exp. Time [s]', 'Airmass', 'BERV [km/s]', 'Instrument',
              'CCF FWHM [km/s]','CCF FWHM Err. [km/s]',
              'CCF Contrast','CCF Contrast Err.',
              'BIS [km/s]','H-alpha Emission','S Index']

if __name__ == '__main__':
    import sys
    sys.exit(main())