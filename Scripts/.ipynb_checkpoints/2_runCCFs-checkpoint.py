# Generate Data Set with Given Parameters
import os
from glob import glob
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

import sys
sys.path.append('/Users/lilyzhao/Documents/Employment/ESSP/4SolarTests/ESSP4/')
from utils import essp_dir, standardFile_file2inst, iccf_offset, instrument_nickname2Fullname
from ccf import ccf, ccfFit, default_mask_file, vwidth_dict

def main():
    
    parser = argparse.ArgumentParser(description='Generate CCF files for ESSP IV spectra')
    
    # Specify data set(s) for which to run CCFs
    parser.add_argument('-d','--data-set',nargs='*',default=[],
                        help='Numbers of data sets to run CCFs for')
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing files")
    
    # Use iCCF code (over EXPRES pipeline code)
    parser.add_argument('--iccf', action='store_true',
                        help="Use iCCF code (over EXPRES pipeline code)")
    
    # CCF Parameters
    parser.add_argument('--mask_file',type=str,default=default_mask_file,
                        help='File name of CCF mask to use')
    parser.add_argument('--vrange',type=int,default=12,
                        help='+/- range of velocity grid in km/s')
    parser.add_argument('--v0',type=float,default=0,
                        help='Center of velocity grid in km/s')
    parser.add_argument('--vspacing',type=float,default=.4,
                        help='Velocity spacing in km/s')
    parser.add_argument('--cont_norm', action='store_true',
                        help="Continuum normalize spectra")
    
    # CCF Fitting
    parser.add_argument('--fit-range',type=int,default=12,
                        help='+/- range of velocity grid to include in fit in km/s')
    parser.add_argument('--sigma_v',type=float,default=3,
                        help='Initial guess of sigma for the CCF Gaussian Fit in km/s')
    parser.add_argument('--rv_guess',type=float,default=1.5,
                        help='Initial guess of mean for the CCF Gaussian Fit in km/s')
    
    args = parser.parse_args()
    
    # Different parameters for EXPRES CCF and iCCF
    ccf_params = {
        'vrange':float(args.vrange),
        'v0':float(args.v0),
        'vspacing':float(args.vspacing),
    }
    
    # Gather Data Sets
    if not args.data_set:
        dset_list = glob(os.path.join(essp_dir,'DS*.csv'))
    else:
        dset_list = [os.path.join(essp_dir,f'DS{dset_num}.csv') for dset_num in args.data_set]
    
    for dset in dset_list:
        dset_name = os.path.basename(dset)[:-4]
        file_list = glob(os.path.join(essp_dir,'*',dset_name,'Spectra','*.fits'))
        for file in tqdm(file_list,desc=f'CCFs {dset_name}'):
            ccf_file = file.replace('Spectra','CCFs').replace('_spec_','_ccfs_')
            if os.path.isfile(ccf_file) and not args.overwrite:
                continue
            
            esspCcfFromFile(file,use_iccf=args.iccf,
                            mask_file=args.mask_file,cont_norm=args.cont_norm,
                            **ccf_params,
                            fit_range=float(args.fit_range),sigma_v=float(args.sigma_v),
                            rv_guess=float(args.rv_guess))
            
            tqdm.write(ccf_file)

ccf_head_comments = {
    'vrange': '+/- range of velcity in km/s',
    'v0': 'Center of velocity grid in km/s',
    'vspacing': 'Velocity spacing in km/s',
    'npix': 'Number of pixels used around CCF window',
}

def esspCcfFromFile(file,save_file=None,use_iccf=True,
                    mask_file=default_mask_file,cont_norm=False,
                    vrange=12,v0=0,vspacing=0.4,
                    fit_range=12,sigma_v=3,rv_guess=1.5):
    
    inst = standardFile_file2inst(file)
    
    time, v_grid, ccfs, e_ccfs, orders = ccf(file,use_iccf=use_iccf,
                                             mask_file=mask_file,
                                             cont_norm=cont_norm,
                                             vrange=vrange,v0=v0,vspacing=vspacing)
    
    sampling = 2 if inst in ['harps','harpsn'] else 1
    ccf_rv, ccf_rv_e, ccf_dict = ccfFit(v_grid,ccfs,e_ccfs,orders,
                                        vrange=fit_range,
                                        sigma_v=sigma_v,
                                        rv_guess=rv_guess,
                                        sample_factor=sampling)
    if np.isnan(ccf_rv):
        ccf_rv, ccf_rv_e = 'NaN', 'NaN'
    else:
        ccf_rv, ccf_rv_e = np.around(ccf_rv,3), np.around(ccf_rv_e,3)
    
    ccf_head = fits.Header()
    ccf_head['spec'] = (os.path.basename(file), 'Spectral file')
    ccf_head['inst'] = (instrument_nickname2Fullname(inst), 'Instrument')
    ccf_head['time'] = (time, 'Time of observation [eMJD]')
    ccf_head['date-ccf'] = (Time.now().fits, 'Time of CCF calculation')
    ccf_head['pipeline'] = ('iCCF' if use_iccf else 'EXPRES', 'Code used to derive CCFs')
    ccf_head['normed'] = (cont_norm, 'If the spectra were normalized')
    ccf_head['rv'] = (ccf_rv, 'Best-fit CCF RV in m/s')
    ccf_head['e_rv'] = (ccf_rv_e, 'CCF RV Error m/s')
    ccf_head['mask'] = (os.path.basename(mask_file), 'CCF mask file used')
    ccf_head['window'] = ('box', 'CCF mask window used')
    ccf_head['mwidth'] = (vwidth_dict[inst], 'Mask width for iCCF in km/s')
    ccf_params = dict(zip(['vrange','v0','vspacing'],[vrange,v0,vspacing]))
    for key in ccf_params:
        if key=='obo_rv':
            ccf_head[key] = (ccf_params[key], ccf_head_comments[key])
        else:
            ccf_head[key] = (ccf_params[key], ccf_head_comments[key])
    ccf_head['sigma_v'] = (float(sigma_v), 'Initial guess of sigma in km/s')
    ccf_head['rv0'] = (float(rv_guess), 'Initial guess of mean in km/s')
    ccf_head['weighted'] = (True, 'If the orders are weighted')
    ccf_head['offset'] = (iccf_offset[inst], 'Suggested instrumental offset')
    
    ### Save FITS File
    hdu = fits.PrimaryHDU(data=None,header=ccf_head)
    hdu_list = [hdu]
    for key in ccf_dict.keys():
        hdu_list.append(fits.ImageHDU(data=ccf_dict[key],name=key))

    if save_file is None:
        save_file = file.replace('Spectra','CCFs').replace('_spec_','_ccfs_')
    fits.HDUList(hdu_list).writeto(save_file,overwrite=True)

if __name__ == '__main__':
    import sys
    sys.exit(main())