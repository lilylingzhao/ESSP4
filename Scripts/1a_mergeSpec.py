# Generate Merged 1D Spectra of a Data Set
import argparse
from glob import glob
from tqdm import tqdm

import sys
sys.path.append('/Users/lilyzhao/Documents/Employment/ESSP/4SolarTests/ESSP4/')
from utils import essp_dir
from data import *
from mergeSpec import bind_resample

def main():
    
    parser = argparse.ArgumentParser(description='Generate an ESSP IV Data Set')
    
    parser.add_argument('-d','--data-set',nargs='*',default=[],
                        help='Numbers of data sets to run CCFs for')
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing files")

    # We could consider adding arguments that define the new wavelength grid
    #     but I don't see much utility in that at present
    
    args = parser.parse_args()
    
    # Gather Data Sets
    if not args.data_set:
        dset_list = glob(os.path.join(essp_dir,'DS*.csv'))
    else:
        dset_list = [os.path.join(essp_dir,f'DS{dset_num}.csv') for dset_num in args.data_set]
    
    for dset in dset_list:
        dset_name = os.path.basename(dset)[:-4]
        file_list = glob(os.path.join(essp_dir,'*',dset_name,'Spectra','*.fits'))
        
        for file in tqdm(file_list,desc=f'Merging {dset_name}'):
            merge_file = file.replace('Spectra','Merged').replace('_spec_','_mrge_')
    
            # Read in spectrum
            hdus = fits.open(file)
            head = hdus[0].header
            wave = hdus['wavelength'].data.copy()
            spec = (hdus['flux'].data/hdus['continuum'].data).copy()
            errs = hdus['uncertainty'].data.copy()
            hdus.close()
            
            # Interpolate onto default wavelength grid
            bind_output = bind_resample(wave,spec,errs)
    
            # Save merged spectrum
            hdu_s2d = fits.PrimaryHDU(data=None,header=head)
            hdu_list_s2d = [hdu_s2d]
            for ikey,key in enumerate(['wavelength','flux','uncertainty']):
                hdu_list_s2d.append(fits.ImageHDU(data=bind_output[ikey],name=key))
            fits.HDUList(hdu_list_s2d).writeto(merge_file,overwrite=True)
            tqdm.write(merge_file)

if __name__ == '__main__':
    import sys
    sys.exit(main())