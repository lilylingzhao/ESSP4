# Generate Merged 1D Spectra of a Data Set
import argparse
from tqdm import tqdm

import sys
sys.path.append('/Users/lilyzhao/Documents/Employment/ESSP/4SolarTests/ESSP4/')
from utils import solar_dir, instruments, mon_min, offset_dict
from data import *
from mergeSpec import bind_resample

def main():
    
    parser = argparse.ArgumentParser(description='Generate an ESSP IV Data Set')
    
    parser.add_argument('-d','--data-set',type=int,
                        help='Number of the data set to use in saved files.')

    # We could consider adding arguments that define the new wavelength grid
    #     but I don't see much utility in that at present
    
    args = parser.parse_args()

    # Define Data Set Name
    data_set_num = int(args.data_set)
    data_set_name = f'DS{data_set_num}'

    for sub_folder in ['Training','Validation']:
        dset_dir = os.path.join(solar_dir,'DataSets',sub_folder,data_set_name)

        merge_dir = os.path.join(dset_dir,'Merged')
        if not os.path.isdir(merge_dir):
            os.makedirs(merge_dir)

        file_list = glob(os.path.join(dset_dir,'Spectra','*.fits'))
        for file in tqdm(file_list,desc=f'Merging {data_set_name} {sub_folder} Data'):
            merge_file = os.path.join(merge_dir,os.path.basename(file))

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