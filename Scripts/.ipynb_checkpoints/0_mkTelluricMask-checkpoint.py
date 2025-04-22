# Generate Telluric Mask with SELENITE Models
import argparse
from tqdm import tqdm

import sys
sys.path.append('../')
from utils import solar_dir, instruments
from planetInjection import *

def main():
    
    parser = argparse.ArgumentParser(description='Generate a SELENITE-based Telluric Mask')
    
    parser.add_argument('-file-name',type=str,default=os.path.join(solar_dir,'telluricMask.csv'),
                        help='Mark SELENITE values under this value as telluric')
    parser.add_argument('-telluric-cut',type=float,default=.99,
                        help='Mark SELENITE values under this value as telluric')
    parser.add_argument('-file-cut',type=int,default=9,
                        help='Only use days with greater than this number of files')
    parser.add_argument('-smoothing-window',type=int,default=9,
                        help='Window for median filter of telluric mask')
    
    # Rewrite telluric mask if it already exists
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing telluric mask if it already exists")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.file_name) and not args.overwrite:
        return
    
    # Read in EXPRES info
    expres_df = pd.read_csv(os.path.join(solar_dir,'expres_drp.csv'))

    # Define Uniform Wavlength Grid
    x_shift = y_shift/wave_diff_log
    tell_wave = np.exp(np.arange(np.log(wave_min-x_shift),np.log(wave_max-x_shift),wave_diff_log))+x_shift

    # Make a Mask for Each Day
    tint = expres_df['Time [MJD]'].astype(int)
    dint_list, dint_num = np.unique(tint,return_counts=True)
    dint_list = dint_list[dint_num>=args.file_cut]
    num_day = len(dint_list)

    tell_mask_all = np.empty((num_day,len(tell_wave)))
    for idint,dint in enumerate(tqdm(dint_list,desc='Gathering Telluric Info')):
        # Get File List for Day
        dint_mask = tint==dint
        file_list = expres_df['File Name'][dint_mask].to_numpy()

        # Gather Data Across Day
        wave, tell, blaz = getSeleniteModel(file_list)

        # Generate Telluric Mask
        _, tell_mask_all[idint] = makeTelluricMask(wave,tell,blaz,tell_wave=tell_wave,
                                                   tell_cut=args.telluric_cut,
                                                   smoothing_window=args.smoothing_window)

    tell_mask = np.sum(tell_mask_all,axis=0)>0
    print('% Tellurics',np.sum(tell_mask)/tell_mask.size)
    
    np.savetxt(args.file_name,np.array([tell_wave,tell_mask]).T,delimiter=',')

if __name__ == '__main__':
    import sys
    sys.exit(main())