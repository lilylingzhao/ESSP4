# Generate Telluric Mask with SELENITE Models
import argparse
from tqdm import tqdm

import sys
sys.path.append('/Users/lilyzhao/Documents/Employment/ESSP/4SolarTests/ESSP4/')
from utils import solar_dir, instruments
from planetInjection import *

def main():
    
    parser = argparse.ArgumentParser(description='Generate a Telluric Mask')
    
    parser.add_argument('-file-name',type=str,default=os.path.join(solar_dir,'telluricMask.fits'),
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

    selenite_mask_all = np.empty((num_day,len(tell_wave)))
    selenite_wmin, selenite_wmax = -np.inf, np.inf
    for idint,dint in enumerate(tqdm(dint_list,desc='Gathering Telluric Info')):
        # Get File List for Day
        dint_mask = tint==dint
        file_list = expres_df['File Name'][dint_mask].to_numpy()

        # Gather Data Across Day
        wave, tell, blaz = getSeleniteModel(file_list)
        wmin, wmax = np.nanmin(wave[np.isfinite(tell)]), np.nanmax(wave[np.isfinite(tell)])
        if wmax<selenite_wmax: # Find the lowest max wavelength
            selenite_wmax = wmax
        if wmin>selenite_wmin: # Find the greatest low wavelength
            selenite_wmin = wmin

        # Generate Telluric Mask
        _, selenite_mask_all[idint] = makeSeleniteTelluricMask(wave,tell,blaz,
                                          tell_wave=tell_wave,tell_cut=args.telluric_cut,
                                          smoothing_window=args.smoothing_window)

    selenite_mask = np.sum(selenite_mask_all,axis=0)>0
    
    # Get TAPAS-based Model
    hdus = fits.open(default_tapas_file)
    tapas_wave = hdus[1].data['wavelength'].copy()[::-1] # Saved in reverse order for some reason?
    tapas_tell = hdus[1].data['transmittance'].copy()[::-1] 
    tapas_wave *= 10
    hdus.close()
    
    _, tapas_mask = getTapasModel(tapas_wave,tapas_tell,tell_cut=0.7, # a looser cut works better here
                                  tell_wave=tell_wave,smoothing_window=args.smoothing_window)
    # Use Selenite between selenite_wmin/wmax wavelengths
    tell_mask = tapas_mask.copy()
    selenite_range = (tell_wave>selenite_wmin) & (tell_wave<selenite_wmax)
    tell_mask[selenite_range] = selenite_mask[selenite_range].copy()
    
    print('% Tellurics',np.sum(tell_mask)/tell_mask.size)
    
    # Save CCF Info
    tell_head = fits.Header()
    tell_head['maskdate'] = (Time.now().fits, 'Time mask was generated')
    tell_head['sln-tcut'] = (args.telluric_cut, 'SELENITE telluric cut')
    tell_head['sln-fcut'] = (args.file_cut, 'SELENITE cut on number of files')
    tell_head['tps-tcut'] = (0.7, 'TAPAS telluric cut')
    tell_head['tps-dcut'] = (1000, 'TAPAS telluric derivative cut')
    tell_head['smoothin'] = (args.smoothing_window, 'Smoothing window')
    tell_head['sln-wmin'] = (selenite_wmin, 'Min. of SELENITE mask')
    tell_head['sln-wmax'] = (selenite_wmin, 'Max. of SELENITE mask')
    tell_head['pcntmask'] = (np.sum(tell_mask)/tell_mask.size, 'Percent masked as tellurics')

    ### Save FITS File
    hdu_list = [fits.PrimaryHDU(data=None,header=tell_head),
                fits.ImageHDU(data=tell_wave.copy(),name='wavelength'),
                fits.ImageHDU(data=tell_mask.astype(int).copy(),name='telluric_mask'),
                fits.ImageHDU(data=selenite_mask.astype(int).copy(),name='selenite_mask'),
                fits.ImageHDU(data=tapas_mask.astype(int).copy(),name='tapas_mask')]
    fits.HDUList(hdu_list).writeto(args.file_name,overwrite=True)

if __name__ == '__main__':
    import sys
    sys.exit(main())