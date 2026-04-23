# Process SDO Observations

import os
from glob import glob
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from scipy import ndimage
from skimage.draw import ellipse as circle
import pandas as pd
from tqdm import tqdm

from utils import sdo_dir

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.coordinates import frames

import SolAster.tools.rvs               as     rvs
import SolAster.tools.calculation_funcs as     sfuncs
import SolAster.tools.lbc_funcs         as     lbfuncs
import SolAster.tools.coord_funcs       as     ctfuncs
import SolAster.tools.utilities         as     utils
from   SolAster.tools.settings          import *
from   SolAster.tools.plotting_funcs    import hmi_plot

# =============================================================================
# Values to be calculated (organized by associated function)

sdo_values = [
    'date_ymd', 'tstamp', 'date_mjd', # Record Keeping
    'f_bright', 'f_spot', 'f',        # getFillingFactors()
    'f_small', 'f_large', 'f_network', 'f_plage', 
    'Bobs', 'quiet_flux', 'ar_flux',  # getMagneticFlux()
    'conv_flux', 'pol_flux', 'pol_conv_flux', 
    'v_quiet', 'v_disc', 'v_conv',    # getVelocities
    'v_phot', 'v_phot_bright', 'v_phot_spot', 
    'v_conv_quiet', 'v_conv_large', 'v_conv_small', 
    'bavg',                              # getBavg
    'n_spot', 'fill_spot',               # labelSpots
    'n_faculae', 'n_plage', 'n_network', # labelFaculae
    'fill_faculae', 'fill_plage', 'fill_network', 
    'date_saved',] #'rv_model'

# =============================================================================
# Managing SDO Files

sdo_file_types = {'dopplergram'  :'hmi.v_720s',  # Velocity map
                  'magnetogram'  :'hmi.m_720s',  # Magnetic field map
                  'intensitygram':'hmi.ic_720s', # Continuum intensity
                  'flatcont'     :'hmi.ic_nolimbdark_720s'} # Limb-Darkening-corrected Intensity
type_list = list(sdo_file_types.keys())

getNName = lambda key : key[0]+'map' if key!='dopplergram' else 'vmap'
nname_list = [getNName(key) for key in sdo_file_types.keys()]

def getOriginalSdoFileName(yyyymmdd,tstamp,file_type,tai_num=3,full_path=False):
    # Format information into SDO file name convention
    file_type = file_type.lower()
    file_name = sdo_file_types[file_type].lower()
    file_name += f'.{yyyymmdd}_{tstamp}_TAI.{tai_num}.'
    if file_type=='dopplergram':
        file_name += file_type.capitalize()
    elif file_type=='flatcont':
        file_name += 'continuum'
    else:
        file_name += file_type
    file_name += '.fits'
    if full_path:
        file_name = os.path.join(sdo_dir,file_type.capitalize(),file_name)
    return file_name

def getSdoFileName(yymmdd,tstamp,file_type,
                   full_path=False,sdo_dir=sdo_dir):
    file_name = f'{yymmdd}.{tstamp}_{file_type}.fits'
    if full_path:
        file_name =  os.path.join(sdo_dir,file_type.capitalize(),file_name)
    return file_name

def downloadDay(ymd,save_dir,min_hr=0,max_hr=24,
                cadence=12*u.minute,overwrite=False,
                e_mail='lilylingzhao@uchicago.edu',
                skip_tstamps=[]):
    day_mjd = Time(f'20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}',format='isot').mjd
    dmin, dmax = Time([day_mjd+min_hr/24, day_mjd+max_hr/24],format='mjd').isot

    for key in type_list:
        matching_images = Fido.search(
            a.Time(dmin,dmax),        # time range in which to search for data
            a.jsoc.Series(sdo_file_types[key]), # list of data products to access
            a.Sample(cadence),        # cadence
            a.jsoc.Notify(e_mail)     # specify user
        )

        # Check For Files That Already Exist
        existing_files = []
        for irec,trec in enumerate(matching_images['jsoc']['T_REC']):
            date, time, _ = trec.split('_')
            # Get saved file names
            ymd = date.replace('.','')[2:]
            tstamp = time.replace(':','')
            jsoc_name = getOriginalSdoFileName('20'+ymd,tstamp,key,tai_num=3,full_path=True)
            essp_name = getSdoFileName(ymd,tstamp,key,full_path=True)
            # If they exist, remove from object
            if tstamp in skip_tstamps or os.path.isfile(jsoc_name) or os.path.isfile(essp_name):
                existing_files.append(irec)
            matching_images['jsoc'].remove_row(existing_files)

        # Download Files
        downloaded_files = Fido.fetch(matching_images,path=os.path.join(save_dir,
                                          key.capitalize(),'{file}'))
        # Iterate over any files that failed to download
        failed_files = downloaded_files.errors
        counter = 0
        while len(failed_files)>0:
            downloaded_files = Fido.fetch(downloaded_files)
            failed_files = downloaded_files.errors
            counter += 1
        print(f'{day}, {key} Redos: {counter}')

        # Rename all files
        dir_name = os.path.join(save_dir,key.capitalize())
        file_list = glob(os.path.join(dir_name,f'hmi*720s.20{ymd}*.fits'))
        for file in file_list:
            yyyymmdd, file_tstamp = os.path.basename(file).split('.')[2].split('_')[:2]
            os.rename(file,getSdoFileName(yyyymmdd[2:],file_tstamp,key,full_path=True))

def deleteDay(yymmdd,save_dir=sdo_dir):
    file_list = glob(save_dir,'*',f'{yymmdd}_*.fits')
    for f in tqdm(file_list,desc=f'Removing {yymmdd} Files'):
        os.remove(f)

class SDO_Obs(object):
    """
    """
    # Intensity and Magnetic Thresholds (for KAM code)
    Icf_lim = 0.89
    Mag_std = 8
    Mag_lim = Mag_std*3
    
    def __init__(self, date, tstamp, save_dir=sdo_dir):
        self.date_ymd = str(date)
        self.tstamp = str(tstamp)
        # Read in Relevant SDO Files
        for key in type_list:
            nname = getNName(key)
            setattr(self,f'{nname}_file',getSdoFileName(date,tstamp,key,full_path=True))
            setattr(self,nname,sunpy.map.Map(getattr(self,f'{nname}_file')))

        # Coordinate Transform for Maps
        # https://tamarervin.github.io/SolAster/calcs/coords/
        self.x, self.y, self.pdim, self.r, self.d, self.mu = ctfuncs.coordinates(self.vmap)
        self.wij, self.nij, self.rij = ctfuncs.vel_coords(self.x, self.y,
                                                          self.pdim, self.r, self.vmap)
        
        # remove bad mu values (i.e. <0.1) mostly about the edge
        self.vmap, self.mmap, self.imap = ctfuncs.fix_mu(self.mu,
                                                         [self.vmap, self.mmap, self.imap])

        ### Values Relevant to KAM code
        imap_meta = self.imap.meta
        self.date_mjd = Time(imap_meta['date-obs']).mjd
        # Data Dimensions
        self.Nrow, self.Ncol = self.imap.data.shape
        # Location of Data
        self.Rsun = int(imap_meta['RSUN_OBS']/imap_meta['CDELT1']*0.99)
        self.Asun = np.pi*self.Rsun**2
        self.xcenter, self.ycenter = int(imap_meta['CRPIX1']), int(imap_meta['CRPIX2'])
        # Mask for Just Data
        sun_mask = np.zeros_like(self.imap.data,dtype=bool)
        circ = circle(self.xcenter,self.ycenter,self.Rsun,self.Rsun)
        sun_mask[circ] = True
        self.sun_mask = sun_mask.copy()
        self.sun_npix = np.sum(sun_mask)

        ### Corrections
        self.correctVelocity()
        self.correctLimbDarkening()
        self.correctMagneticForeshortening()

    # =============================================================================
    # SolAster Functionality

    # =====================================
    # Corrections
    
    def correctVelocity(self):
        ### remove spacecraft velocity and solar rotational velocity
        
        # calculate relative positions
        deltaw, deltan, deltar, dij = sfuncs.rel_positions(self.wij, self.nij, self.rij,
                                                           self.vmap)
    
        # calculate spacecraft velocity
        vsc = sfuncs.spacecraft_vel(deltaw, deltan, deltar, dij, self.vmap)
    
        # optimized solar rotation parameters
        a_parameters = [Parameters.a1, Parameters.a2, Parameters.a3]
        
        # calculation of solar rotation velocity
        self.vrot = sfuncs.solar_rot_vel(self.wij, self.nij, self.rij,
                                         deltaw, deltan, deltar, dij,
                                         self.vmap, a_parameters)
    
        # calculate corrected velocity
        corrected_vel = self.vmap.data - np.real(vsc) - np.real(self.vrot)
    
        # corrected velocity maps
        self.vmap_cor = sfuncs.corrected_map(corrected_vel, self.vmap,
                                             map_type='Corrected-Dopplergram',
                                             frame=frames.HeliographicCarrington)
    
    def correctLimbDarkening(self):
        # remove limb darkening map that's loaded into SolAster
        # (map uses parameterization in Allen 1973)
        
        # limb brightening
        self.Lij = lbfuncs.limb_polynomial(self.imap)
        
        # calculate corrected data
        Iflat = self.imap.data / self.Lij
        
        # corrected intensity maps
        self.imap_cor = sfuncs.corrected_map(Iflat, self.imap,
                                             map_type='Corrected-Intensitygram',
                                             frame=frames.HeliographicCarrington)
    
    def correctMagneticForeshortening(self):
        # Correct using the unsigned magnetic field strength and magnetic noise
        
        # calculate unsigned field strength
        self.Bobs, Br = sfuncs.mag_field(self.mu, self.mmap,
                                         B_noise=Parameters.B_noise,
                                         mu_cutoff=Parameters.mu_cutoff)
    
        # corrected observed magnetic data map
        self.mmap_obs = sfuncs.corrected_map(self.Bobs, self.mmap,
                                             map_type='Corrected-Magnetogram',
                                             frame=frames.HeliographicCarrington)
    
        # radial magnetic data map
        self.mmap_cor = sfuncs.corrected_map(Br, self.mmap,
                                             map_type='Corrected-Magnetogram',
                                             frame=frames.HeliographicCarrington)

    # =====================================
    # Process Images
    
    def addDictValues(self,new_dict):
        for attr,val in new_dict.items():
            setattr(self,attr,val)
    
    def getActiveRegions(self):
        # calculate magnetic threshold
        active, quiet = sfuncs.mag_thresh(self.mu, self.mmap,
                                      Br_cutoff=Parameters.Br_cutoff,
                                      mu_cutoff=Parameters.mu_cutoff)
                
        # calculate intensity threshold
        fac_inds, spot_inds = sfuncs.int_thresh(self.imap_cor,active, quiet)
    
        # create threshold array
        self.threshold_arr = sfuncs.thresh_map(fac_inds, spot_inds)
        # full threshold maps
        self.threshold_map = sfuncs.corrected_map(self.threshold_arr, self.mmap,
                                 map_type='Threshold',
                                 frame=frames.HeliographicCarrington)
        
        actv_dict = {'active':active,'quiet':quiet,
                     'fac_inds':fac_inds,'spot_inds':spot_inds}

        self.addDictValues(actv_dict)
        
        return actv_dict
    
    def getFillingFactors(self):        
        # filling factor
        filling_factors = sfuncs.filling_factor(self.mu, self.mmap,
                                                self.active,self.fac_inds,self.spot_inds,
                                                mu_cutoff=Parameters.mu_cutoff)
        filling_dict = dict(zip(['f_bright','f_spot','f'],filling_factors))
    
        # calculate the area filling factor
        pixA_hem = ctfuncs.pix_area_hem(self.wij, self.nij, self.rij, self.vmap)
        self.area = sfuncs.area_calc(self.active, pixA_hem)
        area_filling_factors = sfuncs.area_filling_factor(self.active,
                                   self.area,self.mu,self.mmap,self.fac_inds,
                                   athresh=Parameters.athresh,
                                   mu_cutoff=Parameters.mu_cutoff)
        filling_dict |= dict(zip(['f_small','f_large','f_network','f_plage'],
                                 area_filling_factors))

        self.addDictValues(filling_dict)
        
        return self.area, filling_dict

    
    def getMagneticFlux(self):
        # unsigned magnetic flux
        # unsigned observed flux
        unsigned_obs_flux = sfuncs.unsigned_flux(self.mmap_obs, self.imap)
        flux_dict = {'Bobs':unsigned_obs_flux}
    
        # get the unsigned flux
        fluxes = sfuncs.area_unsigned_flux(self.mmap_obs, self.imap,
                                           self.area, self.active,
                                           athresh=Parameters.athresh)
        flux_types = ['quiet_flux', 'ar_flux', 'conv_flux',
                      'pol_flux', 'pol_conv_flux']
        flux_dict |= dict(zip(flux_types,fluxes))

        self.addDictValues(flux_dict)
        
        return flux_dict
    
    def getVelocities(self):
        # velocity contribution due to convective motion of quiet-Sun
        v_quiet = sfuncs.v_quiet(self.vmap_cor, self.imap, self.quiet)
        velocity_dict = {'v_quiet':v_quiet}
    
        # velocity contribution due to rotational Doppler imbalance of active regions (faculae/sunspots)
        # calculate photospheric velocity
        vphots = sfuncs.v_phot(self.quiet, self.active,
                               self.Lij, self.vrot, self.imap, self.mu,
                               self.fac_inds, self.spot_inds,
                               mu_cutoff=Parameters.mu_cutoff)
        velocity_dict |= dict(zip(['v_phot','v_phot_bright','v_phot_spot'],
                                 vphots))
    
        # velocity contribution due to suppression of convective blueshift by active regions
        # calculate disc-averaged velocity
        v_disc = sfuncs.v_disc(self.vmap_cor, self.imap)
        velocity_dict['v_disc'] = v_disc
    
        # calculate convective velocity
        velocity_dict['v_conv'] = v_disc - v_quiet
    
        # get area weighted convective velocities
        vconvs = sfuncs.area_vconv(self.vmap_cor, self.imap,
                                   self.active, self.area,
                                   athresh=Parameters.athresh)
        velocity_dict |= dict(zip(['v_conv_quiet','v_conv_large','v_conv_small'],
                                  vconvs))
        
        self.addDictValues(velocity_dict)
        
        return velocity_dict

    def getRvModel(self,inst):
        # calculate model RV
        self.rv_model = rvs.calc_model(inst, self.v_conv, self.v_phot)
        return self.rv_model

    def getSolAsterValues(self):
        self.getActiveRegions();
        self.getFillingFactors();
        self.getMagneticFlux();
        self.getVelocities();
        #self.getRvModel(inst);

    # =============================================================================
    # Identify Active Regions (Khaled's implementation of Haywood+ 2016)

    # Use uncorrected data b/c that's what Khaled has tested
    def getSdoFitsData_kam(self,file_type):
        hdus = fits.open(getattr(self,f'{getNName(file_type)}_file'))
        imag = np.flip(hdus[1].data.astype('float64').copy(),axis=1)
        hdus.close()
        return imag
    
    # =====================================
    # Coordinate Functions

    # Function for mu angle
    def muang(self,iy,ix):
        # Cartesian coordinates
        x = ix - self.Ncol/2
        y = iy - self.Nrow/2
        z = np.sqrt(self.Rsun**2 - x**2 - y**2)
        
        # Mu angle
        muang = np.cos(np.arctan2(np.sqrt(x**2+y**2),z))
    
        return muang
    
    # Function for latitude & longitude
    def latlon(self,iz,iy):
    
        # Cartesian coordinates
        z = iz - self.Nrow/2
        y = iy - self.Ncol/2
        x = np.sqrt(self.Rsun**2 - y**2 - z**2)
        
        # Spherical coordinates
        lat = np.degrees(np.arctan2(np.sqrt(x**2+y**2),z)) - 90
        lon = np.degrees(np.arctan2(y,x))
    
        return lat, lon

    # =====================================
    # Convenience Functions

    def groupLabels(self,imag,lim,less_than=True):
        if less_than:
            labelImage = lambda imag : ndimage.label(np.abs(imag) < lim)
        else:
            labelImage = lambda imag : ndimage.label(np.abs(imag) > lim)
        imag_lab, lab = labelImage(imag.copy())
        imag_map = imag_lab > 0
        
        # Pixel sizes of all groups
        size_pix = np.zeros(lab, dtype=int)
        pix      = np.where(imag_map)
        for i in range(len(pix[0])):
            size_pix[imag_lab[pix[0][i],pix[1][i]]-1] += 1
        
        return imag_lab, lab, size_pix
    
    def bigGroupLabels(self,imag,lim,less_than=True,min_group_size=2):
        imag_lab, lab, size_pix = self.groupLabels(imag,lim,less_than=less_than)
    
        # Image copy with NaNs at positions of groups that are too small
        imag_copy = np.copy(imag)
        lab_small = np.where(size_pix < min_group_size)[0] + 1
        for i in range(len(lab_small)):
            imag_copy[imag_lab == lab_small[i]] = np.nan
        
        return self.groupLabels(imag_copy,lim,less_than=less_than)
    
    def getCoordinates(self,imag_lab,lab_list):
        z = np.ones_like(lab_list)
        y = np.ones_like(lab_list)
        for i,idx in enumerate(lab_list):
            pix  = np.where(imag_lab == idx+1)
            z[i] = np.round(np.mean(pix[0])).astype(int)
            y[i] = np.round(np.mean(pix[1])).astype(int)
    
        return y,z
    
    def getAreaAndRadius(self,y,z,size_pix):
        area_obs = size_pix
        area_abs = area_obs/self.muang(z,y)
        area_rel = area_abs/self.Asun
        
        radi_obs = np.sqrt(area_obs/np.pi)
        radi_abs = np.sqrt(area_abs/np.pi)
        radi_rel = radi_abs/self.Rsun
    
        names = np.concatenate([[f'{i}_{j}' for j in ['obs','abs','rel']]for i in ['area','radi']])
    
        return dict(zip(names,[area_obs,area_abs,area_rel,
                               radi_obs,radi_abs,radi_rel]))
    
    # =====================================
    # Gather Info on Spots/Faculae
    
    def labelSpots(self):
        # Mask and Normalize Continuum Image w/ No Limb Darkening
        imag_icf = self.getSdoFitsData_kam('flatcont') # Read in from FITS file as in original script
        #imag_icf = np.flip(self.imap_cor.data,axis=1) # mostly okay, but some discrepancies
        imag_icf[~self.sun_mask] = np.nan
        imag_icf /= np.nanmedian(imag_icf)
        
        # Group Labels
        imag_Slab, Slab, Ssize_pix = self.bigGroupLabels(imag_icf, lim=self.Icf_lim,
                                                         less_than=True, min_group_size=2)
    
        # Coordinates
        Sy, Sz = self.getCoordinates(imag_Slab,range(Slab))
        
        # Area and radius
        ar_dict = self.getAreaAndRadius(Sy,Sz,Ssize_pix)
    
        # Store all values
        Sdata = {}
        Sdata['size_pix' ] = Ssize_pix
        Sdata['z']         = Sz.copy()
        Sdata['y']         = Sy.copy()
        Sdata['mu_angle' ] = self.muang(Sz,Sy)
        Sdata['latitude' ], Sdata['longitude'] = self.latlon(Sz,Sy)
        Sdata |= ar_dict

        self.spot_imag = imag_Slab
        self.n_spot = Slab
        self.fill_spot = np.sum(Ssize_pix/self.muang(Sz,Sy))/self.sun_npix
        self.spot_dict = Sdata
    
    def labelFaculae(self):
        # Mask and Normalize Magnetic Image
        imag_mag = self.getSdoFitsData_kam('magnetogram') # Read in from FITS file as in original script
        #imag_mag = np.flip(self.mmap_obs.data,axis=1) # Fine to use except for about the edges
        imag_mag[~self.sun_mask] = np.nan
        imag_mag[np.abs(imag_mag)<self.Mag_std] = 0
        
        ### FACULAE
        imag_Flab, Flab, Fsize_pix = self.bigGroupLabels(imag_mag,lim=self.Mag_lim,
                                                         less_than=False,min_group_size=2)

        ### Get Info On (Larger) Plages
        plag_mask = (Fsize_pix/self.Asun) > 20e-6
        # Coordinates
        Py, Pz = self.getCoordinates(imag_Flab,np.arange(Flab)[plag_mask])
        # Area and radius
        ar_dict = self.getAreaAndRadius(Py,Pz,Fsize_pix[plag_mask])
        # Store all values
        Pdata = {}
        Pdata['size_pix'] = Fsize_pix[plag_mask]
        Pdata['z']         = Pz.copy()
        Pdata['y']         = Py.copy()
        Pdata['mu_angle']  = self.muang(Pz,Py)
        Pdata['latitude'], Pdata['longitude'] = self.latlon(Pz,Py)
        Pdata |= ar_dict

        self.facl_imag = imag_Flab
        
        self.facl_size = Fsize_pix[~plag_mask]
        self.n_faculae = len(self.facl_size)
        self.fill_faculae = np.sum(self.facl_size)/self.sun_npix
        
        self.plag_dict  = Pdata
        self.plag_mask  = plag_mask.copy()
        self.n_plage    = np.sum(self.plag_mask)
        self.fill_plage = np.sum(Fsize_pix[plag_mask]/self.muang(Pz,Py))/self.sun_npix

        self.n_network = self.n_faculae-self.n_plage
        self.fill_network = self.fill_faculae-self.fill_plage

    def getBavg(self):
        imag_ico = self.getSdoFitsData_kam('intensitygram')
        #imag_ico = np.flip(self.imap.data,axis=1)
        imag_ico[~self.sun_mask] = np.nan
        # Mask and Normalize Magnetic Image
        imag_mag = self.getSdoFitsData_kam('magnetogram')
        #imag_mag = np.flip(self.mmap_obs.data,axis=1)
        imag_mag[~self.sun_mask] = np.nan
        imag_mag[np.abs(imag_mag)<self.Mag_std] = 0

        self.bavg = np.nansum(np.abs(imag_mag)*imag_ico)/np.nansum(imag_ico)

    def getHaywoodValues(self):
        self.getBavg();
        self.labelSpots();
        self.labelFaculae();

    # =============================================================================
    # Save All Values
    def save(self,save_file,save_dir=sdo_dir,save_regions=True,
             existing_df=None):
        # Save Values
        df = pd.DataFrame({key:getattr(self,key,np.nan) for key in sdo_values},index=[0])
        df['date_saved'] = Time.now().isot
        df['obs_name']   = f'{self.date_ymd}.{self.tstamp}'
        # Add to existing data frame if given
        if existing_df is not None:
            df = pd.concat([existing_df,df],ignore_index=True)
        df.to_csv(save_file,index=False)

        # Save Regions if Requested
        if save_regions:
            # Spot Regions
            np.save(os.path.join(save_dir,'Haywood','SpotMaps',
                        f'{self.date_ymd}.{self.tstamp}_spotImag.npy'),
                    self.spot_imag)
            pd.DataFrame(self.spot_dict).to_csv(os.path.join(save_dir,'Haywood','SpotProps',
                             f'{self.date_ymd}.{self.tstamp}_spotDict.csv'),index=False)

            # Plage Regions
            np.save(os.path.join(save_dir,'Haywood','FaculaeMaps',
                        f'{self.date_ymd}.{self.tstamp}_faclImag.npy'),
                    self.facl_imag)
            pd.DataFrame(self.plag_dict).to_csv(os.path.join(save_dir,'Haywood','PlageProps',
                             f'{self.date_ymd}.{self.tstamp}_plagDict.csv'),index=False)
        
        return df