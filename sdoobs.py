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
# Managing SDO Files

sdo_file_types = {'dopplergram':'hmi.v_720s',  # Velocity map
                  'magnetogram':'hmi.m_720s',  # Magnetic field map
                  'continuum'  :'hmi.IC_720s'} # Continuum intensity

def getSdoFileName(yyyymmdd,tstamp,file_type):
    # Format information into SDO file name convention
    file_type = file_type.lower()
    file_name = sdo_file_types[file_type.lower()].lower()
    file_name += f'.{yyyymmdd}_{tstamp}_TAI.3.'
    file_name += file_type.capitalize() if file_type=='dopplergram' else file_type
    file_name += '.fits'
    return file_name

def downloadDay(day,save_dir,
                cadence=12*u.minute,
                e_mail='lilylingzhao@uchicago.edu'):
    day_mjd = Time(day,format='isot').mjd if '-' in day else int(day)
    dmin, dmax = Time([day_mjd+7/24, day_mjd+25/24],format='mjd').isot

    for key in sdo_file_types.keys():
        matching_images = Fido.search(
            a.Time(dmin,dmax),        # time range in which to search for data
            a.jsoc.Series(sdo_file_types[key]), # list of data products to access
            a.Sample(cadence),        # cadence
            a.jsoc.Notify(e_mail)     # specify uset
        )
        
        downloaded_files = Fido.fetch(matching_images,
                                      path=os.path.join(save_dir,key,'{file}'))
        failed_files = downloaded_files.errors
        counter = 0
        while len(failed_files)>0:
            downloaded_files = Fido.fetch(downloaded_files)
            failed_files = downloaded_files.errors
            counter += 1
        print(f'{day}, {key} Redos: {counter}')

def deleteDay(yyyymmdd,save_dir):
    file_list = glob(save_dir,'*',f'hmi*_720s.{yyyymmdd}_*.fits')
    for f in tqdm(file_list,desc=f'Removing {yyyymmdd} Files'):
        os.remove(f)

class SDO_Obs(object):
    """
    """
    def __init__(self, date, tstamp, save_dir):
        # Assemble List of Relvant SDO Files
        file_list = [os.path.join(save_dir,key.capitalize(),
                                  getSdoFileName(date,tstamp,key)) \
                     for key in sdo_file_types.keys()]
        map_seq = sunpy.map.Map(file_list)
        # split into data types
        vmap, mmap, imap = None, None, None
        for j, map_obj in enumerate(map_seq):
            if map_obj.meta['content'] == 'DOPPLERGRAM':
                vmap = map_obj
            elif map_obj.meta['content'] == 'MAGNETOGRAM':
                mmap = map_obj
            elif map_obj.meta['content'] == 'CONTINUUM INTENSITY':
                imap = map_obj
            else:
                assert False, print('File list includes map of unexpected content')
    
        # Make sure we have all the map types
        assert vmap is not None, "No dopplergram file"
        assert mmap is not None, "No magnetogram file"
        assert imap is not None, "No continuum intensity file"

        # Coordinate Transform for Maps
        # https://tamarervin.github.io/SolAster/calcs/coords/
        self.x, self.y, self.pdim, self.r, self.d, self.mu = ctfuncs.coordinates(vmap)
        self.wij, self.nij, self.rij = ctfuncs.vel_coords(self.x, self.y,
                                                          self.pdim, self.r, vmap)
        
        # remove bad mu values (i.e. <0.1) mostly about the edge
        self.vmap, self.mmap, self.imap = ctfuncs.fix_mu(self.mu, [vmap, mmap, imap])

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

# =====================================
# Coordinate Functions

getRsun = lambda data_dict : int(data_dict['header']['RSUN_OBS']/data_dict['header']['CDELT1']*0.99)

# Function for mu angle
def muang(data_dict,iy,ix):

    # Dimension
    Ny, Nx = data_dict['image'].shape
    Rsun = getRsun(data_dict)
    
    # Cartesian coordinates
    x = ix - Nx/2
    y = iy - Ny/2
    z = np.sqrt(Rsun**2 - x**2 - y**2)
    
    # Mu angle
    muang = np.cos(np.arctan2(np.sqrt(x**2+y**2),z))

    return muang

# Function for latitude & longitude
def latlon(data_dict,iz,iy):
    
    # Dimension
    Nz, Ny = data_dict['image'].shape
    Rsun = getRsun(data_dict)

    # Cartesian coordinates
    z = iz - Nz/2
    y = iy - Ny/2
    x = np.sqrt(Rsun**2 - y**2 - z**2)
    
    # Spherical coordinates
    lat = np.degrees(np.arctan2(np.sqrt(x**2+y**2),z)) - 90
    lon = np.degrees(np.arctan2(y,x))

    return lat, lon


# =====================================
# Convenience Functions

def assembleHaywoodImages(date,data_dir,mask=True,norm=True,mag_std=8):
    ico_fits = fits.open(glob(os.path.join(data_dir,'Intensitygram_cont',f'hmi.Ic_720s.{date}*.fits'))[0])
    icf_fits = fits.open(glob(os.path.join(data_dir,'Intensitygram_flat',f'hmi.Ic_noLimbDark_720s.{date}*.fits'))[0])
    mag_fits = fits.open(glob(os.path.join(data_dir,'Magnetogram',f'hmi.M_720s.{date}*.fits'))[0])

    data_products = []
    for data_type,hdus in zip(('ico','icf','mag'),[ico_fits,icf_fits,mag_fits]):
        imag, head = np.flip(hdus[1].data.copy(),axis=1), hdus[1].header.copy()

        if mask:
            # Mask image outside R_sun
            Rsun = int(head['RSUN_OBS']/head['CDELT1']*0.99)
            xcen = int(head['CRPIX1'])
            ycen = int(head['CRPIX2'])
            circ = circle(xcen, ycen, Rsun, Rsun)
            r_mask = np.ones_like(imag, dtype=bool)
            r_mask[circ] = False
            imag[r_mask] = np.nan

        if norm:
            if data_type=='icf':
                imag /= np.nanmedian(imag)
            elif data_type=='mag':
                imag[np.abs(imag) < mag_std] = 0
        
        data_products.append(dict(zip(['image','header'],[imag,head])))
        hdus.close()

    return data_products

def groupLabels(imag,lim,less_than=True):
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

def bigGroupLabels(imag,lim,less_than=True,min_group_size=2):
    imag_lab, lab, size_pix = groupLabels(imag,lim,less_than=less_than)

    # Image copy with NaNs at positions of groups that are too small
    imag_copy = np.copy(imag)
    lab_small = np.where(size_pix < min_group_size)[0] + 1
    for i in range(len(lab_small)):
        imag_copy[imag_lab == lab_small[i]] = np.nan
    
    return groupLabels(imag_copy,lim,less_than=less_than)

def getCoordinates(imag_lab,lab):
    z = np.ones_like(lab)
    y = np.ones_like(lab)
    for i,idx in enumerate(lab):
        pix  = np.where(imag_lab == idx+1)
        z[i] = np.round(np.mean(pix[0])).astype(int)
        y[i] = np.round(np.mean(pix[1])).astype(int)

    return y,z

def getAreaAndRadius(data_dict,y,z,Rsun,size_pix):
    Asun = np.pi*Rsun**2
    area_obs = size_pix
    area_abs = area_obs/muang(data_dict,z,y)
    area_rel = area_abs/Asun
    
    radi_obs = np.sqrt(area_obs/np.pi)
    radi_abs = np.sqrt(area_abs/np.pi)
    radi_rel = radi_abs/Rsun

    names = np.concatenate([[f'{i}_{j}' for j in ['obs','abs','rel']]for i in ['area','radi']])

    return dict(zip(names,[area_obs,area_abs,area_rel,
                           radi_obs,radi_abs,radi_rel]))


# =====================================
# Gather Info on Spots/Faculae

def spots(icf,icf_lim=0.89):
    # Group Labels
    imag_Slab, Slab, Ssize_pix = bigGroupLabels(icf['image'],lim=icf_lim,
                                                less_than=True,min_group_size=2)

    # Coordinates
    Sy, Sz = getCoordinates(imag_Slab,range(Slab))
    
    # Area and radius
    Rsun = getRsun(icf)
    ar_dict = getAreaAndRadius(icf,Sy,Sz,Rsun,Ssize_pix)

    # Store all values
    Sdata = {}
    Sdata['size_pix' ] = Ssize_pix
    Sdata['z']         = Sz.copy()
    Sdata['y']         = Sy.copy()
    Sdata['mu_angle' ] = muang (icf,Sz,Sy)
    Sdata['latitude' ] = latlon(icf,Sz,Sy)[0]
    Sdata['longitude'] = latlon(icf,Sz,Sy)[1]
    Sdata |= ar_dict

    return imag_Slab, pd.DataFrame(Sdata)

def faculae(mag,mag_lim=24):
    ### FACULAE
    imag_Flab, Flab, Fsize_pix = bigGroupLabels(mag['image'],lim=mag_lim,
                                                less_than=False,min_group_size=2)
    
    # Coordinates
    Fy, Fz = getCoordinates(imag_Flab,range(Flab))
    
    # Area and radius
    Rsun = getRsun(mag)
    Asun = np.pi*Rsun**2
    ar_dict = getAreaAndRadius(mag,Fy,Fz,Rsun,Fsize_pix)

    # Store all values
    Fdata = {}
    Fdata['size_pix' ] = Fsize_pix
    Fdata['plage']     = Fsize_pix/Asun > 20e-6 # Mark whether it's a plage or not
    Fdata['z']         = Fz.copy()
    Fdata['y']         = Fy.copy()
    Fdata['mu_angle' ] = muang (mag,Fz,Fy)
    Fdata['latitude' ] = latlon(mag,Fz,Fy)[0]
    Fdata['longitude'] = latlon(mag,Fz,Fy)[1]
    Fdata |= ar_dict

    return imag_Flab, pd.DataFrame(Fdata)