# Function to Merge Spectra (i.e. S1D style)
# Note: binDensity does not play nice w/ the EXPRES pipeline
#           so both cannot be at play at the same time
# https://www.astro.unige.ch/~delisle/bindensity/doc/_autosummary/bindensity.resampling.html#bindensity.resampling
import bindensity
import numpy as np

# =============================================================================
# Merge Spectra
wmin, wmax, wdiff = 3770, 8968, 0.01
default_wnew = np.logspace(np.log10(wmin),np.log10(wmax),int((wmax-wmin)/wdiff+1))
def bind_resample(wave, spec, errs, wnew=default_wnew, err_cut=None):
    """
    Interpolate a spectrum on a new wavelength solution with bindensity.
    (Credit: YinNan)
    
    Parameters
    ----------
    wave, spec, errs : array, floats
       wavelength, flux, and associated errors of spectrum to be interpolated
    wnew : array, floats
        Values of new wavelength solution
    err_cut : array, float
        If given, spectral values with errors above this level will be masked out
    
    Returns
    ----------
    wnew, snew, enew: array, floats
        Spectral values and associated errors of the interpolated spectrum
    """
    # Flatten Input Data
    warr = wave.flatten()
    wsort = np.argsort(warr)
    sarr, earr = spec.flatten()[wsort], errs.flatten()[wsort]
    warr = warr[wsort]
    # Error Cut if Given
    if err_cut is not None:
        e_mask = earr<err_cut
        warr, sarr, earr = warr[e_mask], sarr[e_mask], earr[e_mask]

    # Pad wavelengths w/ "last fence post"
    warr_tmp = np.append(warr,warr[-1]+np.diff(warr)[-1])
    wnew_tmp = np.append(wnew,wnew[-1]+np.diff(wnew)[-1])

    snew, cov_new = bindensity.resampling(wnew_tmp, warr_tmp, sarr,
                                          np.array([earr**2]), kind='cubic')
    return wnew, snew, np.sqrt(cov_new[0])