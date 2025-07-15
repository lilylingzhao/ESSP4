import bindensity as bind

def interpolate_with_bindensity(old_wav, new_wav, spctr, err):
    """
    Routine to interpolate a spectrum on a new wavelength solution with bindensity.
    Parameters
    ----------
    :param old_wav: array, 'old' wavelength solution of the spectrum in question.
    :param new_wav: array, new wavelength solution on which we want to interpolate the spectrum.
    :param spctr: array, flux values of the spectrum.
    :param err: array, error on the flux values of the spectrum.
    Returns
    ----------
    :param new_spctr: array, containing the flux values of the spectrum interpolated on the new wavelength solution.
    :param new_err: array, containing the error on the flux values of the spectrum interpolated on the
    new wavelength solution.
    """

    # bindensity requires the edges of the old wavelength solution as input. This means that if the size
    # of spctr is n then the wavelength solution fed to bindensity needs to have a size of n+1.
    # Similarly, bindensity takes the edges of the new wavelength solution as input. So if the new wavelength
    # solution you want has a size k, then the new wavelength solution fed to Antaress for the interpolation
    # must have a size of k+1. This is why I input two new wavelength solutions, old_wav_tmp and
    # new_wav_tmp, into the resampling function instead of old_wav and new_wav.
    old_wav_tmp = np.append(old_wav, old_wav[-1] + np.diff(old_wav)[-1])
    new_wav_tmp = np.append(new_wav, new_wav[-1] + np.diff(new_wav)[-1])

    bindensity_output = bind.resampling(new_wav_tmp, old_wav_tmp, spctr, cov=np.array([err**2]), kind="cubic")

    new_spctr = bindensity_output[0]

    new_err = np.sqrt(bindensity_output[1][0])

    return new_spctr, new_err