# Inject Doppler Shifts into Spectra

import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.constants import c, M_sun, M_earth, M_jup
from astropy.timeseries import LombScargle
import pandas as pd

def findPlanet(time,rvel,errs):
    ls = LombScargle(time,rvel,errs)
    freq, powr = ls.autopower()

