import numpy as np
import math

from astropy.constants import c, M_earth, M_jup, M_sun


def calcKepler(Marr_in, eccarr_in, conv=1e-12):
    nm = np.size(Marr_in)
    nec = np.size(eccarr_in)

    eccarr = eccarr_in
    Marr = Marr_in

    k = 0.85 #some parameter for guessing ecc
    ssm = np.sign(np.sin(Marr))
    Earr = Marr+(ssm*k*eccarr)  #first guess at E
    fiarr = (Earr-(eccarr*np.sin(Earr))-Marr)  #E - e*sin(E)-M; should go to 0 when converges
    convd = np.where(abs(fiarr) > conv) #which indices are unconverged

    count = 0
    while np.size(convd) > 1:
        count += 1
        
        M = np.copy(Marr[convd]) #we only run the unconverged elements
        ecc = eccarr #[convd] ??
        E = np.copy(Earr[convd])
        fi = np.copy(fiarr[convd])
        
        fip = 1.-ecc*np.cos(E) #;d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc*np.sin(E)  #;d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1.-fip #;d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)
        
        d1 = -fi/fip #;first order correction to E
        d2 = -fi/(fip+(d1*fipp/2.)) #;second order correction to E
        d3 = -fi/(fip+(d2*fipp/2.)+(d2*d2*fippp/6.)) #;third order correction to E
        E += d3 #apply correction to E
        
        Earr[convd] = E #update values
        
        fiarr = (Earr-eccarr*np.sin(Earr)-Marr)     #;how well did we do?
        convd = np.where(abs(fiarr) > conv)   #;test for convergence; update indices
    
        if count > 100:
            print("WARNING!  Kepler's equation not solved!!!")
            break

    return Earr

def trueAnomaly(t, p, t0=0, e=0): #I believe tp is also degenerate with e=0?
    phase = (t-t0)/p #phase at each obsList time
    M = 2.*math.pi*(phase-np.floor(phase)) #Mean anom array: at each obsList time
    E1 = calcKepler(M, np.array([e]))

    n1 = 1. + e
    n2 = 1. - e

    #True Anomaly
    return 2.*np.arctan(np.sqrt(n1/n2)*np.tan(E1/2.))

# Units of Suns, Earth masses, days
def getK(Mstar, Mpl, p, e=0, i=np.pi/2):
    K = 28.4329*(1-e**2)**(-.5)*((Mpl*M_earth.value)/M_jup.value)*np.sin(i)
    K *=(((Mpl*M_earth.value)+(Mstar*M_sun.value))/M_sun.value)**(-2./3.)*(p/365.24)**(-1./3.)
    return K
def getRV(t, Mstar, Mpl, p, e=0, w=0, t0=0, i=np.pi/2):
    t = np.array(t)
    v = trueAnomaly(t,p,t0,e)
    RV = getK(Mstar, Mpl, p, e=e, i=i)*(np.cos(w+v)+e*np.cos(w))
    return RV
# Just feed it a K rather than calculate it
def getRV_K(t, K, p, e=0, w=0, t0=0, i=np.pi/2):
    t = np.array(t)
    v = trueAnomaly(t,p,t0,e)
    RV = K*np.sin(i)*(np.cos(w+v)+e*np.cos(w))
    return RV

def getMfromK(K,Mstar,p,e=0,i=np.pi/2):
    Mpl = K/(28.4329*(1-e**2)**(-.5))
    Mpl /= np.sin(i)
    Mpl /= ((Mstar*M_sun.value)/M_sun.value)**(-2./3.)
    Mpl /= (p/365.24)**(-1./3.)
    return Mpl*M_jup.value/M_earth.value # Earth masses

def orbit3D(t,p,t0=0,e=0,w=0,Omega=0,i=0,ret_v=False):
    a = (p)**(2/3)
    t=np.array(t)

    v=trueAnomaly(t,p,t0=t0,e=e)
    r=a*(1-e**2)/(1+e*np.cos(v))

    #returns coordinates in arcseconds
    x=r*(np.cos(Omega)*np.cos(w+v)-np.sin(Omega)*np.sin(w+v)*np.cos(i))
    y=r*(np.sin(Omega)*np.cos(w+v)+np.cos(Omega)*np.sin(w+v)*np.cos(i))
    z=r*(np.sin(w+v)*np.sin(i))
    
    if ret_v:
        return np.array(x),np.array(y),np.array(z),v/np.pi*180
    else:
        return np.array(x),np.array(y),np.array(z)