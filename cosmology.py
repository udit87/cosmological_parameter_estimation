import numpy as np
from scipy import integrate
import pandas as pd
import emcee
import matplotlib.pyplot as plt
import corner
import scipy.integrate as it
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.integrate import quad, odeint
from data_processing import process_data_and_covariance


# Define file paths
data_file_path = "C:/Users/uditt/Downloads/Pantheon+SH0ES.txt"
cov_file_path = "C:/Users/uditt/Downloads/Pantheon+SH0ES_STAT+SYS.txt"
# Process data and covariance
zcmb, zhel, mu, cov = process_data_and_covariance(data_file_path, cov_file_path)
mu = mu + 19.2
cov_inv = np.linalg.inv(cov)

# Define cosmological parameters

# Constants
c = 3*10**5 # km/sec

minimum_redshifts = np.min(zcmb)
maximum_redshifts = np.max(zcmb)
    
range_redshifts = np.linspace(minimum_redshifts, maximum_redshifts, int(len(zcmb)/10))

## for lambda_cdm

def E_lcdm(z,Omegam0,Omegade):
    return np.sqrt(Omegam0* (1 + z)**3 + Omegade)
    
def D_L_lcdm(z,Omegam0,Omegade,H0):
    minimum_redshifts = 0
    maximum_redshifts = np.max(z)+1
    z_int = np.linspace(minimum_redshifts, maximum_redshifts, 1000) 

    interpolation_E_z = interp1d(z_int,E_lcdm(z_int,Omegam0,Omegade), kind='cubic', bounds_error=False)
    
    def Ly(y,t):
        return 1/interpolation_E_z(t)

    y = it.odeint(Ly,0,z_int)
    tint = y[:,0]
    cf = tint * ((c ) / H0)
    range_distances = cf
    return interp1d(z_int,range_distances)(z)

## Distance Modulus
def D_M_lcdm(z,Omegam0,Omegade,H0):
    return 5 * np.log10((D_L_lcdm(z,Omegam0,Omegade,H0)*(1+zhel)*10**6)/10)


## for Tsallis 
def dOmegaDE(Omegade, x, Omegam0, neta, beta):
    """
    x = ln(a)
    Returns the derivative of OmegaDE wrt x
    """
    A1 = Omegade*(1-Omegade)
    C1 = (1.-beta)/(2. * (2.-beta))
    D1 = 1./(2. * (2.-beta))
    E1 = (3*(1.-beta))/(2*(2.-beta))
    BB1 = (2.*(beta-1.)) + (neta * (1-Omegade)**C1) * (Omegade**D1) * np.exp(E1 * x)
    return A1 * BB1

def Q(H0, Omegam0, beta):
    """
    Returns the value of neta
    """
    A2 = H0*np.sqrt(Omegam0)
    D2 = (1-beta)/(beta-2)
    C2 = 1/(2.*(2-beta))
    # E2 = B/3.
    E2 = 1 ## E2 = B/3 and B=3

    return 2*(2-beta) * (A2**D2) * (E2**C2)

def E_tsallis(z, H0, Omegam0, beta):
    lna = np.log(1/(1+z))
    neta_here = Q(H0, Omegam0, beta)
    OmegaDE = it.odeint(dOmegaDE, 1 - Omegam0 , lna, args=(Omegam0, neta_here, beta))[:,0]
    HoH0 = np.sqrt(Omegam0*(1+z)**3/(1-OmegaDE)) # H/H0
    return HoH0
    
def D_L_tsallis(z,H0, Omegam0, beta):
    minimum_redshifts = 0
    maximum_redshifts = np.max(z)+1
    z_int = np.linspace(minimum_redshifts, maximum_redshifts, 1000) 


    interpolation_E_z = interp1d(z_int,E_tsallis(z_int, H0, Omegam0, beta), kind='cubic', bounds_error=False)
    
    def Ly(y,t):
        return 1/interpolation_E_z(t)

    y = it.odeint(Ly,0,z_int)
    tint = y[:,0]
    cf = tint * ((c ) / H0)
    range_distances = cf
    return interp1d(z_int,range_distances)(z)

## Distance Modulus
def D_M_tsallis(z,H0, Omegam0,beta):
    return 5 * np.log10((D_L_tsallis(z, H0, Omegam0,beta)*(1+zhel)*10**6)/10)
