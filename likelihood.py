# Log likelihood
import numpy as np
from cosmology import D_M_lcdm, D_M_tsallis

from data_processing import process_data_and_covariance


# Define file paths
data_file_path = "C:/Users/uditt/Downloads/Pantheon+SH0ES.txt"
cov_file_path = "C:/Users/uditt/Downloads/Pantheon+SH0ES_STAT+SYS.txt"
# Process data and covariance
zcmb, zhel, mu, cov = process_data_and_covariance(data_file_path, cov_file_path)
mu = mu + 19.2
cov_inv = np.linalg.inv(cov)


def log_likelihood_lcdm(params_lcdm):
    Omegam0,H0 = params_lcdm
    Omegade = 1 - Omegam0
    diff = mu - D_M_lcdm(zcmb,Omegam0,Omegade,H0)
    lnL = -0.5 * np.dot(diff, np.dot(cov_inv, diff))
    return lnL

def log_likelihood_tsallis(params_tsallis):
    H0, Omegam0,beta= params_tsallis
    diff = mu - D_M_tsallis(zcmb, H0, Omegam0,beta)
    lnL = -0.5 * np.dot(diff, np.dot(cov_inv, diff))
    return lnL

# Log prior
def log_prior_lcdm(params_lcdm):
    Omegam0,H0 = params_lcdm
    if  0.1 < Omegam0 < 0.5  and 50 < H0 < 90:
        return 0.0
    return -np.inf

def log_prior_tsallis(params_tsallis):
    H0, Omegam0,beta= params_tsallis
    if  0.1 < Omegam0 < 0.5  and 50 < H0 < 90 and 0.5 < beta <1.5:
        return 0.0
    return -np.inf

# Log posterior
def log_posterior_lcdm(params_lcdm):
    lp = log_prior_lcdm(params_lcdm)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(params_lcdm)

def log_posterior_tsallis(params_tsallis):
    lp = log_prior_tsallis(params_tsallis)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_tsallis(params_tsallis)
