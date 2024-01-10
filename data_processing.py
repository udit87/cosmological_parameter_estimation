import numpy as np
def process_data_and_covariance(data_file_path, cov_file_path):
    data = np.genfromtxt(data_file_path, skip_header=1)

    wh = (data[:, 2] > 0.01) # & (data[:, 2] < 1.0)
    zhel = data[wh,6 ]
    zcmb = data[wh, 2] # Modification: To use coloumn 2 instead of coloumn 4
    mu = data[wh, 8]

    with open(cov_file_path, 'r') as file:
        cov = np.loadtxt(file, skiprows=1)

    n = int(round(np.sqrt(len(cov))))
    cov = np.reshape(cov, (n, n))

    mask = np.array(wh, bool)
    cov = cov[mask][:, mask]

    return zcmb, zhel, mu, cov