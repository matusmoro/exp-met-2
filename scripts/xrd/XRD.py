import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy import optimize
from scipy import constants as c
from matplotlib import pyplot as plt

def gauss(x, mu, A, B):
    """ Simply returns the gaussian for a given x and parameters. Used to find fitting constants.

    Parameters
    ----------

    x : float
        Independent variable
    mu : float
        X-axis offset
    A : float
        Scaling/normalization parameter
    B : float
        Fitting constant related to variance
    C : float
        Y-axis offset

    Returns
    -------
    float
        The function value for the given parameters

    """

    return A*np.exp(-(x-mu)**2/B) + 150

def load_data(path):
    """ Loads diffractogram data into a pandas dataframe.

    Parameters
    ----------
    path : string
        Path to the data files.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with the 2theta and intensity columns

    Raises
    ------
    ValueError
        If the input is not a string
    """

    if not isinstance(path, str):
        raise ValueError("The path must be a string.")

    df = pd.read_csv(path, engine='python', header=None, skiprows=375, skipfooter=3, sep=' ', encoding='unicode_escape')

    return df

def peak_fitter(df, scan_type):
    """Finds the 5 highest peak inside the diffractogram (dataframe). Retrieves their 2theta position and FWHM.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to search the peaks in.
    scan_type : str
        Type of scan, either fast or slow.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with the positions and FWHMs of the 5 highest peaks in the diffractogram
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pd.DataFrame.")

    threshold = 560 if scan_type == 'fast' else 2000

    start_index = 1000 if scan_type == 'fast' else 0

    relative_indices, peak_data = scipy.signal.find_peaks(df.iloc[start_index:,1], height=threshold, distance=100)
    two_theta_indices = relative_indices + start_index

    fit_params = []
    x_data = []
    half_width = 3

    for value in two_theta_indices:
        x = df.iloc[value-half_width:value+half_width, 0].values
        y = df.iloc[value-half_width:value+half_width, 1].values

        params, _ = scipy.optimize.curve_fit(gauss, x, y, p0=[df.iloc[value,0], 1, 1], maxfev=10000000)

        fit_params.append(params)
        x_data.append(x)

    return fit_params, x_data, two_theta_indices

if __name__ == "__main__":

    fast_scan = load_data(r'C:\matt\school\mgr\2025 jar\F8544 Experimentalni metody 2\XRD\xrd\fast_sul.ras')
    slow_scan = load_data(r'C:\matt\school\mgr\2025 jar\F8544 Experimentalni metody 2\XRD\xrd\slow_sul.ras')

    fit_params_fast, x_data_fast, two_theta_indices_fast = peak_fitter(fast_scan, 'fast')
    fit_params_slow, x_data_slow, two_theta_indices_slow = peak_fitter(slow_scan, 'slow')        
    FWHM_fast = [np.sqrt(4*np.log(2)*fit_params_fast[i][2]) for i in range(5)]
    FWHM_slow = [np.sqrt(4*np.log(2)*fit_params_slow[i][2]) for i in range(5)]

    fast_df = pd.DataFrame(columns=['2theta', 'FWHM', 'h', 'k', 'l', 'lambda'])
    fast_df['2theta'] = [p[0] for p in fit_params_fast]
    fast_df['FWHM'] = FWHM_fast
    fast_df['h'] = [2, 2, 2, 4, 4]
    fast_df['k'] = [0, 2, 2, 0, 2]
    fast_df['l'] = [0, 0, 2, 0, 0]
    fast_df['lambda'] = [1.54059, 1.54059, 1.54059, 1.54059, 1.54059]

    slow_df = pd.DataFrame(columns=['2theta', 'FWHM', 'h', 'k', 'l', 'lambda'])
    slow_df['2theta'] = [p[0] for p in fit_params_slow]
    slow_df['FWHM'] = FWHM_slow
    slow_df['h'] = [1, 2, 2, 4, 4]
    slow_df['k'] = [1, 0, 2, 2, 2]
    slow_df['l'] = [1, 0, 0, 0, 2]
    slow_df['lambda'] = [1.54059, 1.54059, 1.54059, 1.54059, 1.54059]

    # Scherrer constant
    K = 0.94

    # --- Process the slow_df ---

    # Convert angles from degrees to radians for numpy calculations
    theta_rad_slow = np.deg2rad(slow_df['2theta'] / 2)
    beta_rad_slow = np.deg2rad(slow_df['FWHM'])

    # 1. Calculate lattice parameter 'a' for each peak (result in Angstroms)
    slow_df['a_param_A'] = (slow_df['lambda'] / (2 * np.sin(theta_rad_slow))) * np.sqrt(slow_df['h']**2 + slow_df['k']**2 + slow_df['l']**2)

    # 2. Calculate crystallite size 'tau' for each peak (result in Angstroms)
    slow_df['tau_A'] = (K * slow_df['lambda']) / (beta_rad_slow * np.cos(theta_rad_slow))

    # Convert tau to nanometers for easier interpretation
    slow_df['tau_nm'] = slow_df['tau_A'] / 10

    # --- Process the fast_df ---

    # Convert angles from degrees to radians
    theta_rad_fast = np.deg2rad(fast_df['2theta'] / 2)
    beta_rad_fast = np.deg2rad(fast_df['FWHM'])

    # 1. Calculate lattice parameter 'a' for each peak (result in Angstroms)
    fast_df['a_param_A'] = (fast_df['lambda'] / (2 * np.sin(theta_rad_fast))) * np.sqrt(fast_df['h']**2 + fast_df['k']**2 + fast_df['l']**2)

    # 2. Calculate crystallite size 'tau' for each peak (result in Angstroms)
    fast_df['tau_A'] = (K * fast_df['lambda']) / (beta_rad_fast * np.cos(theta_rad_fast))

    # Convert tau to nanometers
    fast_df['tau_nm'] = fast_df['tau_A'] / 10


    # --- Display Final Results ---

    print("\n--- Slow Scan Results ---")
    print(slow_df)
    print(f"\nAverage Lattice Parameter (a): {slow_df['a_param_A'].mean():.4f} ± {slow_df['a_param_A'].std(ddof=1):.4f} Å")
    print(f"Average Crystallite Size (τ): {slow_df['tau_nm'].mean():.2f} ± {slow_df['tau_nm'].std(ddof=1):.2f} nm")

    print("\n\n--- Fast Scan Results ---")
    print(fast_df)
    print(f"\nAverage Lattice Parameter (a): {fast_df['a_param_A'].mean():.4f} ± {fast_df['a_param_A'].std(ddof=1):.4f} Å")
    print(f"Average Crystallite Size (τ): {fast_df['tau_nm'].mean():.2f} ± {fast_df['tau_nm'].std(ddof=1):.2f} nm")

    #print(fast_df)
    #print(slow_df)
    # fig, ax = plt.subplots()

    # fig.set_size_inches(8, 6)
    # ax.set_xlabel('$2\Theta [°]$')
    # ax.set_ylabel('Intensity [a. u.]')

    # index = 4
    # color = 'tab:purple'

    # for index, array_i in enumerate(fit_params_fast):
    #     t = np.linspace(x_data_fast[index][0]-0.05, x_data_fast[index][-1]+0.05)
    #     y = gauss(t, *array_i)
    #     ax.plot(t, y, label='Gaussian fit - $2\Theta_\mathrm{max}$ = '+f'{fast_scan.iloc[two_theta_indices_fast[index], 0]}°')

    # ax.scatter(fast_scan.iloc[:, 0], fast_scan.iloc[:, 1], label='fast scan', marker='.')
    # plt.legend()
    # plt.show()

    # fig, ax = plt.subplots()

    # fig.set_size_inches(8, 6)
    # ax.set_xlabel('$2\Theta [°]$')
    # ax.set_ylabel('Intensity [a. u.]')

    # for index, array_i in enumerate(fit_params_slow):
    #     t = np.linspace(x_data_slow[index][0]-0.05, x_data_slow[index][-1]+0.05)
    #     y = gauss(t, *array_i)
    #     ax.plot(t, y, label='Gaussian fit - $2\Theta_\mathrm{max}$ = '+f'{slow_scan.iloc[two_theta_indices_slow[index], 0]}°')

    # ax.scatter(slow_scan.iloc[:, 0], slow_scan.iloc[:, 1], label='slow scan', marker='.', c = 'black')
    # plt.legend()
    # plt.show()