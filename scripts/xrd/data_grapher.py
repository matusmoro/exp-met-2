import numpy as np
import pandas as pd
from scipy import signal
from scipy import optimize
from scipy import constants as c
from matplotlib import pyplot as plt

def gauss(x, mu, A, B):
    return A*np.exp((x-mu)**2/B)

class Data:

    def __init__(self, filename, scan):

        self.data = pd.read_csv(filename, engine='python', header=None, skiprows=375, skipfooter=3, sep=' ', encoding='unicode_escape')
        self.angle = self.data[0]
        self.intensity = self.data[1]

        if scan == 'fast':
            threshold = 580
        elif scan == 'slow':
            threshold = 2000

        [self.max_index, self.max_height] = signal.find_peaks(self.intensity, height=threshold, distance=100)
        self.max_height = self.max_height['peak_heights']

        self.peak_params = np.array((len(self.max_index), 3))

        width = 30

        #for index, max in enumerate(self.max_index):
            #self.peak_params[index] = optimize.curve_fit(gauss, self.angle[max-width:max+width], self.intensity[max-width:max+width], maxfev=10000)[0]

    def graph(self, filename):

        fig, ax = plt.subplots()

        ax.set_xlabel('$2\Theta$ [°]')
        ax.set_ylabel('I [a.u.]')

        ax.scatter(self.angle[self.max_index], self.max_height, c='r')
        ax.plot(self.angle, self.intensity)
        
        plt.savefig(f'/home/matusmoro/school/exp_met_2_xrd/graphs/xrd/{filename}.png')

if __name__ == "__main__":
    fast = Data(r'/home/matusmoro/school/exp_met_2_xrd/data/xrd/fast_sul.ras', 'fast')
    slow = Data(r'/home/matusmoro/school/exp_met_2_xrd/data/xrd/slow_sul.ras', 'slow')

    print('Building plots')
    fast.graph('fast_test')
    slow.graph('slow_test')
    print('Plots saved')

