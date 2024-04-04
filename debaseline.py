#! encoding = utf-8

''' This scripts removes background spectrum (baseline of standing wave)
using experimental data  '''

import numpy as np
import argparse
from scipy import interpolate as intp


def downsample(y, step=20, is_middle=True):
    ''' Downsample data
    Arguments
        y: np1darray            data to be downsampled
        step: int               step size (points)
        is_middle: bool
            if True, extract out data points at index = step//2 + step*n
            if False, extract out data points at index = 0 + step*n
    Returns
        y2: np1darray           downsampled data array
    '''

    # perform this downsample using reshape.
    # we just need to reshape y to the downsampled array shape, and take out the first column.

    n = len(y) // step
    tail = len(y) % step
    # chop off the tail
    if tail:
        y2 = y[:-tail]
    else:
        y2 = y
    # reshape
    y2 = np.reshape(y2, (n, step))

    if is_middle:
        return y2[:, step//2]
    else:
        return y2[:, 0]


def smooth(y, win=20):
    ''' Smooth data with Gaussian window. The window is +/- 5 sigma wide
    Arguments
        y: np1darray    data to be smoothed
        win: int        Gaussian FWHM (number of points)
    Returns
        np1darray       smoothed data
    '''

    sigma = win / (2*np.sqrt(2*np.log(2)))
    x = np.linspace(-5*win, 5*win, num=10*win+1, endpoint=True)
    gaussian = np.exp(-(x/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))

    return np.convolve(y, gaussian, mode='same')


def intp_spline(x_norm, y, step=20):
    ''' Interplote use cubic spline
    Arguments:
        x_norm: np1darray       normalized x [-1, 1]
        y: np1darray            corresponding y data
        step: int               smoothing & downsampling step size
    Returns
        scipy.interpolate function
    '''

    ys = smooth(y, win=step)
    x2 = downsample(x_norm, step=step, is_middle=True)
    y2 = downsample(ys, step=step, is_middle=True)

    cs = intp.CubicSpline(x2, y2)

    return cs


def db(data, base, sens_ratio, unit):
    ''' Main debaseline routine
    Arguments:
        data: np2darray         spectral data array
        base: np2darray         baseline data array
        sens_ratio: float       sensivity ratio spectrum/baseline
        unit: float             unit conversion for x array
    Returns
        data_db: np2darray      debaselined spectral data array
    '''

    x_data = data[:, 0] * unit
    y_data = data[:, 1]
    x_base = base[:, 0] * unit
    y_base = base[:, 1]

    # get interpolated baseline function
    x_base_norm = np.linspace(-1, 1, num=len(y_base), endpoint=True)
    cs = intp_spline(x_base_norm, y_base)

    # interpolate baseline
    xmin = np.min(x_base)
    xmax = np.max(x_base)
    x_data_norm = (2*x_data - xmin - xmax) / (xmax - xmin)
    # subtract baseline
    y_db = y_data - cs(x_data_norm) / sens_ratio

    return np.column_stack((x_data, y_db))


def arg():

    parser = argparse.ArgumentParser(description=__doc__,
                                 epilog='--- Luyao Zou, Dec 2019 ---')
    parser.add_argument('files', nargs=2,
                        help='Spectral file (dat) and then baseline file (dat)')
    parser.add_argument('-skip', nargs=1, type=int, default=0,
                        help='Skip # of lines in the beginning of the data file. Default=0')
    parser.add_argument('-sens', nargs=2, type=float, default=[1., 1.],
                        help='Sensivity of the spectrum (first) and baseline (second). Default=1 1')
    parser.add_argument('-unit', nargs=1, type=float, default=1.,
                        help='Unit conversion. The x array will be multiplied by the input value. Default=1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = arg()

    data_file = args.files[0]
    base_file = args.files[0]
    data_sens = args.sens[0]
    base_sens = args.sens[1]
    out_file = data_file.replace('.dat', '_db.dat')

    hd = []
    with open(data_file, 'r') as f:
        for i in range(args.skip):
            hd.append(f.readline())

    data = np.loadtxt(data_file, skiprows=args.skip)
    base = np.loadtxt(base_file, skiprows=args.skip)
    data_db = db(data, base, data_sens/base_sens, args.unit)

    np.savetxt(out_file, data_db, header='\n'.join(hd), comments='', fmt=['%11.3f', '%15.8e'])
