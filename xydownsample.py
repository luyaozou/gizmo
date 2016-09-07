#! encoding = utf-8
'''
Downsample xy file by averaging
'''

FILTER_FREQ = 5 * 12 # 60 / band

import numpy as np
import argparse
import re
import libfmt
from scipy.signal import decimate


def bsfilter(y):

    x = np.fft.rfftfreq(len(y))
    yfft = np.fft.rfft(y)

    bsfreq = FILTER_FREQ / len(y)       # bandstop center freq
    bswin = gaussian_win(x, bsfreq)

    y_bs = np.fft.irfft(yfft - yfft * bswin)

    return y_bs


def gaussian_win(x, freq):
    ''' return gaussian window center at freq with sigma = 0.1freq '''

    sigma = 0.1 * freq

    return  np.exp(-np.power((x-freq)/(2*sigma), 2))


def downsample(x, ds):
    ''' Down sample x by a factor of ds '''

    x_r, tail = reshape(x, ds)
    x_avg = np.average(x_r, axis=1)
    if tail:
        x_tail = np.average(tail)
        x_avg = np.append(x_avg, x_tail)
    else:
        pass

    return x_avg


def reshape(x, ds):
    ''' Reshape 1D array x to 2D array, with each row to be averaged '''

    tail = len(x) % ds
    row = len(x) // ds
    col = ds

    if tail:
        x_r = x[:-tail].reshape((row, col))
        x_tail = x[-tail:]
    else:
        x_r = x.reshape((row, col))
        x_tail = None

    return x_r, x_tail


def arg():
    ''' Input arguments parser. Returns: argparse Object.'''

    parser = argparse.ArgumentParser(description=__doc__,
                                    epilog='--- Luyao Zou, Aug 2016 ---')
    parser.add_argument('f', nargs=1, help='data file')
    parser.add_argument('-ds', nargs=1, type=int,
                        help='downsample factor')
    parser.add_argument('-o', nargs=1, help='output file')

    return parser.parse_args()


# ---------------- main routine ----------------
if __name__ == '__main__':

    input_args = arg()
    ds = input_args.ds[0]

    data = libfmt.load_single_file(input_args.f[0])
    x = data[:, 0]
    y = data[:, 1]

    yf = bsfilter(y)

    x_ds = downsample(x, ds)
    y_ds = downsample(yf, ds)

    if input_args.o:
        outfile = input_args.o[0]
    else:
        outfile = re.sub('\..{1,4}$', '_ds.csv', input_args.f[0])
    np.savetxt(outfile, np.column_stack((x_ds, y_ds)), delimiter=',',
               comments='', header='freq,intensity')
    print(outfile + ' saved.')
