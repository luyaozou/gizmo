#! encoding = utf-8
''' This script performs spectral average. It takes xy files with identical
x data but multiple shots of y data, and output the y_average
'''

import numpy as np
import argparse
import os
import libfmt

def average(files):

    current = libfmt.load_single_file(files[0])
    total = np.zeros_like(current)
    eff_count = 0

    for i in range(len(files)):
        current = libfmt.load_single_file(files[i])
        # test if array size matches
        if current.shape == total.shape:
            eff_count += 1
            total += current
            yield total / (eff_count)
        else:
            pass


def arg():
    ''' Input arguments parser. Returns: argparse Object.'''

    parser = argparse.ArgumentParser(description=__doc__,
             epilog='--- Luyao Zou @ https://github.com/luyaozou/ ---')
    parser.add_argument('file', nargs='+', help='files need to be averaged')
    parser.add_argument('-o', nargs=1, help='Output file name string')
    parser.add_argument('-d', action='store_true', help='delete original data')
    return parser.parse_args()


if __name__ == "__main__":

    input_args = arg()
    files = input_args.file
    files.sort()

    if input_args.o:
        out_name = input_args.o[0]
    else:
        out_name = 'Average'

    avg = average(files)    # this is a generator

    for i in range(len(files)):
        np.savetxt('{:s}-{:d}.csv'.format(out_name, i+1), next(avg),
                   delimiter=',', fmt='%.6f')
        print('{:s}-{:d}.csv saved.'.format(out_name, i+1))

    if input_args.d:
        for a_file in files:
            os.remove(a_file)
