#! encoding = utf-8
''' This script performs spectral average. It average each single data points
in a set of data files with same dimensions.
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
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument('-csv', action='store_true',
                     help='Save file in commat delimited format | DEFAULT')
    fmt.add_argument('-txt', action='store_true',
                     help='Save file in white space delimited format')
    fmt.add_argument('-npy', action='store_true',
                     help='Save file in numpy binary format')

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

    if input_args.npy:
        for i in range(len(files)):
            np.save('{:s}-{:d}.npy'.format(out_name, i+1), next(avg))
            print('{:s}-{:d}.npy saved.'.format(out_name, i+1))
    else:
        if input_args.txt:
            delm = ' '
            ext = 'txt'
        else:
            delm = ','
            ext = 'csv'
        for i in range(len(files)):
            np.savetxt('{:s}-{:d}.{:s}'.format(out_name, i+1, ext), next(avg),
                       delimiter=delm, fmt='%.6f')
            print('{:s}-{:d}.{:s} saved.'.format(out_name, i+1, ext))

    if input_args.d:
        for a_file in files:
            os.remove(a_file)
