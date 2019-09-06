#! encoding = utf-8
''' Calculate line intensity based on a JPL/CDMS catalog file
Output only frequency and intensity, and truncate frequency range '''

import numpy as np
import argparse

def arg():

    parser = argparse.ArgumentParser(description=__doc__,
                                 epilog='--- Luyao Zou, May 2015 ---')
    parser.add_argument('file', nargs=1, help='Catalog file (csvs)')
    parser.add_argument('-t', nargs=1, type=float, default=300., help='Simulation temperature (K)')
    parser.add_argument('-fmin', nargs=1, type=float, help='Frequency min (GHz)')
    parser.add_argument('-fmax', nargs=1, type=float, help='Frequency max (GHz)')
    parser.add_argument('-linear', action='store_true', default=False, help='Linear molecule')
    parser.add_argument('-o', nargs=1, help='Output file name')
    parser.add_argument('-delm', nargs=1, default=' ', help='Output file delimiter')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = arg()

    if isinstance(args.t, float):
        temp = args.t
    else:
        temp = args.t[0]
    if args.linear:
        a = 1
    else:
        a = 1.5

    cat = np.loadtxt(args.file[0], delimiter=',', usecols=(0,1,2,3,4))
    if args.fmin[0]:
        fmin = args.fmin[0] * 1e3
    else:
        fmin = 0
    if args.fmax[0]:
        fmax = args.fmax[0] * 1e3
    else:
        fmax = float('inf')
    idx = np.logical_and(cat[:, 0] > fmin, cat[:, 0] < fmax)
    # truncate catalog
    cat = cat[idx, :]

    inten = np.power(10, cat[:, 2]) * (300/temp)**a \
            / (np.exp(-cat[:, 4]/0.695/300) - np.exp(-(cat[:, 4] + cat[:, 0]*3.33564e-5)/0.695/300)) \
            * (np.exp(-cat[:, 4]/0.695/temp) - np.exp(-(cat[:, 4] + cat[:, 0]*3.33564e-5)/0.695/temp))

    if args.o:
        fout = args.o[0]
    else:
        fout = args.file[0].replace('.csv', '.txt')
    np.savetxt(fout, np.column_stack([cat[:, 0], inten]), delimiter=args.delm,
                fmt=['%.4f', '%.6e'])
    print(fout + ' saved!')
