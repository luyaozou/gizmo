#! encoding = utf8

'''
This script transforms the SMAP assignment output format into XIAM input format
'''

import re
import argparse


def arg():
    ''' Input arguments parser. Returns: argparse Object.'''

    parser = argparse.ArgumentParser(description=__doc__,
                                    epilog='--- Luyao Zou, Jan 2017 ---')
    parser.add_argument('i', nargs=1, help='Input: SMAP assignment output file')
    parser.add_argument('-o', nargs=1, help='Output: XIAM input file')
    parser.add_argument('-a', action='store_true',
                        help='Append to existing XIAM input')

    return parser.parse_args()


def convert(f, g):
    ''' Convert one assignment. Returns True if EOF '''

    try:
        # first line
        line1 = f.readline().strip().split()
        # second line
        line2 = f.readline().strip().split()
        line1.append(line2[0])
        line1.append(line2[1])
        g.write('{:2s}{:>3s}{:>3s}{:>6s}{:>3s}{:>3s}    S {:1s}  =  {:>11s} MHz   Err  {:s}\n'.format(*line1))
        return False
    except IndexError:
        return True


if __name__ == "__main__":

    input_args = arg()
    f_in = input_args.i[0]

    if input_args.o:
        f_out = input_args.o[0]
    else:
        f_out = re.sub('\.txt$', '.xi', f_in)

    f = open(f_in, 'r')
    if input_args.a:
        g = open(f_out, 'a')
    else:
        g = open(f_out, 'w')

    eof = False

    while not eof:
        eof = convert(f, g)

    f.close()
    g.close()

    print('done')
