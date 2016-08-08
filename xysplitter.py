#! encoding = utf-8
''' This scripts splits a large x,y file into smaller pieces. x is sorted.
Operates in two mode: window mode and points mode.
In window mode, you specify the starting and ending x values.
In points mode, you specify the number of points in each smaller piece.
'''

import argparse
import re
import libfmt


def arg():

    parser = argparse.ArgumentParser(description=__doc__,
             epilog='--- Luyao Zou @ https://github.com/luyaozou/ ---')
    parser.add_argument('file', nargs='+', help='CSV file list')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-w', action='store_true',
                      help='''Window selection mode.
                           Specify starting and ending x values''')
    mode.add_argument('-p', action='store_true',
                      help='Points mode. Specify number of points')
    parser.add_argument('-pts', nargs=1, type=int,
                        help='Split window (points). Default 1 million')
    parser.add_argument('-start', nargs=1, type=float, help='Start x')
    parser.add_argument('-end', nargs=1, type=float, help='End x')

    return parser.parse_args()


def point_split(a_file, pts):

    raw = open(a_file, 'r')
    row_cnt = 1
    split_cnt = 1
    out_name = re.sub('\.csv$', '', a_file)
    split = open('{:s}-{:d}.csv'.format(out_name, split_cnt), 'w')

    for line in raw:
        if row_cnt < pts:
            split.write(line)
            row_cnt += 1
        else:
            split.close()
            print('Split - {:d}'.format(split_cnt))
            row_cnt = 1
            split_cnt += 1
            split = open('{:s}-{:d}.csv'.format(out_name, split_cnt), 'w')

    raw.close()
    split.close()
    return None


def window_split(a_file, x1, x2):

    delm, hd, eof = libfmt.txt_fmt(a_file)

    if eof or isinstance(delm, type(None)):
        print(libfmt.err_msg_str(file_name, 2))
    else:
        raw = open(a_file, 'r')
        out_name = re.sub('\.csv$', '-split.csv', a_file)
        split = open(out_name, 'w')

        for line in raw:
            x, y = line.split(delm)
            if (float(x) > x1 and float(x) < x2):
                split.write(line)
            else:
                pass

        print('x between {:.4g} and {:.4g} extracted to {:s}'.format(x1, x2, out_name))
        raw.close()
        split.close()

    return None


if __name__ == '__main__':

    input_args = arg()

    if input_args.w:
        try:
            x1 = input_args.start[0]
            x2 = input_args.end[0]
        except TypeError:
            print('Window mode must specify staring and ending x values')
            exit()
        for a_file in input_args.file:
            window_split(a_file, x1, x2)
    elif input_args.p:
        if input_args.pts:
            pts = input_args.pts[0]
        else:
            pts = 1000000
        for a_file in input_args.file:
            point_split(a_file, pts)
