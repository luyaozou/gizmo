#! encoding = utf-8

''' Convert .dat or .csv text data to numpy binary .npy '''

import numpy as np
import argparse
import os
import libfmt


def to_npy(textfile):
    ''' Read text delimited file and save as npy.
    Returns status: 0 skip; 1 saved '''

    data = libfmt.load_single_file(textfile)
    if isinstance(data, type(None)):
        print('pass')
        return 0
    else:
        # replace the last string after a dot that is 1-4 long
        # (assume as extenstion)
        outfile = re.sub('\..{1,4}$', '.npy', textfile)
        np.save(outfile, data)
        print('{:s} saved.'.format(outfile))
        return 1


def arg():
    ''' Input arguments parser. Returns: argparse Object.'''

    parser = argparse.ArgumentParser(description=__doc__,
                                    epilog='--- Luyao Zou, Aug 2016 ---')
    parser.add_argument('f', nargs='+', help='data files')
    parser.add_argument('-d', action='store_true',
                        help='Delete original text file')

    return parser.parse_args()


# ---------------- main routine ----------------
if __name__ == '__main__':

    input_args = arg()
    file_list = input_args.f
    for a_file in file_list:
        status = to_npy(a_file)
        if (status and input_args.d):
            os.remove(a_file)
