#! encoding = utf-8

''' Convert .dat or .csv text data to numpy binary .npy '''

import re
import numpy as np
import argparse
import os


# ------------------------------------------
# ------ MESSAGE CONSTANT DECLARATION ------
# ------------------------------------------
FILE_ERR_MSG = {0: '',                              # Silent
                1: '{:s} does not exist',           # FileNotFoundError
                2: 'cannot read {:s} file format',  # Format Issue
                }


def analyze_fmt(file_name):
    ''' Analyze the data text format: delimiter and header

    Arguments:
    file_name -- data file name, str

    Returns:
    delm -- delimiter, str
    hd   -- number of header rows, int
    eof  -- end of file, boolean
    '''

    hd = 0
    delm = None
    a_file = open(file_name, 'r')
    # match two numbers and a delimiter
    pattern = re.compile('(-?\d+\.?\d*(e|E.?\d+)?)( |\t|,)+(-?\d+\.?\d*(e|E.?\d+)?)')

    for a_line in a_file:
        if re.match('-?\d+\.?\d*(e|E)?.?\d+ *$', a_line):
            # if the first line is a pure number
            delm = ','
            break
        else:
            try:
                delm = pattern.match(a_line).group(3)
                break
            except AttributeError:
                hd += 1

    # check if end of the file is reached
    eof = (a_file.read() == '')

    a_file.close()

    return delm, hd, eof


def err_msg_str(f, err_code, msg=FILE_ERR_MSG):
    ''' Generate file error message string

    Arguments:
    f        -- file name, str
    err_code -- error code, int
    msg      -- error message, dict

    Returns:
    msg_str -- formated error message, str
    '''

    return (msg[err_code]).format(f)


def load(file_name):
    ''' Load single data file & raise exceptions.

    Arguments:
    file_name -- input file name, str

    Returns:
    data -- np.array
    '''

    try:
        delm, hd, eof = analyze_fmt(file_name)
        if eof or isinstance(delm, type(None)):
            print(err_msg_str(file_name, 2))
        else:
            data = np.loadtxt(file_name, delimiter=delm, skiprows=hd)
            return data
    except FileNotFoundError:
        print(err_msg_str(file_name, 1))
        return None
    except ValueError:
        print(err_msg_str(file_name, 2))
        return None


def to_npy(textfile):
    ''' Read text delimited file and save as npy.
    Returns status: 0 skip; 1 saved '''

    data = load(textfile)
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
