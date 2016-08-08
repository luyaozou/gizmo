#! encoding = utf-8
''' This is a shared library for xy file format analyzer.
Recognize tab/space/comma delimited text and numpy .npy binary
'''

import re
import numpy as np

# ------------------------------------------
# ------ MESSAGE CONSTANT DECLARATION ------
# ------------------------------------------
FILE_ERR_MSG = {0: '',                              # Silent
                1: '{:s} does not exist',           # FileNotFoundError
                2: '{:s} format is not supported',  # Format Issue
                }


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


def txt_fmt(file_name):
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


def load_single_file(file_name):
    ''' Load single data file & raise exceptions.

    Arguments:
    file_name -- input file name, str

    Returns:
    data -- np.array
    '''

    # test if the file is .npy binary
    if re.search('\.npy$', file_name):
        try:
            data = np.load(file_name, mmap_mode='r', allow_pickle=False)
            return data
        except IOError:
            print(err_msg_str(file_name, 2))
            return None
        except ValueError:
            print(err_msg_str(file_name, 3))
            return None
    else:
        try:
            delm, hd, eof = analyze_txt_fmt(file_name)
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
