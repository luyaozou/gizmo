#! encoding = utf8
''' Parse the JPL format .lwa file into xy file '''

import re
import numpy as np
import argparse


def file_filter(file_list, pattern):
    ''' Filter out file names with a given pattern '''
    extract = []
    prog = re.compile(pattern)
    for file_name in file_list:
        if prog.search(file_name):
            extract.append(file_name)

    return extract


def flatten(nested_list):
    ''' flatten nested list '''
    flat = []
    for x in nested_list:
         if isinstance(x, list):
             for x2 in flatten(x):
                flat.append(x2)
         else:
             flat.append(x)

    return flat


def header(lines):
    ''' Get line numbers of each header '''

    re_header = re.compile('DATE')
    # header's line number
    hd_line = []
    for line in lines:
        # this is the header
        if re_header.match(line):
            hd_line.append(lines.index(line))
    hd_line.sort()

    return hd_line


def freq_calc(line):
    ''' Calculate frequency array from header information '''
    numbers = line.split()
    freq_start = float(numbers[0])
    freq_step = float(numbers[1])
    freq_point = int(numbers[2])
    freq = np.arange(freq_point)*freq_step + freq_start

    return freq


def data_parser(lines):
    ''' Parse data '''
    # flatten list
    data = []
    for line in lines:
        data.append(line.split())

    # convert to nparray
    return np.array(flatten(data), dtype=float)


def sequence(start, end, step=1):
    ''' Generate a number sequence '''

    x = int(start)
    while x <= int(end):
        yield x
        x += int(step)


def scan_num(hd_line_n, num_str):
    ''' Read num_str format and return header line numbers to be extracted '''

    if num_str:
        if re.search(',', num_str):   # comma delimited
            nums = num_str.split(',')
            hd_extract_n = list(int(x) for x in nums)
        elif re.search('^\d+-\d+-?\d?$', num_str): # - delimited
            nums = num_str.split('-')
            hd_extract_n = sequence(*nums)
        else:   # single number
            try:
                hd_extract_n = [int(num_str)]
            except ValueError:
                print('Sequence format error.')
                exit()
        return list(hd_line_n[x] for x in hd_extract_n)
    else:
        return hd_line_n       # extract all scans


def parse(file_name, num_str, hd_bool):
    ''' Parse file. Input arguments:
    file_name:  .lwa file name, str
    num_str:    controlling string for scan numbers, str
    hd_bool:    header control, boolean

    Returns None. Directly outputs parsed file.
    '''

    with open(file_name) as a_file:
        lines = a_file.readlines()

    hd_line_n = header(lines)
    hd_pattern = re.compile('(\d{2}-\d{2}-\d{4}).+SENS ((\d+.\d+)|(\d+e-?\d+)) TAU.+')

    hd_extract_n = scan_num(hd_line_n, num_str)

    for n in hd_extract_n:
        # get information
        hd_info = hd_pattern.search(lines[n]).groups()
        # calculate frequency
        freq = freq_calc(lines[n+2])

        # get next header line number
        try:
            line_next = hd_line_n[hd_line_n.index(n)+1]
        except IndexError:
            line_next = None

        if line_next:
            data = data_parser(lines[n+3:line_next])
        else:
            data = data_parser(lines[n+3:])
        outname = outnamefmt(file_name, hd_info, hd_line_n.index(n))

        # control the header line in the output file
        if hd_bool:
            hd = lines[n]
        else:
            hd = ''

        np.savetxt(outname, np.column_stack((freq, data)), delimiter=',',
                   header=hd, fmt='%.3f', comments='#')
        print('{:s} saved'.format(outname))

    return None


def outnamefmt(file_name, hd_info, index):
    ''' Format output name '''

    a_file = re.match('(.+).lwa', file_name).group(1)

    outname = '{:s}_{:s}_{:0g}mV_{:d}.csv'.format(a_file, hd_info[0], float(hd_info[1])*1e3, index+1)

    return outname


def arg():
    ''' Parse arguments '''

    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='--- Luyao Zou, Oct 2016 ---')
    parser.add_argument('file', nargs='+', help='File names')
    parser.add_argument('-n', nargs=1, help='''Scan number to be extracted.
                        Allow 4 formats. Eg. 1 single number; 1-3 number series;
                        1-9-2; number series with step size; 1,2,4,7 individual
                        numbers delimited by comma''')
    parser.add_argument('-header', action='store_true', help='''Insert header
                        information in the output csv file''')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = arg()
    # filter file name
    file_list = file_filter(args.file, '.lwa\Z')

    if args.n:
        num_str = args.n[0]
    else:
        num_str = ''

    for file_name in file_list:
        print('{:->20}'.format('-'))
        parse(file_name, num_str, args.header)
        print('{:s} parsed'.format(file_name))
