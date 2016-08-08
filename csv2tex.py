#! encoding = utf8
''' This script converts csv/tsv file to LATEX table format '''

import re
import argparse

def get_delm(testline):
    ''' Analyse delimiter in a line '''
    try:
        return re.search('\d+( |\t|,)+-?\d+', testline).group(1) 
    except AttributeError:
        return False

def get_file_info(filename):
    ''' Test the delimiter and header rows in the file '''
    hd = 0
    try:        # Try to open the file
        with open(filename, 'r') as testfile:
            testline = testfile.readline()
            while not get_delm(testline):
                testline = testfile.readline()
                hd += 1
        try:
            delm = get_delm(testline)
        except AttributeError:
            return None, None
    except OSError:
        return None, None
    except UnicodeDecodeError:
        return None, None

    return hd, delm

def print_info(filename, status):
    ''' Print information on screen '''
    if status:
        print('{:s} saved!'.format(filename))
        return None
    else:
        print('File format not supported!')
        exit()


parser = argparse.ArgumentParser(description=__doc__,
                                 epilog='--- Luyao Zou, Sept 2015 ---')
parser.add_argument('csv', nargs=1, help='table file in csv/tsv format')
parser.add_argument('-o', '--out', nargs=1,
                    help='''Specify output file name without extension''')
args = parser.parse_args()

if args.out:
    out_file = args.out[0]
else:
    out_file = 'csv2tex.tex'

hd, delm = get_file_info(args.csv[0])

status = any(delm)

if not status:
    print_info(out_file, status)

with open(args.csv[0], 'r') as csvfile:
    # read one line
    lines = csvfile.readlines()

# delete header
del lines[0:hd]

with open(out_file, 'w') as output:
    for i in range(len(lines)):
        a_line = lines[i].split(delm)
        output.write(' & '.join(a_line))

print_info(out_file, status)

