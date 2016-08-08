#! encoding = utf8
''' Quickly convert space delimited catalog file into comma delimited '''
import argparse
import re

def comma_del(line):
    ''' Insert comma at certain places following JPL catalog format '''
    comma_indices = [13, 22, 31, 34, 45, 49, 61, 74]
    for index in comma_indices:
        line = line[:index] + ',' + line[index:]
    return line

parser = argparse.ArgumentParser(description=__doc__,
                                 epilog='--- Luyao Zou, May 2015 ---')
parser.add_argument('file', nargs='+', help='File names')
args = parser.parse_args()

for file_name in args.file:
    try:
        output = re.sub('cat$', 'csv', file_name)
        catalog_out = []
        with open(file_name, 'r') as cat_file:
            catalog = cat_file.readlines()
        for line in catalog:
            # insert comma at given places 
            line_out = comma_del(line)
            # remove all spaces
            line_out = re.sub(' +', '', line_out)
            catalog_out.append(line_out)
        with open(output, 'w') as cat_out:
            for line in catalog_out:
                cat_out.write(line)
        print('{:s} saved!'.format(output))
    except FileNotFoundError:
        print('{:s} not found! Skip'.format(file_name))
    except UnicodeDecodeError:
        print('File format error. Skip')
    except ValueError:
        print('Import error. Skip')
    except:
        print('Unexpected error. Skip')
        raise


