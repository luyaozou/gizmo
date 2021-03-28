#! encoding = utf8
''' Quickly convert space delimited catalog file into space delimited
For GOBASIC '''
import argparse


def split(line):
    ''' Insert comma at certain places following JPL catalog format '''
    comma_indices = [0, 13, 21, 29, 31, 41, 44]
    a_list = []
    for i1, i2 in zip(comma_indices[:-1], comma_indices[1:]):
        a_list.append(line[i1:i2])
    a_list.append('\n')

    return ' '.join(a_list)


parser = argparse.ArgumentParser(description=__doc__,
                                 epilog='--- Luyao Zou, May 2015 ---')
parser.add_argument('file', nargs='+', help='File names')
args = parser.parse_args()

for file_name in args.file:
    try:
        output = file_name.replace('.cat', '.txt')
        with open(file_name, 'r') as cat_file, open(output, 'w') as cat_out:
            for line in cat_file:
                # insert comma at given places
                line_out = split(line)
                cat_out.write(line_out)
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


