#! encoding = utf-8
""" Parse Gaussian output file for scan results. """
import argparse


def arg():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inp', nargs=1, help='input filename')
    parser.add_argument('-out', nargs=1, default='parse.out', help='output filename')
    parser.add_argument('-field', nargs=1, help='Coordinate that is scanned')
    args = parser.parse_args()

    return args


def parse(inp, out, field):

    with open(inp, 'r') as fi, open(out, 'w') as fo:
        fo.write('{:^10s} {:^16s}\n'.format(field, 'SCF'))
        record_list = []
        search_start = False
        for a_line in fi:
            if a_line.startswith(' SCF Done:  E('):
                a_list = a_line.split()
                energy = a_list[4]
            if a_line.strip() == '-- Stationary point found.':
                search_start = True
            if search_start:
                if a_line.startswith(' ! '):
                    a_list = a_line.split()
                    if a_list[1] == field and a_list[4] == '-DE/DX':
                        value = a_list[3]
                        search_start = False
                        record_list.append((value, energy))
        for value, energy in record_list:
            fo.write('{:>10s} {:>16s}\n'.format(value, energy))
        

if __name__ == '__main__':
    
    args = arg()
    parse(args.inp[0], args.out[0], args.field[0])
