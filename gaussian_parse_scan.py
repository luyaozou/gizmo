#! encoding = utf-8
""" Parse Gaussian output file for scan results. """
import argparse


def arg():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inp', nargs=1, help='input filename')
    parser.add_argument('-out', nargs=1, default='parse.out', help='output filename')
    parser.add_argument('-field', nargs=1, help='Coordinate that is scanned')
    parser.add_argument('-a', action='store_true', help='Append to data')
    args = parser.parse_args()

    return args


def parse(inp, out, field, w_mode):

    with open(inp, 'r') as fi, open(out, w_mode) as fo:
        if w_mode == 'w':
            fo.write('{:^10s} {:^16s}\n'.format(field, 'SCF'))
        else:   # if w_mode =='a', do not write header
            pass
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
                    # depending on the scan options, '-DE/DX' is not necessarily at a_list[4]
                    # we need to find the correct column that labels the 'value'
                    if a_list[1] == 'Name':
                        i_value = a_list.index('Value')
                    if a_list[1] == field:
                        value = a_list[i_value]
                        search_start = False
                        record_list.append((value, energy))
        for value, energy in record_list:
            fo.write('{:>10s} {:>16s}\n'.format(value, energy))
        

if __name__ == '__main__':
    
    args = arg()
    if args.a:
        w_mode = 'a'
    else:
        w_mode = 'w'
    parse(args.inp[0], args.out[0], args.field[0], w_mode)
