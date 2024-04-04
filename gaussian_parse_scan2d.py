#! encoding = utf-8
""" Parse Gaussian output file for 2D scan results. """

import argparse
from operator import itemgetter


def arg():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inp', nargs=1, help='input filename')
    parser.add_argument('-out', nargs=1, default='parse.out', help='output filename')
    parser.add_argument('-field', nargs='+', help='coordinates that are scanned')
    parser.add_argument('-sort', nargs=1, default=0, type=int, help='sorting field')
    parser.add_argument('-pad', nargs=1, default=1, type=int, help='padding result')
    args = parser.parse_args()

    return args


def position(n, x, f):
    """ Change position notation 
    :argument
        n: decimal number
        x: int              base
        f: int              total positions
    :return 
        b: tuple of number in each digit
    """

    b = []
    for i in range(f):
        b.append(n % x)
        n = n // x 
    b.reverse()
    return tuple(b)


def parse(inp, out, fields, ksort, pad):

    fofmt_hd = ' '.join(['{:^10s}'] * len(fields)) + ' {:^16s}\n'
    fofmt = ' '.join(['{:>10.1f}'] * len(fields)) + ' {:>16s}\n'
    if isinstance(ksort, list):
        ksort = ksort[0]
    if isinstance(pad, list):
        pad = pad[0]

    with open(inp, 'r') as fi, open(out, 'w') as fo:
        fo.write(fofmt_hd.format(*fields, 'SCF'))
        record_list = []
        value_list = [""] * len(fields)
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
                    if a_list[4] == '-DE/DX':
                        try:
                            idx = fields.index(a_list[1])
                            value_list[idx] = float(a_list[3])
                            continue
                        except ValueError:
                            pass
                elif a_line.startswith(' Iteration') or a_line.startswith(' Summary'):
                    search_start = False
                    record_list.append((*value_list, energy))
        
        n_space = round(len(record_list) ** (1/len(fields)))
        n_block = round(len(record_list) / n_space)
        field_ranges = []
        for i in range(len(fields)):
            v_max = max(list(x[i] for x in record_list))
            v_min = min(list(x[i] for x in record_list))
            field_ranges.append(v_max - v_min)

        # pad result, if pad > 1
        # this means add extra ptp of each field to its value
        for i_pad in range(pad ** len(fields)):
            add_to_field = position(i_pad, pad, len(fields))
            for i in range(n_block):
                block_list = []
                for rec in record_list[i * n_space: (i+1) * n_space]:
                    block_list.append(rec)
                # sort with first field in block list
                block_list.sort(key=itemgetter(ksort))
                # write the sorted list
                for rec in block_list:
                    new_rec = []
                    for ri, r in enumerate(rec[:-1]):
                        new_r = r + add_to_field[ri] * round(field_ranges[ri])
                        new_rec.append(new_r)
                    fo.write(fofmt.format(*new_rec, rec[-1]))
                # add extra space for each data block
                fo.write('\n')


if __name__ == '__main__':
    
    args = arg()
    parse(args.inp[0], args.out[0], args.field, args.sort, args.pad)
