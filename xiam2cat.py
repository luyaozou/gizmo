import math
import re
import argparse
import itertools

# column slice for quantum number extraction. Modify if necessary
QN_SLICE = ((0, 3),     # J_up
            (3, 6),     # Ka_up
            (6, 9),     # Kc_up
            (9, 13),   # J_low
            (13, 16),   # Ka_low
            (16, 19),   # Kc_low
            (23, 25))   # Symmetry

# column slice for other stuff. Modify if necessary
OTHER_SLICE = ((32, 44),    # frequency
               (52, 62),    # intensity
               (63, 71))    # G_up


# JPL string format
#          FREQ     UNCERT  LOGINT  DOF   ELOW     GUP   TAGS   QNUP   QNLOW
JPLFMT = '{:>13.4f}{:>8.4f}{:-8.4f}{:>2d}{:>10.4f}{:>3d}{:>11s}{:<8s}{:>s}\n'


def arg():
    ''' Input arguments parser. Returns: argparse Object.'''

    parser = argparse.ArgumentParser(description=__doc__,
                                    epilog='--- Luyao Zou, Aug 2016 ---')
    parser.add_argument('xo', nargs=1, help='Input: XIAM output prediction file')
    parser.add_argument('-out', nargs=1, help='Output: JPL catalog file')

    return parser.parse_args()


def xiam_parse(fin, fout):
    ''' XIAM output format parser.

    Arguments: fin -- input file name
               fout -- output file name

    Returns: None
    '''

    f1h = open(fin, 'r')
    f2h = open(fout, 'w')

    for line in f1h:
        if len(line) < 90:  # lines shorter than 80 chars are headers
            pass
        else:
            qn_tuple, freq_int_tuple = xiam_line_parse(line)
            # seprate QNUP and QNLOW
            qnup = ('{:>2}'*3).format(*(qn_tuple[0:3]))
            qnlow = ('{:>2}'*4).format(*(qn_tuple[3:6] + tuple(qn_tuple[-1])))
            # convert freq from GHz to MHz
            freq = freq_int_tuple[0] * 1e3
            try:
                logint = math.log10(freq_int_tuple[1]) + 3
            except ValueError:
                logint = -9
            gup = int(freq_int_tuple[2])
            # output JPL format
            f2h.write(JPLFMT.format(freq, 0, logint, 3, 1, gup, '60006', qnup, qnlow))

    f1h.close()
    f2h.close()
    print('Converted: {:s} --> {:s}'.format(fin, fout))

    return None


def dec2hex(qn):
    ''' Convert decimal quantum number > 99 to hexidecimal. Returns string '''

    if qn > 99:
        return '{:X}'.format(qn // 10) + str(qn % 10)
    else:
        return str(qn)


def xiam_line_parse(line):
    ''' Parse a single output line from XIAM
    Arguments: line -- string
    Returns: qn_tuple -- tuple of quantum numbers, str in tuple
             other_tuple -- frequency in MHz, & log inten & Gup, floats
    '''

    qn_list = []
    for (start, stop) in QN_SLICE:
        qn = int(''.join(list(itertools.islice(line, start, stop))))
        qn_list.append(dec2hex(qn))

    other_list = []
    for (start, stop) in OTHER_SLICE:
        try:
            other_list.append(float(''.join(list(itertools.islice(line, start, stop)))))
        except ValueError:
            other_list.append(1e-30)

    return tuple(qn_list), tuple(other_list)


if __name__ == "__main__":

    input_args = arg()
    fin = input_args.xo[0]

    if input_args.out:
        fout = input_args.out[0]
    else:
        fout = fin + '.cat'

    xiam_parse(fin, fout)
