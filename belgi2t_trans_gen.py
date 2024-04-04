#! encoding = utf-8
""" Generate transitions for BELGI-2tops prediction """

import argparse


def arg():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('j', nargs=1, help='maximum J value')
    args = parser.parse_args()
    return args


def trans_gen(jmax):
    """ Generate transitions. """
    line_fmt = '{:>12.5f}{:>2d}{:>4d}{:>5d}{:>3s}{:>3d}{:>3d}{:>4d}{:>5d}{:>3s}{:>3d}' \
               '  1   1{:0>3d}   {:<s}\n'

    for j1 in range(jmax):
        # generate the list of all lower state J,Ka,Kc
        for k_ord1 in range(1, 2*j1+2):
            ka1 = k_ord1 // 2
            if k_ord1 % 2:
                kc1 = j1 - ka1
            else:
                kc1 = j1 + 1 - ka1
            # transitions are between |Δk_ord| <= 2
            # R type transitions
            j2 = j1 + 1
            for k_ord2 in range(max(1, k_ord1-2), min(2*j2+1, k_ord1+2)+1):
                ka2 = k_ord2 // 2
                sgn_ka2 = 1 if k_ord2 % 2 else -1
                sgn_ka1 = 1 if k_ord1 % 2 else -1
                # for R branch, the signs of A states are the same
                yield line_fmt.format(0, 0, j2, ka2, '+', j2+1, 0, j1, ka1, '+', j1+1, 0, 'A')
                yield line_fmt.format(0, 0, j2, ka2, '-', j2+1, 0, j1, ka1, '-', j1+1, 0, 'A')
                yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2+1, 0, j1, sgn_ka1 * ka1, ' ', j1+1, 0, 'E1')
                yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2+1, 0, j1, sgn_ka1 * ka1, ' ', j1+1, 0, 'E2')
                yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2+1, 0, j1, sgn_ka1 * ka1, ' ', j1+1, 0, 'E3')
                yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2+1, 0, j1, sgn_ka1 * ka1, ' ', j1+1, 0, 'E4')
            # Q type transitions
            j2 = j1
            for k_ord2 in range(max(1, k_ord1-2), min(2*j2+1, k_ord1+2)+1):
                ka2 = k_ord2 // 2
                sgn_ka2 = 1 if k_ord2 % 2 else -1
                sgn_ka1 = 1 if k_ord1 % 2 else -1
                if ka2 == ka1:      # identical levels, pass
                    pass
                else:
                    # for Q branch, the signs of A states are opposite
                    yield line_fmt.format(0, 0, j2, ka2, '-', j2 + 1, 0, j1, ka1, '+', j1 + 1, 0, 'A')
                    yield line_fmt.format(0, 0, j2, ka2, '+', j2 + 1, 0, j1, ka1, '-', j1 + 1, 0, 'A')
                    yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2 + 1, 0, j1, sgn_ka1 * ka1, ' ', j1 + 1, 0, 'E1')
                    yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2 + 1, 0, j1, sgn_ka1 * ka1, ' ', j1 + 1, 0, 'E2')
                    yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2 + 1, 0, j1, sgn_ka1 * ka1, ' ', j1 + 1, 0, 'E3')
                    yield line_fmt.format(0, 0, j2, sgn_ka2 * ka2, ' ', j2 + 1, 0, j1, sgn_ka1 * ka1, ' ', j1 + 1, 0, 'E4')


if __name__ == '__main__':

    args = arg()
    with open('output.txt', 'w') as f:
        for a_line in trans_gen(int(args.j[0])):
            f.write(a_line)
