# encoding = utf-8
""" Convert the param+unc to scientific notation. """

from pyfit.libs.spec import scifmt
import argparse



def arg():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('f', nargs=1, help='input file name')
    args = parser.parse_args()
    return args


def run(inp):

    with open(inp, 'r') as f:
        for a_line in f:
            if a_line.strip():
                a_list = a_line.split()
                if len(a_list) == 3:
                    name = a_list[0]
                    par = float(a_list[1])
                    err = float(a_list[2])
                else:
                    name = ''
                    par = float(a_list[0])
                    err = float(a_list[1])
                print('{:<18s}{:>20s}'.format(name, scifmt(par, err, exp='plain')))


if __name__ == '__main__':

    args = arg()
    run(args.f[0])
