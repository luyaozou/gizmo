#! encoding = utf-8
""" Interactive library to convert xyz coordinate to zmatrix coordinate """

import math
import argparse
import numpy as np
from numpy.linalg import norm


def arg():
    """ Argutnem parser """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs=1, help='input XYZ & Z-matrix order ')
    return parser.parse_args()


def bondlen(q1, q2):
    """ Calculate bond length between points q1 and q2."""

    return norm(q2 - q1)


def angle(v1, v2):
    """ Calculate angle between vectors v1 and v2 """

    angle_rad = math.acos(np.vdot(v1, v2) / (norm(v1) * norm(v2)))

    return angle_rad / math.pi * 180


def bondang(q1, q2, q3):
    """ Calculate bond angle q1-q2-q3 """

    v1 = q2 - q1
    v2 = q2 - q3

    return angle(v1, v2)


def dihedral(q1, q2, q3, q4):
    """ Calculate dihedral angle between q1-q2-q3 and q2-q3-q4 """

    v12 = q2 - q1
    v23 = q3 - q2
    v34 = q4 - q3

    n123 = np.cross(v12, v23)
    n234 = np.cross(v23, v34)

    return angle(n123, n234)


def process(filename):
    """ Process file """

    q_list = []
    atom_list = []
    # read xyz coordinate
    with open(filename, 'r') as f:
        for a_line in f:
            if a_line:
                if a_line.startswith('*') or a_line.startswith('-'):
                    # separation
                    break
                else:
                    a_list = a_line.split()
                    if len(a_list) == 4:    # cartesian
                        q = list(float(x) for x in a_list[1:])
                        q_list.append(np.array(q))
                        atom_list.append(a_list[0])
        current_idx = 0
        for a_line in f:
            a_list = a_line.split()
            atom = a_list[0]
            if len(a_list) == 0:
                pass
            if len(a_list) == 1:
                print(atom)
            elif len(a_list) == 2:
                idx = int(a_list[1])
                q1 = q_list[current_idx]
                q2 = q_list[idx-1]
                r = bondlen(q1, q2)
                print(atom, idx, '{:.6f}'.format(r))
            elif len(a_list) == 3:
                idx2 = int(a_list[1])
                idx3 = int(a_list[2])
                q1 = q_list[current_idx]
                q2 = q_list[idx2-1]
                q3 = q_list[idx3-1]
                r = bondlen(q1, q2)
                ang = bondang(q1, q2, q3)
                print(atom, idx2, '{:.6f}'.format(r), idx3, '{:8.3f}'.format(ang))
            elif len(a_list) == 4:
                idx2 = int(a_list[1])
                idx3 = int(a_list[2])
                idx4 = int(a_list[3])
                q1 = q_list[current_idx]
                q2 = q_list[idx2-1]
                q3 = q_list[idx3-1]
                q4 = q_list[idx4-1]
                r = bondlen(q1, q2)
                ang = bondang(q1, q2, q3)
                dih = dihedral(q1, q2, q3, q4)
                print(atom, idx2, '{:.6f}'.format(r), idx3, '{:8.3f}'.format(ang), idx4, '{:8.3f}'.format(dih))
            current_idx += 1


if __name__ == "__main__":

    args = arg()
    process(args.file[0])
