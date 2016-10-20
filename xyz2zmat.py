#! encoding = utf-8
''' Interactive library to convert xyz coordinate to zmatrix coordinate '''

import math
import numpy as np
from numpy.linalg import norm


def bondlen(q1, q2):
    ''' Calculate bond length between points q1 and q2.'''

    return norm(q2 - q1)


def angle(v1, v2):
    ''' Calculate angle between vectors v1 and v2 '''

    angle_rad = math.acos(np.vdot(v1, v2) / (norm(v1) * norm(v2)))

    return angle_rad / math.pi * 180


def bondang(q1, q2, q3):
    ''' Calculate bond angle q1-q2-q3 '''

    v1 = q2 - q1
    v2 = q2 - q3

    return angle(v1, v2)


def dihedral(q1, q2, q3, q4):
    ''' Calculate dihedral angle between q1-q2-q3 and q2-q3-q4 '''

    v12 = q2 - q1
    v23 = q3 - q2
    v34 = q4 - q3

    n123 = np.outer(v12, v23)
    n234 = np.outer(v23, v34)

    return angle(n123, n234)


if __name__ == "__main__":
    pass
