#! encoding = utf-8

""" This is an interactive Z-matrix component calculator based on cartesian coordinates. """

import numpy as np
import argparse


def arg():
    """ Argument parser """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs=1, 
    help="""
    After reading the file, type in the commands to calculate 
    r(#1, #2)  bond length between atom #1 and #2
    a(#1, #2, #3)  bond angle between atom #1, #2, and #3
    d(#1, #2, #3, #4)  dihedral angle between atom #1, #2, #3, and #4
    """)
    return parser.parse_args()


def readfile(filename):
    """ Read xyz file
    :return
        coord_list: list of (x, y, z) coordinates for all atoms
    """
    coord_list = []
    with open(filename, 'r') as f:
        for a_line in f:
            if a_line.strip():
                a_list = a_line.split()
                n = len(a_list)
                if n >= 4:  # cartesian
                    q = list(float(x) for x in a_list[n-3:])
                    coord_list.append(np.array(q))
                    print(f'{len(coord_list):>2d}', a_line, end='')
    return coord_list


def bond_length(xyz, i, j):
    """ Calculate bond length between atom i and j """
    return np.linalg.norm(xyz[i] - xyz[j])


def bond_angle(xyz, i, j, k):
    """ Calculate bond angle between atom i, j, and k """
    v1 = xyz[i] - xyz[j]
    v2 = xyz[k] - xyz[j]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos_theta) * 180 / np.pi


def dihedral_angle(xyz, i, j, k, l):
    """ Calculate dihedral angle between atom i, j, k, and l """
    b1 = xyz[j] - xyz[i]
    b2 = xyz[k] - xyz[j]
    b3 = xyz[l] - xyz[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.arctan2(y, x) * 180 / np.pi


def extract_substring(s):
    start_idx = s.find('(')
    if start_idx == -1:
        return None

    end_idx = s.find(')', start_idx)
    if end_idx == -1:
        return None

    return s[start_idx + 1:end_idx]


if __name__ == '__main__':

    args = arg()
    xyz_list = readfile(args.file[0])
    print()
    print("Enter command r(1,2), a(1,2,3) or d(1,2,3,4) or 'E(xit)/Q(uit)' to quit: ")
    while True:
        command = input().strip()
        if command in ['exit', 'E', 'Q', 'e', 'q', 'quit']:
            break
        try:
            if command.startswith('r'):
                i, j = extract_substring(command).split(',')
                i, j = int(i), int(j)
                print(f"Bond length r({i}, {j}): {bond_length(xyz_list, i-1, j-1):.8f}")
            elif command.startswith('a'):
                i, j, k = extract_substring(command).split(',')
                i, j, k = int(i), int(j), int(k)
                print(f"Bond angle a({i}, {j}, {k}): {bond_angle(xyz_list, i-1, j-1, k-1):.8f}")
            elif command.startswith('d'):
                i, j, k, l = extract_substring(command).split(',')
                i, j, k, l = int(i), int(j), int(k), int(l)
                print(f"Dihedral angle d({i}, {j}, {k}, {l}): {dihedral_angle(xyz_list, i-1, j-1, k-1, l-1):.8f}")
            else:
                print("Invalid command. Use 'r', 'a', or 'd'.")
        except Exception as e:
            print(f"Error: {e}")

