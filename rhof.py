#! encoding = utf-8

""" Read xyz file and calculate moments of inertia, rotational constants,
and those of internal rotor

The internal rotor top is specified by first labeling the rotor axis, 
and then its atom indices

Sample input format:

N            0.045851    1.430604    0.000000
C            0.000000    0.160884    0.000000
H            1.009858    1.765235    0.000000
C           -1.340657   -0.522812    0.000000
H           -2.139230    0.213600    0.000000
H           -1.442462   -1.166617    0.876838
H           -1.442462   -1.166617   -0.876838
C            1.203944   -0.750005    0.000000
H            2.134213   -0.184329    0.000000
H            1.189706   -1.401951   -0.876286
H            1.189706   -1.401951    0.876286

*** CH3 top 1 ***
4 2, 4 5 6 7

*** CH3 top 2 ***
8 2, 8 9 10 11

"""

import numpy as np
import argparse

# atomic mass dictionary
MASS = {
    'H': 1,
    'D': 2,
    'C': 12,
    'N': 14,
    'O': 16,
    'F': 19,
    'P': 31,
    'S': 32,
}


def arg():
    """ Argutnem parser """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs=1, 
    help="""input XYZ, coordinates following internal rotor atom indicies 
    
    Sample format:
    N            0.045851    1.430604    0.000000
    C            0.000000    0.160884    0.000000
    H            1.009858    1.765235    0.000000
    C           -1.340657   -0.522812    0.000000
    H           -2.139230    0.213600    0.000000
    H           -1.442462   -1.166617    0.876838
    H           -1.442462   -1.166617   -0.876838
    
    *** CH3 top ***
    4 2, 4 5 6 7
    """)
    
    parser.add_argument('-rep', type=int, default=1,
                        help='Representation, 1 (prolate) or 3 (oblate)')
    return parser.parse_args()


def readfile(filename):
    """ Read xyz file
    :return
        xyz_list: list of (mass, x, y, z) coordinates for all atoms
        internal_rotor_list: list of atomic index tuple for each internal rotor
    """
    # read xyz coordinate
    xyz_list = []
    internal_rotor_list = []
    bulk_finished = False
    a_line = 'start'
    with open(filename, 'r') as f:
        while a_line != '':
            a_line = f.readline()
            if not bulk_finished:
                if a_line.startswith('*') or a_line.startswith('-'):
                    bulk_finished = True
                else:
                    a_list = a_line.split()
                    if len(a_list) == 4:  # cartesian
                        try:
                            mass = int(a_list[0])
                        except ValueError:
                            mass = MASS[a_list[0]]
                        q = [mass, ] + list(float(x) for x in a_list[1:])
                        xyz_list.append(np.array(q))
            else:
                if a_line.startswith('*') or a_line.startswith('-'):
                    pass
                elif a_line.strip() == '':
                    pass
                else:
                    a_list = a_line.split(', ')
                    axis = tuple(int(x) for x in a_list[0].split())
                    top = list(int(x) for x in a_list[1].split())
                    internal_rotor_list.append((axis, top))

    return xyz_list, internal_rotor_list


def calc_inertia_mat(xyz_list):
    mat = np.array(
            [
                [sum(q[0] * (q[2]**2 + q[3]**2) for q in xyz_list),
                 - sum(q[0] * q[1] * q[2] for q in xyz_list),
                 - sum(q[0] * q[1] * q[3] for q in xyz_list)],
                [- sum(q[0] * q[1] * q[2] for q in xyz_list),
                 sum(q[0] * (q[1]**2 + q[3]**2) for q in xyz_list),
                 - sum(q[0] * q[2] * q[3] for q in xyz_list)],
                [- sum(q[0] * q[1] * q[3] for q in xyz_list),
                 - sum(q[0] * q[2] * q[3] for q in xyz_list),
                 sum(q[0] * (q[1]**2 + q[2]**2) for q in xyz_list)],
            ])
    return mat


def calc_proj_plane(axis, v1, v2):
    """ Calculate projection of axis on the plane formed by vectors <v1, v2> """
    # norm of <v1, v2> plane
    n = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
    # projection of axis on n
    p1 = np.dot(axis, n) / np.linalg.norm(axis)
    # the projection on the plane is the vector difference
    return axis - p1


def dircos(v1, v2):
    """ Return direction cosine between two vectors """

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc(xyz_list, rotor_list, rep):
    """ calculate moments of inertia and rotational constants for the molecule
    :returns
        inertia_mat: moments of inertia matrix
        pa_dir: principle axes direction
        abc: rotational constants
    """

    mass_sum = sum(q[0] for q in xyz_list)
    com = (
        sum(q[0] * q[1] for q in xyz_list) / mass_sum,
        sum(q[0] * q[2] for q in xyz_list) / mass_sum,
        sum(q[0] * q[3] for q in xyz_list) / mass_sum,
    )
    inertia_mat = calc_inertia_mat(xyz_list)
    eig_val, eig_vec = np.linalg.eig(inertia_mat)
    abc = 6.62607004 * 6.02214086 / (8 * np.pi**2 * eig_val) * 1e3
    idx = np.argsort(abc)
    pa = eig_vec[:, idx]
    a_axis = pa[:, 0]
    b_axis = pa[:, 1]
    c_axis = pa[:, 2]

    i_aa = np.dot(a_axis, np.dot(inertia_mat, a_axis))
    i_ab = np.dot(a_axis, np.dot(inertia_mat, b_axis))
    i_ac = np.dot(a_axis, np.dot(inertia_mat, c_axis))
    i_bb = np.dot(b_axis, np.dot(inertia_mat, b_axis))
    i_bc = np.dot(b_axis, np.dot(inertia_mat, c_axis))
    i_cc = np.dot(c_axis, np.dot(inertia_mat, c_axis))
    inertia_mat_abc = np.array([[i_aa, i_ab, i_ac], [i_ab, i_bb, i_bc], [i_ac, i_bc, i_cc]])

    if rep == 1:
        rep_str = 'Ir'
        bx = abc[1]
        by = abc[2]
        bz = abc[0]
    elif rep == 3:
        rep_str = 'IIIr'
        bx = abc[0]
        by = abc[1]
        bz = abc[2]
    else:
        print('Unknown representation')
        exit(1)
    bj = (bx + by) / 2
    bk = bz - bj
    b_ = (bx - by) / 2

    print('*' * 20, ' bulk molecule ', '*'*20, end='\n\n')
    print(' Moments of inertia tensor (amu*AA^2) ', end='\n\n')
    print(inertia_mat, end='\n\n')
    print(' Principal axis direction ', end='\n\n')
    print(pa, end='\n\n')
    print(' Rotational constants (GHz) {:s}'.format(rep_str), end='\n\n')
    print(abc, end='\n\n')
    print(' BJ = {:.6f}'.format(bj))
    print(' BK = {:.6f}'.format(bk))
    print(' B- = {:.6f}'.format(b_))

    # now calculate internal rotor parameters
    for rotor in rotor_list:
        axis, top = rotor
        rotor_xyz = list(mol_xyz[i - 1] for i in top)
        rotor_axis = np.array(mol_xyz[axis[1]-1][1:]) - np.array(mol_xyz[axis[0]-1][1:])
        calc_rhof(rotor_xyz, rotor_axis, eig_val, pa, inertia_mat_abc, rep)


def calc_rhof(xyz, rotor_axis, mol_eig, mol_pa, mat_abc, rep):

    rotor_mat = calc_inertia_mat(xyz)
    rotor_eig, rotor_vec = np.linalg.eig(rotor_mat)
    # find the true top axis, which is the eigen vector closest to the specified rotor_axis
    # compute the product of rotor_axis with the eigen vector, the component closest to 1 is the true rotor axis
    _p = abs(np.dot(rotor_axis, rotor_vec)) / np.linalg.norm(rotor_axis)
    idx = np.argmin(np.abs(_p - 1))
    axis = rotor_vec[:, idx]

    if abs(rotor_eig[0] - rotor_eig[1]) > abs(rotor_eig[1] - rotor_eig[2]):
        i_alpha = rotor_eig[0]
    else:
        i_alpha = rotor_eig[-1]
    if rep == 1:
        lambda_x = dircos(axis, mol_pa[:, 1])
        lambda_y = dircos(axis, mol_pa[:, 2])
        lambda_z = dircos(axis, mol_pa[:, 0])
        r_ = 1 - lambda_y**2 * i_alpha / mol_eig[2] - lambda_z**2 * i_alpha / mol_eig[0]
        d = r_ * mol_eig[0] * mol_eig[2]
        delta = np.arccos(abs(lambda_y))
        epsilon = np.arccos(lambda_x * np.sign(lambda_z) / np.sqrt(lambda_x**2 + lambda_z**2))
    else:
        lambda_x = dircos(axis, mol_pa[:, 0])
        lambda_y = dircos(axis, mol_pa[:, 1])
        lambda_z = dircos(axis, mol_pa[:, 2])
        r_ = 1 - lambda_y**2 * i_alpha / mol_eig[1] - lambda_z**2 * i_alpha / mol_eig[2]
        d = r_ * mol_eig[1] * mol_eig[2]
        delta = np.arccos(abs(lambda_z))
        epsilon = np.arccos(lambda_x * np.sign(lambda_y) / np.sqrt(lambda_x**2 + lambda_y**2))
    f = 6.62607004 * 6.02214086 / (8 * np.pi**2 * r_ * i_alpha) * 1e3
    f0 = 505.379 / i_alpha
    rho = np.sqrt(mat_abc[1, 1]**2 + mat_abc[1, 2]**2) * i_alpha / (
            mat_abc[1, 1] * mat_abc[2, 2] - mat_abc[1, 2]**2)
    print()
    print('*' * 20, ' internal rotor ', '*' * 20, end='\n\n')
    print(' I_alpha (amu*AA^2): {:.6f}'.format(i_alpha))
    print(' rotor axis: ', axis)
    print(' rho       : {:.9f}'.format(rho))
    print(' F (GHz)   : {:.6f}'.format(f))
    print(' F0 (GHz)  : {:.6f}'.format(f0))
    print(' epsilon (rad / deg): {:.6f} {:7.3f}'.format(epsilon, epsilon * 180 / np.pi))
    print(' delta   (rad / deg): {:.6f} {:7.3f}'.format(delta, delta * 180 / np.pi))
    print(' delta (deg) <internal rotor axis, principal axis> ')
    print('   x: {:.3f}    y: {:.3f}   z: {:.3f}'.format(
            np.arccos(abs(lambda_x)) * 180 / np.pi,
            np.arccos(abs(lambda_y)) * 180 / np.pi,
            np.arccos(abs(lambda_z)) * 180 / np.pi
    ))


if __name__ == '__main__':

    args = arg()
    mol_xyz, internal_rotor = readfile(args.file[0])
    calc(mol_xyz, internal_rotor, args.rep)
