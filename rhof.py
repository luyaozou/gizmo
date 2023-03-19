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


ABC2XYZ_Ir = np.array([1, 2, 0], dtype='int16')      # (x, y, z) -> (b, c, a)    # z = a axis, apply the index on ABC vector to get XYZ vector
ABC2XYZ_IIIr = np.array([0, 1, 2], dtype='int16')    # (x, y, z) -> (a, b, c)    # z = c axis, apply the index on ABC vector to get XYZ vector


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


def calc_inertia_tensor(xyz_list):
    """ The inertia tensor needs to be calculated from the center of mass """
    com = sum(np.array([q[0] * q[1], q[0] * q[2], q[0] * q[3]]) for q in xyz_list) \
            / sum(q[0] for q in xyz_list)
    qcom_list = list(
        (q[0], q[1] - com[0], q[2] - com[1], q[3] - com[2]) for q in xyz_list
    )
    mat = np.array(
            [
                [sum(q[0] * (q[2]**2 + q[3]**2) for q in qcom_list),
                 - sum(q[0] * q[1] * q[2] for q in qcom_list),
                 - sum(q[0] * q[1] * q[3] for q in qcom_list)],
                [- sum(q[0] * q[1] * q[2] for q in qcom_list),
                 sum(q[0] * (q[1]**2 + q[3]**2) for q in qcom_list),
                 - sum(q[0] * q[2] * q[3] for q in qcom_list)],
                [- sum(q[0] * q[1] * q[3] for q in qcom_list),
                 - sum(q[0] * q[2] * q[3] for q in qcom_list),
                 sum(q[0] * (q[1]**2 + q[2]**2) for q in qcom_list)],
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
    inertia_tensor = calc_inertia_tensor(xyz_list)
    eig_val, eig_vec = np.linalg.eig(inertia_tensor)
    abc_ghz = 6.62607004 * 6.02214086 / (8 * np.pi**2 * eig_val) * 1e3
    idx = np.argsort(eig_val)
    i_abc = eig_val[idx]
    abc_ghz = abc_ghz[idx]
    abc_cm = abc_ghz / 29.9792458
    # the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    pa = eig_vec[:, idx]
    # now we need to tweak the sign of the principle axes.
    # the eigen vector solver will return randomly + or - sign,
    # but by convention we let the largest component in each eigen vector to be positive
    for i in range(3):
        max_elem = pa[np.argmax(abs(pa[:, i])), i]
        if max_elem > 0:
            pass
        else:
            pa[:, i] = -pa[:, i]

    if rep == 1:
        rep_str = 'Ir'
        b = abc_ghz[ABC2XYZ_Ir]
    elif rep == 3:
        rep_str = 'IIIr'
        b = abc_ghz[ABC2XYZ_IIIr]
    else:
        print('Unknown representation')
        exit(1)
    bj = (b[0] + b[1]) / 2
    bk = b[2] - bj
    b_ = (b[0] - b[1]) / 2

    print('*' * 20, ' whole molecule ', '*'*20, end='\n\n')
    print(' Moments of inertia tensor (amu*AA^2) ', end='\n\n')
    print(inertia_tensor, end='\n\n')
    print(' Principal axis direction ', end='\n\n')
    print(pa, end='\n\n')
    print(' Rotational constants (GHz / cm-1) {:s}'.format(rep_str), end='\n\n')
    print(' A  = {:>10.6f} {:>12.9f}'.format(abc_ghz[0], abc_cm[0]))
    print(' B  = {:>10.6f} {:>12.9f}'.format(abc_ghz[1], abc_cm[1]))
    print(' C  = {:>10.6f} {:>12.9f}'.format(abc_ghz[2], abc_cm[2]))
    print()
    print(' BJ = {:>10.6f} {:>12.9f}'.format(bj, bj / 29.9792458))
    print(' BK = {:>10.6f} {:>12.9f}'.format(bk, bk / 29.9792458))
    print(' B- = {:>10.6f} {:>12.9f}'.format(b_, b_ / 29.9792458))

    # now calculate internal rotor parameters
    for rotor in rotor_list:
        axis, top = rotor
        rotor_xyz = list(mol_xyz[i - 1] for i in top)
        rotor_axis = np.array(mol_xyz[axis[1]-1][1:]) - np.array(mol_xyz[axis[0]-1][1:])
        calc_rhof(rotor_xyz, rotor_axis, pa, i_abc, abc_ghz, rep)


def calc_rhof(xyz, rotor_axis, mol_pa, i_abc, abc_ghz, rep):
    """ Calculate internal rotor parameters
    :arguments
        xyz:            cartesian of internal rotor atoms
        rotor_axis:     axis that the user points out as the rotor axis (use to find true rotor axis)
        mol_pa:         principal axis (ABC) of the whole molecule
        i_abc:          moment of inertia of the whole molecule, ABC axis
        abc_ghz:        rotational constants (GHz) of the whole molecule, ABC axis
    """
    rotor_inertia_tensor = calc_inertia_tensor(xyz)
    rotor_eig, rotor_vec = np.linalg.eig(rotor_inertia_tensor)
    # find the true top axis, which is the eigen vector closest to the specified rotor_axis
    # compute the product of rotor_axis with the eigen vector, the component closest to 1 is the true rotor axis
    _p = np.dot(rotor_axis, rotor_vec) / np.linalg.norm(rotor_axis)
    idx = np.argmin(1 - abs(_p))
    # match the direction of the true_axis with the rotor_axis that the input specifies
    true_axis = rotor_vec[:, idx] if _p[idx] > 0 else -rotor_vec[:, idx]
    i_alpha = rotor_eig[idx]
    # don't know why I have this weird condition here. Does not make sense
    # if abs(rotor_eig[0] - rotor_eig[1]) > abs(rotor_eig[1] - rotor_eig[2]):
    #     i_alpha = rotor_eig[0]
    # else:
    #     i_alpha = rotor_eig[-1]
    if rep == 1:
        i_xyz = i_abc[ABC2XYZ_Ir]
        lambda_x = dircos(true_axis, mol_pa[:, 1])
        lambda_y = dircos(true_axis, mol_pa[:, 2])
        lambda_z = dircos(true_axis, mol_pa[:, 0])
    else:
        i_xyz = i_abc[ABC2XYZ_IIIr]
        lambda_x = dircos(true_axis, mol_pa[:, 0])
        lambda_y = dircos(true_axis, mol_pa[:, 1])
        lambda_z = dircos(true_axis, mol_pa[:, 2])
    lambda_xyz = np.array([lambda_x, lambda_y, lambda_z])
    r_ = 1 - sum(lambda_xyz[i]**2 * i_alpha / i_xyz[i] for i in range(3))
    d_ = r_ * i_xyz[1] * i_xyz[2]
    f = 6.62607004 * 6.02214086 / (8 * np.pi**2 * r_ * i_alpha) * 1e3
    f0 = 505.379 / i_alpha
    delta = np.arccos(lambda_z)
    epsilon = np.arcsin(lambda_y / np.sqrt(lambda_x ** 2 + lambda_y ** 2))
    # rho = sqrt(Ibb^2 + Ibc^2) * Ia / (Ibb * Icc - Ibc^2)  (from Lin & Swalen 1959)
    # rho = np.sqrt(i_tensor_abs[1, 1] ** 2 + i_tensor_abs[1, 2] ** 2) * i_alpha / (
    #         i_tensor_abs[1, 1] * i_tensor_abs[2, 2] - i_tensor_abs[1, 2] ** 2)
    rho_xyz = lambda_xyz / i_xyz * i_alpha
    theta = np.arctan(rho_xyz[0] / rho_xyz[2])
    gamma = np.arccos(rho_xyz[0] / np.sqrt(rho_xyz[0]**2 + rho_xyz[1]**2))
    beta = np.arccos(rho_xyz[2] / np.linalg.norm(rho_xyz))
    dxy = f * rho_xyz[0] * rho_xyz[1]  # DAB, DBC, DAC
    dxz = f * rho_xyz[0] * rho_xyz[2]  # DAB, DBC, DAC
    dyz = f * rho_xyz[1] * rho_xyz[1]  # DAB, DBC, DAC
    q_vec = - 2 * f * rho_xyz
    if rep == 1:
        dab = dxz
        dbc = dxy
        dac = dyz
    else:
        dab = dxy
        dbc = dyz
        dac = dxz

    print()
    print('*' * 20, ' internal rotor ', '*' * 20, end='\n\n')
    print(' I_alpha (amu*AA^2): {:.6f}'.format(i_alpha))
    print(' rotor axis: ', true_axis)
    print(' rho vector: ', rho_xyz)
    print(' |rho|     :  {:.9f}'.format(np.linalg.norm(rho_xyz)))
    print(' DAB (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(dab, dab / 29.9792458))
    print(' DBC (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(dbc, dbc / 29.9792458))
    print(' DAC (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(dac, dac / 29.9792458))
    print(' F   (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(f, f / 29.9792458))
    print(' F0  (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(f0, f0 / 29.9792458))
    print(' Qx  (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(q_vec[0], q_vec[0] / 29.9792458))
    print(' Qy  (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(q_vec[1], q_vec[1] / 29.9792458))
    print(' Qz  (GHz / cm-1) : {:>10.6f} {:>12.9f}'.format(q_vec[2], q_vec[2] / 29.9792458))
    print(' theta   (rad / deg) : {:>9.6f} {:>7.3f}'.format(theta, theta * 180 / np.pi))
    print(' beta    (rad / deg) : {:>9.6f} {:>7.3f}'.format(beta, beta * 180 / np.pi))
    print(' gamma   (rad / deg) : {:>9.6f} {:>7.3f}'.format(gamma, gamma * 180 / np.pi))
    print(' epsilon (rad / deg) : {:>9.6f} {:>7.3f}'.format(epsilon, epsilon * 180 / np.pi))
    print(' delta   (rad / deg) : {:>9.6f} {:>7.3f}'.format(delta, delta * 180 / np.pi))
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
