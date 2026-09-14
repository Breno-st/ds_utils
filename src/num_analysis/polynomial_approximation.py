"""Approxmation
* :class:`.Splines`
* :function:`.cubic`
* :function:`.PIA`
* :function:`.LSPIA`
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def _progessive_points(Q):
    """
    Contruct progressive percentual values based on the distances between the original points setorder.
    """
    n = len(Q[0])
    t = np.zeros((1,n))
    for i in range (1,n):
        dis = 0
        for j in range(len(Q)):
            dis = dis + (Q[j][i] - Q[j][i - 1]) ** 2  # (deltaX)^2 + (deltaY)^2
        dis = np.sqrt(dis)                            # euclidian distance
        t[0][i] = t[0][i - 1] + dis                   # adding up fo accumulative
    # normalizing
    for i in range(1, n):
        t[0][i] = t[0][i] / t[0][n - 1]
    return t[0]

def _knot_vector(degree, num_dpts, num_cpts, params):
    """
    Computes a knot vector ensuring that every knot span has at least one :math:`\\overline{u}_{k}`.

    Note: the original implementation incorrectly referenced a module-level variable ``k``
    instead of the ``degree`` parameter.  Fixed here to use ``degree`` throughout.
    See ``curve_splines._knot_vector`` for the improved Piegl-Tiller averaging version.
    """
    m = num_cpts + degree
    knot = np.zeros((1, m + 1))
    for i in range(m - degree, m + 1):
        knot[0][i] = 1
    for i in range(degree + 1, m - degree):
        j = i - degree
        jd = j * num_dpts / (num_cpts - degree)
        n = int(jd)
        alpha = jd - n
        knot[0][i] = (1-alpha)*params[n-1]+alpha*params[n]
    return knot[0]

def basis_function(i, k, u, knot):
    """
    Computes the non-vanishing basis functions for a single parameter
    """
    if k > 1:
        init = (u - knot[i]) if knot[i + k - 1] == knot[i] else (u - knot[i]) / (knot[i + k - 1] - knot[i])
        end = (knot[i + k] - u)  if knot[i + k] == knot[i + 1] else (knot[i + k] - u)  / (knot[i + k] - knot[i + 1])
        Nik_u = init* basis_function(i, k - 1, u, knot) + end * basis_function(i + 1, k - 1, u, knot)
        return Nik_u
    else:
        return 1 if u >= knot[i] and u <= knot[i + 1] else 0


def curve_fitting_error(D, P, Nik):
    '''
    Calculate the curve fitting error.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: fitting error
    '''
    error = 0
    Nik = np.array(Nik)
    for dim in range(len(D)):
        D_dim = np.array(D[dim])
        P_dim = np.array(P[dim])
        error = error + np.sum(np.square(D_dim - np.dot(P_dim, np.transpose(Nik))))
    return error

def curve_adjusting_control_points(D, P, Nik, mu):
    '''
    Adjusting the curve control points with the adjusting vector.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: new control points
    '''
    Nik = np.array(Nik)
    for dim in range(len(D)):
        D_dim = np.array(D[dim])
        P_dim = np.array(P[dim])
        delta = mu * np.dot(D_dim - np.dot(P_dim, np.transpose(Nik)), Nik)
        P[dim] = (P_dim + delta).tolist()
    return P

def curve(param, P, Nik):
    '''
    Calculate the data points on the b-spline curve.
    :param param: the piece of param
    :param P: the control points
    :param Nik: the basis spline function
    :return: data points
    '''
    Nik = np.array(Nik)
    D = []
    for dim in range(len(P)):
        P_dim = np.array(P[dim])
        D_dim = np.dot(P_dim, np.transpose(Nik))
        D.append(D_dim.tolist())
    return D


def plot_splines(Ps, Pi_num, knot_vector, tit):
    piece = 200
    p_piece = np.linspace(0, 1, piece)
    Nik_piece = np.zeros((piece, Pi_num))
    for i in range(piece):
        for j in range(Pi_num):
            Nik_piece[i][j] = basis_function(j, k + 1, p_piece[i], knot_vector)
    P_piece = curve(p_piece, Ps, Nik_piece)

    # Draw b-spline curve
    for i in range(Q_num):
        plt.scatter(Q[0][i], Q[1][i], color='r')

    for i in range(piece - 1):
        tmp_x = [P_piece[0][i], P_piece[0][i + 1]]
        tmp_y = [P_piece[1][i], P_piece[1][i + 1]]
        plt.plot(tmp_x, tmp_y, color='g')

    plt.title(tit)
    plt.savefig('{}.png'.format(tit))

def load_curve_data(filename):
    x = []
    y = []
    with open(os.path.join(sys.path[0], filename), "r") as file:
        for line in file.readlines():
            line = line.strip()
            word = line.split(' ')
            x.append(float(word[0]))
            y.append(float(word[1]))
    return [x, y]







if __name__ == '__main__':
    # Getting data points
    Q = load_curve_data('cur_data'.format())
    Q_num = len(Q[0])

    # Degree
    # Percentage of data points as control points
    percentage, k = 0.25, 3

    Pi_num = int(Q_num * percentage)

    # Assembling control points
    Pi_x, Pi_y = [], []
    for i in range(0, Pi_num):
        Pi_x.append(0)
        Pi_y.append(0)
    # Equal begin and end of Q
    Pi_x[0] = Pi_x[-1] = Q[0][0]
    Pi_y[0] = Pi_y[-1] = Q[1][0]
    Pi = [Pi_x, Pi_y]

    # Calculate the parameters
    t_param = _progessive_points(Q)

    # Calculate knot vector
    knot_vector = _knot_vector(k, Q_num, Pi_num, t_param)

    # Calculate the A matrix of the NTP blending basis
    A_mtx = np.zeros((Q_num, Pi_num))
    c = np.zeros((1, Pi_num))
    for i in range(Q_num):
        for j in range(Pi_num):
            A_mtx[i][j] = basis_function(j, k + 1, t_param[i], knot_vector)
            c[0][j] = c[0][j] + A_mtx[i][j]
    C = max(c[0].tolist())
    miu = 2 / C

    # Fitting iteration
    errors = []
    errors.append(curve_fitting_error(Q, Pi, A_mtx))

    # Adjusting control points
    P = curve_adjusting_control_points(Q, Pi, A_mtx, miu)
    tit = 'Initial Iterations: error = {}'.format(errors[0])
    plot_splines(P, Pi_num, knot_vector, tit)
    errors.append(curve_fitting_error(Q, Pi, A_mtx))

    i = 0
    while abs(errors[-1] - errors[-2]) >= 1e-5:
        i += 1
        P = curve_adjusting_control_points(Q, Pi, A_mtx, miu)
        errors.append(curve_fitting_error(Q, Pi, A_mtx))
        if i % 5 == 0:
            tit = 'Iterations #{}: error = {}'.format(i, errors[-1])
            plot_splines(P, Pi_num, knot_vector, tit)

    tit = 'Final iterations: error = {}'.format(errors[-1])
    plot_splines(P, Pi_num, knot_vector, tit)
