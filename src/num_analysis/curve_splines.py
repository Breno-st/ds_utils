"""Parametric Curve Splines
* :function:`._chord_params`   -- chord-length parametrisation for N-D point sequences
* :function:`._knot_vector`    -- clamped B-spline knot vector (Piegl & Tiller averaging)
* :function:`._basis`          -- recursive Cox-de Boor basis function N_{i,k}(u)
* :function:`._collocation`    -- B-spline collocation matrix
* :function:`.interpolate_curve` -- interpolating cubic spline, exact pass-through
* :function:`.smooth_curve`    -- LSPIA / direct least-squares B-spline approximation

Motivation
----------
The scalar interpolation and approximation functions in ``polynomial_interpolant.py``
and ``polynomial_approximation.py`` all operate on 1-D functions y = f(x).  A geographic
or geometric curve (e.g. a transit line, a road centreline) cannot be represented this
way: it folds back on itself and has no single-valued y for a given x.

The correct setting is a *parametric* curve: assign each data point a scalar parameter
t and fit N independent smooth functions -- one per coordinate dimension -- sharing the
same parametrisation.  This module provides that machinery, generalised to arbitrary
dimension d (not just 2-D lon/lat).

Key improvements over ``polynomial_approximation.py``
-------------------------------------------------------
* ``_knot_vector`` no longer references a global variable ``k`` -- degree is an
  explicit parameter.
* LSPIA is fully vectorised: one matrix update step per iteration across all
  dimensions simultaneously, replacing the per-dimension Python loop.
* Direct least-squares (``use_lspia=False``, the default) completes in one pass and is
  mathematically equivalent for smooth geographic data.
* Both functions accept (n, d) NumPy arrays and return results in the same shape,
  making them composable with other NumPy pipelines.
* ``smooth_curve`` optionally returns per-iteration RMSE for convergence monitoring.
"""

import numpy as np
from scipy.interpolate import CubicSpline as _CubicSpline


# ── Private helpers ──────────────────────────────────────────────────────────

def _chord_params(pts):
    """
    Summary:    Chord-length parametrisation of an ordered N-D point sequence.
    --------
    Input:      - pts: (n, d) numpy array of ordered points
    ------
    Output:     - t: (n,) array of parameter values in [0, 1], where each t_i is
    -------       proportional to the cumulative Euclidean distance from pts[0].

    Notes:      Chord-length parametrisation distributes parameter intervals in
    -----       proportion to inter-point distances, preventing the oscillation
                that arises with uniform parametrisation when points are unevenly
                spaced.  It is the standard choice for geometric curve fitting
                (Piegl & Tiller, 1997, Chapter 9).
    """
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumlen[-1]
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(pts))
    return cumlen / total


def _knot_vector(degree, n_data, n_cpts, params):
    """
    Summary:    Clamped B-spline knot vector using the Piegl & Tiller averaging method.
    --------
    Input:      - degree: B-spline degree k (int)
    ------      - n_data: number of data points n
                - n_cpts: number of control points (must satisfy n_cpts >= degree + 1)
                - params: chord-length parameter array (n_data,)
    Output:     - knots: list of length n_cpts + degree + 1.  First degree+1 values
    -------       are 0.0 (clamped left); last degree+1 values are 1.0 (clamped right).

    Notes:      The clamped form guarantees that the B-spline interpolates the first
    -----       and last control points exactly, which is essential for endpoint pinning.
                This function fixes the global-variable bug in _knot_vector from
                polynomial_approximation.py, where the loop bounds used the module-level
                ``k`` instead of the ``degree`` parameter.
    """
    n_interior = n_cpts - degree - 1
    knots = [0.0] * (degree + 1)
    for j in range(1, n_interior + 1):
        idx = int(round(j * n_data / (n_interior + 1)))
        idx = min(max(idx, 1), n_data - 2)
        knots.append(float(params[idx]))
    knots.extend([1.0] * (degree + 1))
    return knots


def _basis(i, k, u, knots):
    """
    Summary:    Recursive Cox-de Boor B-spline basis function N_{i,k}(u).
    --------
    Input:      - i: basis function index (int)
    ------      - k: degree (int)
                - u: evaluation parameter (float)
                - knots: knot vector (list or array)
    Output:     - scalar float basis value N_{i,k}(u)
    -------

    Notes:      Right-endpoint clamping: N_{i,0}(1.0) = 1 when knots[i] < 1 <= knots[i+1],
    -----       ensuring the last basis function evaluates to 1 at u = 1.
    """
    if k == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        if u == 1.0 and knots[i] < 1.0 <= knots[i + 1]:
            return 1.0
        return 0.0
    left = right = 0.0
    d1 = knots[i + k] - knots[i]
    d2 = knots[i + k + 1] - knots[i + 1]
    if d1 > 1e-10:
        left = (u - knots[i]) / d1 * _basis(i, k - 1, u, knots)
    if d2 > 1e-10:
        right = (knots[i + k + 1] - u) / d2 * _basis(i + 1, k - 1, u, knots)
    return left + right


def _collocation(params, n_cpts, degree, knots):
    """
    Summary:    Build the (n_data, n_cpts) B-spline collocation matrix A, where
    --------    A[r, c] = N_{c, degree}(params[r]).
    Input:      - params: parametrisation array (n_data,)
    ------      - n_cpts: number of control points
                - degree: B-spline degree
                - knots: knot vector
    Output:     - A: (n_data, n_cpts) numpy array
    -------
    """
    n = len(params)
    A = np.zeros((n, n_cpts))
    for r, u in enumerate(params):
        for c in range(n_cpts):
            A[r, c] = _basis(c, degree, float(u), knots)
    return A


# ── Public API ───────────────────────────────────────────────────────────────

def interpolate_curve(coords, n_eval=20, bc_type='not-a-knot'):
    """
    Summary:    Cubic spline that passes exactly through every input point
    --------    (interpolating spline), using chord-length parametrisation.
                Each coordinate dimension is fitted independently with the
                same parameter array, producing a smooth parametric curve.

    Input:      - coords: list of [x, y, ...] sequences or (n, d) array.
    ------                Ordering matters -- the spline follows the sequence.
                - n_eval: number of evaluation points per inter-node segment
                          (default 20).  Total output points ≈ (n-1) * n_eval.
                - bc_type: boundary condition for scipy CubicSpline.
                           'not-a-knot' (default) -- minimises oscillation at
                           endpoints without constraining derivatives.
                           'natural' -- zero second derivative at endpoints.
                           'periodic' -- use for closed curves (loop lines).
                           'clamped' -- zero first derivative at endpoints.

    Output:     - list of [x, y, ...] of length ≈ (n-1)*n_eval that passes
    -------       exactly through every input point.

    Notes:      This function was developed for the TfL network map pipeline,
    -----       where it draws tube/Overground/DLR lines as smooth curves
                through station anchor positions.  At zoom 12 on Mapbox,
                20 evaluation points per segment produces visually smooth
                curves without perceptible polygon faceting.

                For 2-D geographic data (lon, lat) the chord-length parametrisation
                avoids the Runge oscillation that equi-spaced parametrisation
                causes on non-uniform station distributions.

    Example:    coords = [[-0.45, 51.50], [-0.30, 51.52], [-0.12, 51.51]]
    --------    curve  = interpolate_curve(coords, n_eval=10)
    """
    pts = np.array(coords, dtype=float)
    if pts.ndim == 1:
        pts = pts[:, np.newaxis]
    n, d = pts.shape
    if n < 2:
        return [list(row) for row in pts]
    if n == 2:
        t_dense = np.linspace(0.0, 1.0, n_eval)
        return [list(pts[0] + s * (pts[1] - pts[0])) for s in t_dense]

    t = _chord_params(pts)

    splines = [_CubicSpline(t, pts[:, dim], bc_type=bc_type)
               for dim in range(d)]

    t_dense = np.concatenate([
        np.linspace(t[i], t[i + 1], n_eval, endpoint=(i == n - 2))
        for i in range(n - 1)
    ])

    result = np.column_stack([cs(t_dense) for cs in splines])
    return [list(row) for row in result]


def smooth_curve(coords, percentage=0.5, degree=3,
                 max_iter=500, tol=1e-5, use_lspia=False,
                 return_errors=False):
    """
    Summary:    B-spline approximation of a parametric N-D curve using either
    --------    LSPIA (iterative) or direct least-squares.  Generalises the
                scalar LSPIA in polynomial_approximation.py to N-dimensional
                parametric curves with a vectorised implementation.

    Input:      - coords: list of [x, y, ...] sequences or (n, d) array.
    ------      - percentage: fraction of n used as number of B-spline control
                  points.  0.25 = very smooth / schematic; 0.75 = near-interpolation.
                  Default 0.5.
                - degree: B-spline degree (default 3 -- cubic).
                - max_iter: maximum LSPIA iterations (default 500).
                  Ignored when use_lspia=False.
                - tol: LSPIA convergence tolerance on per-iteration RMSE change
                  (default 1e-5).
                - use_lspia: if True use the LSPIA iterative algorithm (Deng & Lin,
                  2014); if False use direct NumPy lstsq (default).  Both solve
                  the same least-squares problem; LSPIA is slower but gives
                  convergence diagnostics via return_errors.
                - return_errors: if True return a second value -- a list of
                  per-iteration RMSE values.  Useful for diagnosing convergence.
                  When use_lspia=False the list contains a single value (the
                  final residual).

    Output:     - smoothed list of [x, y, ...], same length as coords.  First and
    -------       last points are clamped to the exact input endpoints.
                - (if return_errors=True) list of float RMSE values.

    Notes:      LSPIA convergence is guaranteed when the step size satisfies
    -----       0 < mu < 2 / spectral_radius(A^T A).  The bound mu = 2 / max_col_sum(A)
                used here is a conservative approximation that always satisfies
                this condition because max_col_sum(A) >= spectral_radius(A^T A)^0.5
                for non-negative matrices (Perron-Frobenius).

                The vectorised update  P += mu * A^T @ (pts - A @ P)  replaces
                the per-dimension Python loop in polynomial_approximation.py,
                reducing wall time by factor d for d-dimensional data.

    Example:    coords = [[-0.45,51.50],[-0.30,51.52],[-0.12,51.51],[0.02,51.50]]
    --------    smoothed, errors = smooth_curve(coords, percentage=0.4,
                                               use_lspia=True, return_errors=True)
    """
    pts = np.array(coords, dtype=float)
    if pts.ndim == 1:
        pts = pts[:, np.newaxis]
    n, d = pts.shape

    if n < degree + 2:
        result = [list(row) for row in pts]
        return (result, []) if return_errors else result

    n_cpts = max(degree + 1, int(round(n * percentage)))
    n_cpts = min(n_cpts, n)

    params = _chord_params(pts)
    knots  = _knot_vector(degree, n, n_cpts, params)
    A      = _collocation(params, n_cpts, degree, knots)   # (n, n_cpts)

    errors = []

    if use_lspia:
        col_sums = A.sum(axis=0)
        mu = 2.0 / float(np.max(col_sums[col_sums > 0]))

        P = np.zeros((n_cpts, d))
        P[0]  = pts[0]
        P[-1] = pts[-1]

        prev_rmse = float('inf')
        for _ in range(max_iter):
            residual = pts - A @ P                  # (n, d)
            delta    = mu * (A.T @ residual)        # (n_cpts, d)
            delta[0]  = 0.0
            delta[-1] = 0.0
            P += delta
            rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
            errors.append(rmse)
            if abs(prev_rmse - rmse) < tol:
                break
            prev_rmse = rmse

        fitted = A @ P

    else:
        P = np.column_stack([
            np.linalg.lstsq(A, pts[:, dim], rcond=None)[0]
            for dim in range(d)
        ])
        fitted = A @ P
        rmse = float(np.sqrt(np.mean(np.sum((pts - fitted) ** 2, axis=1))))
        errors = [rmse]

    fitted[0]  = pts[0]
    fitted[-1] = pts[-1]

    result = [list(row) for row in fitted]
    return (result, errors) if return_errors else result
