import numpy as np
import warnings

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import calculate_norm, lu_decomposition, lu_solve
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import calculate_norm, lu_decomposition, lu_solve


def fourier_series_analytical_sol(rho, theta, n_terms=100):
    """
    Analytical solution of the problem at point (rho,theta) computed using n_terms not null from
    Fourier series expansion

    Args:
        rho (float): radial non-dimensional coordinate
        theta (float): azimuthal coordinate
        n_terms (int): number of non-0 terms to use in the expansion (default=100)

    Returns:
         u(float): non-dimensional analytical solution
    """
    u = 0
    for ii in range(n_terms):
        n = 2*ii+1
        u += 4/np.pi/n * rho**n * np.sin(n*theta)

    return u


def polar_laplace_eq_df_system(n, m, r_range, th_range, bcs):
    """
    Build the linear system representing the Laplace equation with Dirichlet BCs

    Args:
        n (int): number of intervals in radial coordinate
        m (int): number of intervals in azimuthal coordinate
        r_range (tuple): min and max values of r
        th_range (tuple): min and max values of theta
        bcs (list): list of boundary conditions at [r_min, r_max, theta_min, theta_max]

    Returns:
        matrix(np.ndarray): matrix of coefficients of the resulting linear system
        vector(np.ndarray): vector of the independent terms in the linear system
    """

    # Extract coordinates relevant info
    rho_vals = np.linspace(r_range[0], r_range[-1], n+1)
    d_rho = np.abs(rho_vals[1] - rho_vals[0])
    d_theta = np.abs(th_range[-1] - th_range[0])/m

    # Initialize matrix & vector
    matrix = np.zeros(((n-1)*(m-1), (n-1)*(m-1)))
    vector = np.zeros((n-1)*(m-1))

    # Build matrix & vector
    a = 1/(d_rho**2)
    for ii in range(1, n):
        b = 1/(2 * d_rho * rho_vals[ii])
        c = 1/(d_theta**2 * rho_vals[ii]**2)
        for jj in range(1, m):
            index = (m-1)*(ii-1)+jj-1
            # Diagonal elem
            matrix[index, index] = -2*(a+c)

            # Check boundary conditions:
            # - r_min
            if ii == 1:
                vector[index] -= (a-b)*bcs[0]
            else:
                matrix[index, index-(m-1)] = a-b
            # - r_max
            if ii == n-1:
                vector[index] -= (a+b)*bcs[1]
            else:
                matrix[index, index+(m-1)] = a+b
            # - theta_min
            if jj == 1:
                vector[index] -= c*bcs[2]
            else:
                matrix[index, index-1] = c
            # - theta_max
            if jj == m-1:
                vector[index] -= c*bcs[3]
            else:
                matrix[index, index+1] = c

    return matrix, vector
