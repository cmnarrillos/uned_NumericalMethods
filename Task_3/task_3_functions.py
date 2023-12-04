import numpy as np
from datetime import datetime
import warnings

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import calculate_norm, lu_decomposition, lu_solve, inverse_lower_triangular
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import calculate_norm, lu_decomposition, lu_solve, inverse_lower_triangular


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


def jacobi_matrix(A):
    """
    Function returning the iterative matrix applied when using Jacobi iterative formula
    Args:
        A (np.ndarray): matrix

    Returns:
        H (np.ndarray): matrix of the formula x(n+1) = H*x(n) + B*b
    """
    # Step 1: Compute matrices D, L, and U
    D_inv = np.diag(1/np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    H = np.matmul(D_inv, (L + U))

    return H


def gs_matrix(A):
    """
    Function returning the iterative matrix applied when using Jacobi iterative formula
    Args:
        A (np.ndarray): matrix

    Returns:
        H (np.ndarray): matrix of the formula x(n+1) = H*x(n) + B*b
    """
    # Step 1: Compute matrices D, L, and U
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    H = np.matmul(inverse_lower_triangular(D - L), U)

    return H


def sor_matrix(A, w):
    """
    Function returning the iterative matrix applied when using SOR iterative formula
    Args:
        A (np.ndarray): matrix
        w (float): relaxation param

    Returns:
        H (np.ndarray): matrix of the formula x(n+1) = H*x(n) + B*b
    """
    n = A.shape[0]

    # Step 1: Compute matrices D, L, and U
    D_inv = np.diag(1/np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    lhs = np.eye(n) - w * np.matmul(D_inv, L)
    lhs_inv = inverse_lower_triangular(lhs)

    rhs = (1 - w) * np.eye(n) + w * np.matmul(D_inv, U)

    H = np.matmul(lhs_inv, rhs)

    return H


def get_error_diff_grids(solution, analytical_sol, aim_shape):
    """
    Compare solution vs analytical solution when both are computed at points corresponding to aim_shape
    Args:
        solution (np.ndarray): Matrix with values at nodes
        analytical_sol (np.ndarray): Reference solution used to compute error
        aim_shape (tuple): Shape of the grid where errors are computed

    Returns:
        error (np.ndarray): Matrix with errors at points of the aim_shape grid
    """

    if (not (solution.shape[0]-1) % (aim_shape[0]-1)) & (not (analytical_sol.shape[0]-1) % (aim_shape[0]-1)) & \
            (not (solution.shape[1]-1) % (aim_shape[1]-1)) & (not (analytical_sol.shape[1]-1) % (aim_shape[1]-1)):
        step_sol_0 = (solution.shape[0] - 1) // (aim_shape[0] - 1)
        step_sol_1 = (solution.shape[1] - 1) // (aim_shape[1] - 1)
        step_analytical_sol_0 = (analytical_sol.shape[0] - 1) // (aim_shape[0] - 1)
        step_analytical_sol_1 = (analytical_sol.shape[1] - 1) // (aim_shape[1] - 1)
        error = solution[::step_sol_0, ::step_sol_1] - analytical_sol[::step_analytical_sol_0, ::step_analytical_sol_1]

        return error
    else:
        raise ValueError("Input matrices cannot match aim shape")


def document_test_polar(filename, solution, info='', latex_shape=None, analytical_sol=None, n_terms=None):
    """
    Document result obtained

    Args:
        filename (str): Name of the file to be created
        solution (np.ndarray): Matrix with values at nodes
        latex_shape (tuple): Size of the table to be put in LaTeX
        info (str): Information to be included in the doc file after preamble
        analytical_sol (np.ndarray): Reference solution used to compute error
        n_terms (int): Number of terms of the Fourier series used for computing the analytical solution

    Returns:
        None

    Creates file with formatted documentation
    """
    n = solution.shape[0] - 1
    m = solution.shape[1] - 1

    with open(filename, 'w') as f:
        # Write preamble + additional info
        f.write(f'Test run on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(info + '\n\n\n')

        # Write the solution array in the file
        for row in solution:
            formatted_row = ' ' + ', '.join([f'{value:12.10f}' for value in row])
            f.write(formatted_row + '\n')

        # If latex_shape is provided and those points are matched, writhe the tabular expression for LaTeX
        if latex_shape is not None:
            if (not (n - 1) % (latex_shape[0])) & (not (m - 1) % (latex_shape[1])):
                f.write('\n\n\n')
                f.write('Table for LaTeX:\n')
                f.write('\\begin{tabular}{|' + '|'.join(['c'] * (latex_shape[1]-1)) + '|}\n')
                f.write('\hline\n')
                for i in range((n - 1) // (latex_shape[0]), solution.shape[0] - 1, (n - 1) // (latex_shape[0])):
                    row = solution[i]
                    elems = row[(m - 1) // (latex_shape[1]):-1:(m - 1) // (latex_shape[1])]
                    formatted_row = ' & '.join([f'{value:12.10f}' for value in elems])
                    f.write(formatted_row + '\\\\\hline\n')
                f.write('\hline\n')
                f.write('\\end{tabular}\n')
            f.write('\n\n\n')

        # If analytical solution is provided, try to compute error and document it
        if analytical_sol is not None:
            if solution.shape == analytical_sol.shape:
                f.write(f'Error wrt analytical solution obtained with {n_terms} terms (error ~O(1/N)~'
                        f'{1 / (n_terms - 1)}\n')
                for row in solution - analytical_sol:
                    formatted_row = ' ' + ', '.join([f'{value:+12.10f}' for value in row])
                    f.write(formatted_row + '\n')

                if (not n % (latex_shape[0] + 1)) & (not m % (latex_shape[1] + 1)):
                    f.write('\n\n\n')
                    f.write('Table for LaTeX:\n')
                    f.write('\\begin{tabular}{|' + '|'.join(['c'] * latex_shape[1]) + '|}\n')
                    f.write('\hline\n')
                    for i in range(n // (latex_shape[0] + 1), solution.shape[0] - 1, n // (latex_shape[0] + 1)):
                        row = solution[i] - analytical_sol[i]
                        elems = row[m // (latex_shape[1] + 1):-1:m // (latex_shape[1] + 1)]
                        formatted_row = ' & '.join([f'{value:12.10f}' for value in elems])
                        f.write(formatted_row + '\\\\\hline\n')
                    f.write('\hline\n')
                    f.write('\\end{tabular}\n')

            elif (not (analytical_sol.shape[0]-1) % latex_shape[0]) & (not n % latex_shape[0]) & \
                    (not (analytical_sol.shape[1]-1) % latex_shape[1]) & (not m % latex_shape[1]):
                f.write(f'Error wrt analytical solution obtained with {n_terms} terms (error ~O(1/N)~'
                        f'{1 / (n_terms - 1)}\n')
                f.write('\n\n\n')
                f.write('Table for LaTeX:\n')
                f.write('\\begin{tabular}{|' + '|'.join(['c'] * (latex_shape[1]-1)) + '|}\n')
                f.write('\hline\n')
                error = get_error_diff_grids(solution=solution, analytical_sol=analytical_sol,
                                             aim_shape=(latex_shape[0]+1, latex_shape[1]+1))
                for i in range(1, error.shape[0] - 1):
                    row = error[i]
                    formatted_row = ' & '.join([f'{value:12.10f}' for value in row[1:-1]])
                    f.write(formatted_row + '\\\\\hline\n')
                f.write('\hline\n')
                f.write('\\end{tabular}\n')

