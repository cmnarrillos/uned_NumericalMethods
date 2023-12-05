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


def cartesian_laplace_eq_df_system(n, m, r, bcs):
    """
    Build the linear system representing the Laplace equation with Dirichlet BCs in cartesian coordinates

    Args:
        n (int): number of intervals in x coordinate
        m (int): number of intervals in y coordinate
        r (float): Radius of the semicircular domain
        bcs (list): list of boundary conditions at [plain_side(y=0), circular_edge(x^2+y^2=r^2)]

    Returns:
        matrix(np.ndarray): matrix of coefficients of the resulting linear system
        vector(np.ndarray): vector of the independent terms in the linear system
    """

    # Extract coordinates relevant info
    x_vals = np.linspace(-r, r, n+1)
    y_vals = np.linspace(0, r, m+1)
    d_x = np.abs(x_vals[1] - x_vals[0])
    d_y = np.abs(y_vals[1] - y_vals[0])

    # Get info about max jj index for each fixed ii
    max_jj = []
    cc_x = []
    for ii, x in enumerate(x_vals):
        for jj, y in enumerate(y_vals):
            if x**2+y**2 >= (1-1e-6)*r**2:
                max_jj.append(jj-1)
                cc_x.append((np.sqrt(r**2-x**2) - y_vals[jj-1])/d_y)
                break

    # Get info about min/max ii index for each fixed jj
    min_ii = []
    max_ii = []
    aa_y = []
    bb_y = []
    for jj, y in enumerate(y_vals):
        start = True
        end = False
        for ii, x in enumerate(x_vals):
            if x**2+y**2 <= (1-1e-6)*r**2:
                if start:
                    min_ii.append(ii)
                    aa_y.append((x_vals[ii] + np.sqrt(r**2-y**2))/d_x)
                    start = False
                    end = True
            elif end:
                max_ii.append(ii-1)
                bb_y.append((np.sqrt(r**2-y**2) - x_vals[ii-1])/d_x)
                break
    num_elems_row = [1 + x - y for x, y in zip(max_ii, min_ii)]

    # Initialize matrix & vector
    size_problem = np.sum([max_jj[ii] for ii in range(1, n)])
    matrix = np.zeros((size_problem, size_problem))
    vector = np.zeros(size_problem)

    # Build matrix & vector
    a = 1/(d_x**2)
    b = 1/(d_y**2)
    index = -1
    for jj in range(1, m):
        for ii in range(min_ii[jj], max_ii[jj]+1):
            index += 1

            # If affected by upper boundary
            if jj == max_jj[ii]:

                # Affected by both upper & left boundaries
                if ii == min_ii[jj]:
                    # Diagonal
                    matrix[index, index] = -2*(a/aa_y[jj] + b/cc_x[ii])
                    # Left/Right elems
                    vector[index] -= 2*a/(aa_y[jj]*(1+aa_y[jj]))*bcs[1]
                    matrix[index, index+1] = 2*a/(1+aa_y[jj])
                    # Up/Down elems
                    vector[index] -= 2*b/(cc_x[ii]*(1+cc_x[ii]))*bcs[1]
                    # Check if affected by y=0 boundary
                    if jj > 1:
                        matrix[index, index-(num_elems_row[jj]+num_elems_row[jj-1])//2] = 2*b/(1+cc_x[ii])
                    else:
                        vector[index] -= 2*b/(1+cc_x[ii])*bcs[0]

                # Affected by both upper & right boundaries
                elif ii == max_ii[jj]:
                    # Diagonal
                    matrix[index, index] = -2*(a/bb_y[jj] + b/cc_x[ii])
                    # Left/Right elems
                    matrix[index, index-1] = 2*a/(1+bb_y[jj])
                    vector[index] -= 2*a/(bb_y[jj]*(1+bb_y[jj]))*bcs[1]
                    # Up/Down elems
                    vector[index] -= 2*b/(cc_x[ii]*(1+cc_x[ii]))*bcs[1]
                    # Check if affected by y=0 boundary
                    if jj > 1:
                        matrix[index, index-(num_elems_row[jj]+num_elems_row[jj-1])//2] = 2*b/(1+cc_x[ii])
                    else:
                        vector[index] -= 2*b/(1+cc_x[ii])*bcs[0]

                # Affected only by upper boundary
                else:
                    # Diagonal
                    matrix[index, index] = -2*(a + b/cc_x[ii])
                    # Left/Right elems
                    matrix[index, index-1] = a
                    matrix[index, index+1] = a
                    # Up/Down elems
                    vector[index] -= 2*b/(cc_x[ii]*(1+cc_x[ii]))*bcs[1]
                    # Check if affected by y=0 boundary
                    if jj > 1:
                        matrix[index, index-(num_elems_row[jj]+num_elems_row[jj-1])//2] = 2*b/(1+cc_x[ii])
                    else:
                        vector[index] -= 2*b/(1+cc_x[ii])*bcs[0]

            # If affected by left boundary
            elif ii == min_ii[jj]:
                # Diagonal
                matrix[index, index] = -2*(a/aa_y[jj] + b)
                # Left/Right elems
                vector[index] -= 2*a/(aa_y[jj]*(1+aa_y[jj]))*bcs[1]
                matrix[index, index+1] = 2*a/(1+aa_y[jj])
                # Up/Down elems
                matrix[index, index+(num_elems_row[jj]+num_elems_row[jj+1])//2] = b
                # Check if affected by y=0 boundary
                if jj > 1:
                    matrix[index, index-(num_elems_row[jj]+num_elems_row[jj-1])//2] = b
                else:
                    vector[index] -= b*bcs[0]

            # If affected by right boundary
            elif ii == max_ii[jj]:
                # Diagonal
                matrix[index, index] = -2*(a/bb_y[jj] + b)
                # Left/Right elems
                matrix[index, index-1] = 2*a/(1+bb_y[jj])
                vector[index] -= 2*a/(bb_y[jj]*(1+bb_y[jj]))*bcs[1]
                # Up/Down elems
                matrix[index, index+(num_elems_row[jj]+num_elems_row[jj+1])//2] = b
                # Check if affected by y=0 boundary
                if jj > 1:
                    matrix[index, index-(num_elems_row[jj]+num_elems_row[jj-1])//2] = b
                else:
                    vector[index] -= b*bcs[0]

            # If point is not affected by any circular boundary
            else:
                # Diagonal
                matrix[index, index] = -2*(a + b)
                # Left/Right elems
                matrix[index, index-1] = a
                matrix[index, index+1] = a
                # Up/Down elems
                matrix[index, index+(num_elems_row[jj]+num_elems_row[jj+1])//2] = b
                # Check if affected by y=0 boundary
                if jj > 1:
                    matrix[index, index-(num_elems_row[jj]+num_elems_row[jj-1])//2] = b
                else:
                    vector[index] -= b*bcs[0]

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
    L = -np.tril(A, k=-1)
    U = -np.triu(A, k=1)

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
    L = -np.tril(A, k=-1)
    U = -np.triu(A, k=1)

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
    L = -np.tril(A, k=-1)
    U = -np.triu(A, k=1)

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


def document_test_cartesian(filename, solution, info='', latex_shape=None, analytical_sol=None, n_terms=None):
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

    return


def unpack_cartesian(solution, n, m, bcs):
    """
    Unpacks vector solution of the linear algebraic system and puts it in a 2d array

    Args:
        solution (np.ndarray): vector with the solution of the linear system of eqs governing the problem
        n (int): number of intervals in x coordinate
        m (int): number of intervals in y coordinate
        bcs (list): list of boundary conditions at [plain_side(y=0), circular_edge(x^2+y^2=r^2)]

    Returns:
        rearranged_solution (np.ndarray): solution taken as input but with the proper format

    """
    x_vals = np.linspace(-1, 1, n+1)
    y_vals = np.linspace(0, 1, m+1)

    # Get info about max jj index for each fixed ii
    max_jj = []
    for ii, x in enumerate(x_vals):
        for jj, y in enumerate(y_vals):
            if x**2+y**2 >= 1:
                max_jj.append(jj-1)
                break

    # Get info about min/max ii index for each fixed jj
    min_ii = []
    max_ii = []
    for jj, y in enumerate(y_vals):
        start = True
        end = False
        for ii, x in enumerate(x_vals):
            if x**2+y**2 <= (1-1e-6):
                if start:
                    min_ii.append(ii)
                    start = False
                    end = True
            elif end:
                max_ii.append(ii-1)
                break

    # Unpack solution to its semicircular shape
    rearranged_solution = np.zeros((m+1, n+1))
    index = -1
    for jj in range(1, m):
        for ii in range(min_ii[jj], max_ii[jj] + 1):
            index += 1
            rearranged_solution[jj, ii] = solution[index]

    # Include bcs
    for ii in range(n+1):
        rearranged_solution[max_jj[ii]+1:, ii] = bcs[1]
    rearranged_solution[0, :] = bcs[0]

    return rearranged_solution
