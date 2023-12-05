import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_3_functions import fourier_series_analytical_sol, cartesian_laplace_eq_df_system, document_test, \
                             get_error_diff_grids, sor_matrix, unpack_cartesian

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import lu_decomposition, lu_solve, sor_method
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import lu_decomposition, lu_solve, sor_method

# Try to import from the current folder; if not found, import from the parent folder
try:
    from task_2_functions import power_method
except ImportError:
    from Task_2.task_2_functions import power_method


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')
if not os.path.exists('./results/'):
    os.makedirs('./results/')


# Define general parameters of the problem:
N_latex = 6
M_latex = 3
radius = 1
boundary_conditions = [0, 1]
N_Fourier = 100001
n_tries = 15


# Obtain the analytical solution at the points of the grid
N = N_latex
M = M_latex

x_vals = np.linspace(-radius, radius, N+1)
y_vals = np.linspace(0, radius, M+1)

print('Computing analytical solution')
tinit = time.time()

analytical_sol = np.ones((M+1, N+1))
for jj, y in enumerate(y_vals):
    for ii, x in enumerate(x_vals):
        rho = np.sqrt(x**2 + y**2)
        if rho <= radius:
            theta = np.arctan2(y, x)
            analytical_sol[jj, ii] = fourier_series_analytical_sol(rho, theta, n_terms=N_Fourier)
print(f' execution time: {time.time()-tinit} s')
print()

filename = f'./results/analytical_sol_cartesian_{N}x{M}_{N_Fourier}_terms.txt'
info = f'Analytical solution of Laplace eq in cartesian coordinates over a mesh with:\n' \
       f' - {N} evenly spaced intervals ({N+1} points) between [{-radius, radius}] in x\n' \
       f' - {M} evenly spaced intervals ({M+1} points) between [{0, radius}] in y\n' \
       f'using {N_Fourier} terms of the series: u(rho,theta) = sum_[n odd] 4/(n*pi)*rho^n*sin(n*theta)'
document_test(filename=filename, solution=analytical_sol, info=info, latex_shape=(M_latex, N_latex))


# Get the linear system representing the edp in polar coordinates
n_subint = 1
print()
print()
print(f'Number of subintervals between req points: {n_subint}')
N = N_latex * n_subint
M = M_latex * n_subint
print(f'Initializing cartesian finite differences system: [{N}x{M}] grid')
A, b = cartesian_laplace_eq_df_system(N, M, radius, boundary_conditions)


# Solve the system using LU decomposition
print('Solving the system using LU decomposition')
tinit = time.time()

L, U = lu_decomposition(A)
u = lu_solve(L, U, b)

print(f' execution time: {time.time()-tinit} s')

# Unpack result
lu_sol = unpack_cartesian(u, N, M, boundary_conditions)

lu_error = get_error_diff_grids(solution=lu_sol, analytical_sol=analytical_sol,
                                aim_shape=(M_latex + 1, N_latex + 1))
print(f' max error: {np.max(np.abs(lu_error))}')

# Document test results
filename = f'./results/finiteDiff_cartesian_LU_sol_{N}x{M}.txt'
info = f'Finite differences solution of Laplace eq in cartesian coordinates over a mesh with:\n' \
       f' - {N} evenly spaced intervals ({N+1} points) between [{-radius, radius}] in x\n' \
       f' - {M} evenly spaced intervals ({M+1} points) between [{0, radius}] in y\n' \
       f'Obtained using LU decomposition for linear system solving'
document_test(filename=filename, solution=lu_sol, info=info, latex_shape=(M_latex, N_latex),
              analytical_sol=analytical_sol, n_terms=N_Fourier)

# Apply SOR method
w = 1.25
print(f'Solving the system using Succesive Over-Relaxation method (w={w})')
# Extract max eigval, to check for convergence
H = sor_matrix(A, w)
max_abs_eigval = 0.0
for ii in range(n_tries):  # Make several tries to ensure we get the actual maximum
    eigval, _ = power_method(H, max_iterations=5000)
    if abs(eigval) > abs(max_abs_eigval):
        max_abs_eigval = eigval
# sor_eigval.append(max_abs_eigval)
print(f' maximum eigenvalue associated to SOR method: {max_abs_eigval}')

try:
    # Using Aitken's acceleration method
    tinit = time.time()
    x, niter = sor_method(A, b, w=w, max_iterations=50000, tolerance=1e-5, aitken=True)
    # sor_nsubint.append(n_subint)
    # sor_texe_aitken.append(time.time()-tinit)
    # sor_niter_aitken.append(niter)
    # print(f' execution time: {sor_texe_aitken[-1]} s')
    print(f' execution time: {time.time()-tinit} s')
    print(f' Converged after {niter} iterations')

    # Unpack result
    sor_sol = unpack_cartesian(x, N, M, boundary_conditions)

    # Compare wrt analytical solution to get error:
    sor_error = get_error_diff_grids(solution=sor_sol, analytical_sol=analytical_sol,
                                     aim_shape=(M_latex+1, N_latex+1))
    # sor_maxerr.append(np.max(np.abs(sor_error)))
    print(f' max eror: {np.max(np.abs(sor_error))}')
    print()

    # Document test results
    filename = f'./results/finiteDiff_cartesian_SOR_{N}x{M}.txt'
    info = f'Finite differences solution of Laplace eq in cartesian coordinates over a mesh with:\n' \
           f' - {N} evenly spaced intervals ({N+1} points) between [{-radius, radius}] in x\n' \
           f' - {M} evenly spaced intervals ({M+1} points) between [{0, radius}] in y\n' \
           f'Obtained using Succesive Over-Relaxation (SOR) method for linear system solving with w={w} ' \
           f'({niter} iterations for convergence)'
    document_test(filename=filename, solution=sor_sol, info=info, latex_shape=(M_latex, N_latex),
                  analytical_sol=analytical_sol, n_terms=N_Fourier)

except RuntimeError as e:
    # Handle the exception (e.g., print an error message)
    print(f"Error: {e}")
    print()
