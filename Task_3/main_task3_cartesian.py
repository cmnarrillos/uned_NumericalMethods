import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_3_functions import fourier_series_analytical_sol, cartesian_laplace_eq_df_system, document_test_cartesian, \
                             get_error_diff_grids, sor_matrix, unpack_cartesian

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import lu_decomposition, lu_solve, sor_method
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import lu_decomposition, lu_solve, sor_method


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


# Get the linear system representing the edp in polar coordinates
n_subint = 10
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


print()