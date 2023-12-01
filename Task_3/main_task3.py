import os
import numpy as np
import matplotlib.pyplot as plt

from task_3_functions import fourier_series_analytical_sol, polar_laplace_eq_df_system

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import lu_decomposition, lu_solve
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import lu_decomposition, lu_solve


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')
if not os.path.exists('./results/'):
    os.makedirs('./results/')


# Define general parameters of the problem:
N = 12
M = 24
rho_range = (0, 1)
theta_range = (0, np.pi)
boundary_conditions = [0, 1, 0, 0]


# Obtain the analytical solution at the points of the grid
rho_vals = np.linspace(rho_range[0], rho_range[-1], N+1)
theta_vals = np.linspace(theta_range[0], theta_range[-1], M+1)

analytical_sol = np.zeros((N+1, M+1))
for ii, rho in enumerate(rho_vals):
    for jj, theta in enumerate(theta_vals):
        analytical_sol[ii, jj] = fourier_series_analytical_sol(rho, theta, n_terms=100001)


# Get the linear system representing the edp in polar coordinates
A, b = polar_laplace_eq_df_system(N, M, rho_range, theta_range, boundary_conditions)

# Solve the system using LU decomposition
L, U = lu_decomposition(A)
u = lu_solve(L, U, b)

# Unpack result
lu_sol = np.zeros((N+1, M+1))
lu_sol[1:N, 1:M] = np.reshape(u, (N-1, M-1))
# Include BCs:
lu_sol[0, :] = boundary_conditions[0]
lu_sol[N, :] = boundary_conditions[1]
lu_sol[:, 0] = boundary_conditions[2]
lu_sol[:, M] = boundary_conditions[3]

print()
