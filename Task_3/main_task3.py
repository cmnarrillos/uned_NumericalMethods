import os
import time
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
N = 15
M = 30
rho_range = (0, 1)
theta_range = (0, np.pi)
boundary_conditions = [0, 1, 0, 0]
N_Fourier = 100001


# Obtain the analytical solution at the points of the grid
rho_vals = np.linspace(rho_range[0], rho_range[-1], N+1)
theta_vals = np.linspace(theta_range[0], theta_range[-1], M+1)

print('Computing analytical solution')
tinit = time.time()

analytical_sol = np.zeros((N+1, M+1))
for ii, rho in enumerate(rho_vals):
    for jj, theta in enumerate(theta_vals):
        analytical_sol[ii, jj] = fourier_series_analytical_sol(rho, theta, n_terms=N_Fourier)
print(f' execution time: {time.time()-tinit} s')
print()

# Document the results
with open(f'./results/analytical_sol_{N}x{M}_{N_Fourier}_terms.txt', 'w') as f:
    f.write(f'Analytical solution of Laplace eq in polar coordinates over a mesh with:\n')
    f.write(f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n')
    f.write(f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n')
    f.write(f'using {N_Fourier} terms of the series: u(rho,theta) = sum_[n odd] 4/(n*pi)*rho^n*sin(n*theta)\n\n\n')

    for row in analytical_sol:
        formatted_row = ' ' + ', '.join([f'{value:12.10f}' for value in row])
        f.write(formatted_row + '\n')

    if (not N % 3) & (not M % 6):
        f.write('\n\n\n')
        f.write('Table for LaTeX:\n')
        f.write('\\begin{tabular}{|c|c|c|c|c|}\n')
        f.write('\hline\n')
        for i in range(N//3, analytical_sol.shape[0]-1, N//3):
            row = analytical_sol[i]
            elems = row[M//6:-2:M//6]
            formatted_row = ' & '.join([f'{value:12.10f}' for value in elems])
            f.write(formatted_row + '\\\\\hline\n')
        f.write('\\end{tabular}\n')


# Get the linear system representing the edp in polar coordinates
print('Initializing polar finite differences system')
A, b = polar_laplace_eq_df_system(N, M, rho_range, theta_range, boundary_conditions)


# Solve the system using LU decomposition
print('Solving the system using LU decomposition')
tinit = time.time()

L, U = lu_decomposition(A)
u = lu_solve(L, U, b)
print(f' execution time: {time.time()-tinit} s')
print()

# Unpack result
lu_sol = np.zeros((N+1, M+1))
lu_sol[1:N, 1:M] = np.reshape(u, (N-1, M-1))
# Include BCs:
lu_sol[0, :] = boundary_conditions[0]
lu_sol[N, :] = boundary_conditions[1]
lu_sol[:, 0] = boundary_conditions[2]
lu_sol[:, M] = boundary_conditions[3]

with open(f'./results/finiteDiff_LU_sol_{N}x{M}.txt', 'w') as f:
    f.write(f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n')
    f.write(f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n')
    f.write(f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n')
    f.write(f'Obtained using LU decomposition for linear system solving \n\n\n')

    for row in lu_sol:
        formatted_row = ' ' + ', '.join([f'{value:12.10f}' for value in row])
        f.write(formatted_row + '\n')

    if (not N % 3) & (not M % 6):
        f.write('\n\n\n')
        f.write('Table for LaTeX:\n')
        f.write('\\begin{tabular}{|c|c|c|c|c|}\n')
        f.write('\hline\n')
        for i in range(N//3, lu_sol.shape[0]-1, N//3):
            row = lu_sol[i]
            elems = row[M//6:-2:M//6]
            formatted_row = ' & '.join([f'{value:12.10f}' for value in elems])
            f.write(formatted_row + '\\\\\hline\n')
        f.write('\\end{tabular}\n')
    f.write('\n\n\n')

    f.write(f'Error wrt analytical solution obtained with {N_Fourier} terms (error ~O(1/N)~{1/(N_Fourier-1)}\n')
    for row in lu_sol-analytical_sol:
        formatted_row = ' ' + ', '.join([f'{value:+12.10f}' for value in row])
        f.write(formatted_row + '\n')

    if (not N % 3) & (not M % 6):
        f.write('\n\n\n')
        f.write('Table for LaTeX:\n')
        f.write('\\begin{tabular}{|c|c|c|c|c|}\n')
        f.write('\hline\n')
        for i in range(N//3, lu_sol.shape[0]-1, N//3):
            row = lu_sol[i]-analytical_sol[i]
            elems = row[M//6:-2:M//6]
            formatted_row = ' & '.join([f'{value:12.10f}' for value in elems])
            f.write(formatted_row + '\\\\\hline\n')
        f.write('\\end{tabular}\n')

