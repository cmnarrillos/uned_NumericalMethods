import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_3_functions import fourier_series_analytical_sol, polar_laplace_eq_df_system, document_test_polar

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import lu_decomposition, lu_solve, jacobi_method, gauss_seidel, sor_method
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import lu_decomposition, lu_solve, jacobi_method, gauss_seidel, sor_method


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')
if not os.path.exists('./results/'):
    os.makedirs('./results/')


# Define general parameters of the problem:
N = 3
M = 6
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

# Document results
filename = f'./results/analytical_sol_{N}x{M}_{N_Fourier}_terms.txt'
info = f'Analytical solution of Laplace eq in polar coordinates over a mesh with:\n' \
       f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
       f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
       f'using {N_Fourier} terms of the series: u(rho,theta) = sum_[n odd] 4/(n*pi)*rho^n*sin(n*theta)'
document_test_polar(filename=filename, solution=analytical_sol, info=info, latex_shape=(2, 5))


# Get the linear system representing the edp in polar coordinates
print('Initializing polar finite differences system')
N = 30
M = 60
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

# Document test results
filename = f'./results/finiteDiff_LU_sol_{N}x{M}.txt'
info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
       f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
       f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
       f'Obtained using LU decomposition for linear system solving'
document_test_polar(filename=filename, solution=lu_sol, info=info, latex_shape=(2, 5), analytical_sol=analytical_sol,
                    n_terms=N_Fourier)


# Solve the system using Jacobi iterative method
print('Solving the system using Jacobi method')
tinit = time.time()

try:
    x, niter = jacobi_method(A, b)
    print(f' execution time: {time.time()-tinit} s')
    print(f' Converged after {niter} iterations')
    print()

    # Unpack result
    jacobi_sol = np.zeros((N + 1, M + 1))
    jacobi_sol[1:N, 1:M] = np.reshape(x, (N - 1, M - 1))
    # Include BCs:
    jacobi_sol[0, :] = boundary_conditions[0]
    jacobi_sol[N, :] = boundary_conditions[1]
    jacobi_sol[:, 0] = boundary_conditions[2]
    jacobi_sol[:, M] = boundary_conditions[3]

    # Document test results
    filename = f'./results/finiteDiff_Jacobi_{N}x{M}.txt'
    info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
           f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
           f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
           f'Obtained using Jacobi method for linear system solving ({niter} iterations for convergence)'
    document_test_polar(filename=filename, solution=jacobi_sol, info=info, latex_shape=(2, 5), analytical_sol=analytical_sol,
                        n_terms=N_Fourier)

except RuntimeError as e:
    # Handle the exception (e.g., print an error message)
    print(f"Error: {e}")
    print()

# Solve the system using Gauss-Seidel iterative method
print('Solving the system using Gauss-Seidel method')
tinit = time.time()

try:
    x, niter = gauss_seidel(A, b)
    print(f' execution time: {time.time()-tinit} s')
    print(f' Converged after {niter} iterations')
    print()

    # Unpack result
    gs_sol = np.zeros((N+1, M+1))
    gs_sol[1:N, 1:M] = np.reshape(x, (N-1, M-1))
    # Include BCs:
    gs_sol[0, :] = boundary_conditions[0]
    gs_sol[N, :] = boundary_conditions[1]
    gs_sol[:, 0] = boundary_conditions[2]
    gs_sol[:, M] = boundary_conditions[3]

    # Document test results
    filename = f'./results/finiteDiff_GaussSeidel_{N}x{M}.txt'
    info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
           f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
           f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
           f'Obtained using Gauss-Seidel method for linear system solving ({niter} iterations for convergence)'
    document_test_polar(filename=filename, solution=gs_sol, info=info, latex_shape=(2, 5), analytical_sol=analytical_sol,
                        n_terms=N_Fourier)

except RuntimeError as e:
    # Handle the exception (e.g., print an error message)
    print(f"Error: {e}")
    print()


# Solve the system using SOR iterative method
print('Solving the system using Succesive Over-Relaxation method')
tinit = time.time()

try:
    x, niter = sor_method(A, b, w=1.25)
    print(f' execution time: {time.time()-tinit} s')
    print(f' Converged after {niter} iterations')
    print()

    # Unpack result
    sor_sol = np.zeros((N+1, M+1))
    sor_sol[1:N, 1:M] = np.reshape(x, (N-1, M-1))
    # Include BCs:
    sor_sol[0, :] = boundary_conditions[0]
    sor_sol[N, :] = boundary_conditions[1]
    sor_sol[:, 0] = boundary_conditions[2]
    sor_sol[:, M] = boundary_conditions[3]

    # Document test results
    filename = f'./results/finiteDiff_SOR_{N}x{M}.txt'
    info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
           f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
           f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
           f'Obtained using Succesive Over-Relaxation (SOR) method for linear system solving ({niter} iterations for ' \
           f'convergence)'
    document_test_polar(filename=filename, solution=gs_sol, info=info, latex_shape=(2, 5), analytical_sol=analytical_sol,
                        n_terms=N_Fourier)

except RuntimeError as e:
    # Handle the exception (e.g., print an error message)
    print(f"Error: {e}")
    print()