import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_3_functions import fourier_series_analytical_sol, polar_laplace_eq_df_system, document_test_polar, \
                             get_error_diff_grids, jacobi_matrix, gs_matrix, sor_matrix

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import lu_decomposition, lu_solve, jacobi_method, gauss_seidel, sor_method
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import lu_decomposition, lu_solve, jacobi_method, gauss_seidel, sor_method

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
N_latex = 3
M_latex = 6
rho_range = (0, 1)
theta_range = (0, np.pi)
boundary_conditions = [0, 1, 0, 0]
N_Fourier = 100001

subintervals = [1, 2, 3]#, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Initialize lists to store general vars
lu_texe = []
lu_maxerr = []

jacobi_nsubint = []
jacobi_texe = []
jacobi_maxerr = []
jacobi_niter = []

gs_nsubint = []
gs_texe = []
gs_maxerr = []
gs_niter = []

sor_nsubint = []
sor_texe = []
sor_maxerr = []
sor_niter = []


# Obtain the analytical solution at the points of the grid
N = N_latex
M = M_latex

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


for n_subint in subintervals:
    # Get the linear system representing the edp in polar coordinates
    print()
    print()
    print(f'Number of subintervals between req points: {n_subint}')
    N = N_latex * n_subint
    M = M_latex * n_subint
    print(f'Initializing polar finite differences system: [{N}x{M}] grid')
    A, b = polar_laplace_eq_df_system(N, M, rho_range, theta_range, boundary_conditions)


    # Solve the system using LU decomposition
    print('Solving the system using LU decomposition')
    tinit = time.time()

    L, U = lu_decomposition(A)
    u = lu_solve(L, U, b)

    lu_texe.append(time.time()-tinit)
    print(f' execution time: {lu_texe[-1]} s')

    # Unpack result
    lu_sol = np.zeros((N+1, M+1))
    lu_sol[1:N, 1:M] = np.reshape(u, (N-1, M-1))
    # Include BCs:
    lu_sol[0, :] = boundary_conditions[0]
    lu_sol[N, :] = boundary_conditions[1]
    lu_sol[:, 0] = boundary_conditions[2]
    lu_sol[:, M] = boundary_conditions[3]

    # Compare wrt analytical solution to get error:
    lu_error = get_error_diff_grids(solution=lu_sol, analytical_sol=analytical_sol, aim_shape=(N_latex+1, M_latex+1))
    lu_maxerr.append(np.max(np.abs(lu_error)))
    print(f' max eror: {lu_maxerr[-1]}')
    print()

    # Document test results
    filename = f'./results/finiteDiff_LU_sol_{N}x{M}.txt'
    info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
           f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
           f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
           f'Obtained using LU decomposition for linear system solving'
    document_test_polar(filename=filename, solution=lu_sol, info=info, latex_shape=(N_latex, M_latex),
                        analytical_sol=analytical_sol, n_terms=N_Fourier)


    # Solve the system using Jacobi iterative method
    print('Solving the system using Jacobi method')
    H = jacobi_matrix(A)
    eigval, eigvec = power_method(H)
    print(f' maximum eigenvalue associated to Jacobi method: {eigval}')
    tinit = time.time()

    try:
        x, niter = jacobi_method(A, b, max_iterations=1000000, tolerance=1e-5)

        jacobi_nsubint.append(n_subint)
        jacobi_texe.append(time.time()-tinit)
        jacobi_niter.append(niter)
        print(f' execution time: {jacobi_texe[-1]} s')
        print(f' Converged after {niter} iterations')

        # Unpack result
        jacobi_sol = np.zeros((N + 1, M + 1))
        jacobi_sol[1:N, 1:M] = np.reshape(x, (N - 1, M - 1))
        # Include BCs:
        jacobi_sol[0, :] = boundary_conditions[0]
        jacobi_sol[N, :] = boundary_conditions[1]
        jacobi_sol[:, 0] = boundary_conditions[2]
        jacobi_sol[:, M] = boundary_conditions[3]

        # Compare wrt analytical solution to get error:
        jacobi_error = get_error_diff_grids(solution=jacobi_sol, analytical_sol=analytical_sol,
                                            aim_shape=(N_latex+1, M_latex+1))
        jacobi_maxerr.append(np.max(np.abs(jacobi_error)))
        print(f' max eror: {jacobi_maxerr[-1]}')
        print()

        # Document test results
        filename = f'./results/finiteDiff_Jacobi_{N}x{M}.txt'
        info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
               f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
               f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
               f'Obtained using Jacobi method for linear system solving ({niter} iterations for convergence)'
        document_test_polar(filename=filename, solution=jacobi_sol, info=info, latex_shape=(N_latex, M_latex),
                            analytical_sol=analytical_sol, n_terms=N_Fourier)

    except RuntimeError as e:
        # Handle the exception (e.g., print an error message)
        print(f"Error: {e}")
        print()

    # Solve the system using Gauss-Seidel iterative method
    print('Solving the system using Gauss-Seidel method')
    H = gs_matrix(A)
    eigval, eigvec = power_method(H)
    print(f' maximum eigenvalue associated to Gauss-Seidel method: {eigval}')
    tinit = time.time()

    try:
        x, niter = gauss_seidel(A, b, max_iterations=1000000, tolerance=1e-5)

        gs_nsubint.append(n_subint)
        gs_texe.append(time.time()-tinit)
        gs_niter.append(niter)
        print(f' execution time: {gs_texe[-1]} s')
        print(f' Converged after {niter} iterations')

        # Unpack result
        gs_sol = np.zeros((N+1, M+1))
        gs_sol[1:N, 1:M] = np.reshape(x, (N-1, M-1))
        # Include BCs:
        gs_sol[0, :] = boundary_conditions[0]
        gs_sol[N, :] = boundary_conditions[1]
        gs_sol[:, 0] = boundary_conditions[2]
        gs_sol[:, M] = boundary_conditions[3]

        # Compare wrt analytical solution to get error:
        gs_error = get_error_diff_grids(solution=gs_sol, analytical_sol=analytical_sol,
                                        aim_shape=(N_latex+1, M_latex+1))
        gs_maxerr.append(np.max(np.abs(gs_error)))
        print(f' max eror: {gs_maxerr[-1]}')
        print()

        # Document test results
        filename = f'./results/finiteDiff_GaussSeidel_{N}x{M}.txt'
        info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
               f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
               f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
               f'Obtained using Gauss-Seidel method for linear system solving ({niter} iterations for convergence)'
        document_test_polar(filename=filename, solution=gs_sol, info=info, latex_shape=(N_latex, M_latex),
                            analytical_sol=analytical_sol, n_terms=N_Fourier)

    except RuntimeError as e:
        # Handle the exception (e.g., print an error message)
        print(f"Error: {e}")
        print()


    # Solve the system using SOR iterative method
    w = 1.25
    print(f'Solving the system using Succesive Over-Relaxation method (w={w})')
    H = sor_matrix(A, w)
    eigval, eigvec = power_method(H)
    print(f' maximum eigenvalue associated to SOR method: {eigval}')
    tinit = time.time()

    try:
        x, niter = sor_method(A, b, w=w, max_iterations=1000000, tolerance=1e-5)

        sor_nsubint.append(n_subint)
        sor_texe.append(time.time()-tinit)
        sor_niter.append(niter)
        print(f' execution time: {sor_texe[-1]} s')
        print(f' Converged after {niter} iterations')

        # Unpack result
        sor_sol = np.zeros((N+1, M+1))
        sor_sol[1:N, 1:M] = np.reshape(x, (N-1, M-1))
        # Include BCs:
        sor_sol[0, :] = boundary_conditions[0]
        sor_sol[N, :] = boundary_conditions[1]
        sor_sol[:, 0] = boundary_conditions[2]
        sor_sol[:, M] = boundary_conditions[3]

        # Compare wrt analytical solution to get error:
        sor_error = get_error_diff_grids(solution=sor_sol, analytical_sol=analytical_sol,
                                         aim_shape=(N_latex+1, M_latex+1))
        sor_maxerr.append(np.max(np.abs(sor_error)))
        print(f' max eror: {sor_maxerr[-1]}')
        print()

        # Document test results
        filename = f'./results/finiteDiff_SOR_{N}x{M}.txt'
        info = f'Finite differences solution of Laplace eq in polar coordinates over a mesh with:\n' \
               f' - {N} evenly spaced intervals ({N+1} points) between [{rho_range[0], rho_range[-1]}] in r\n' \
               f' - {M} evenly spaced intervals ({M+1} points) between [{theta_range[0], theta_range[-1]}] in theta\n' \
               f'Obtained using Succesive Over-Relaxation (SOR) method for linear system solving with w={w} ' \
               f'({niter} iterations for convergence)'
        document_test_polar(filename=filename, solution=gs_sol, info=info, latex_shape=(N_latex, M_latex),
                            analytical_sol=analytical_sol, n_terms=N_Fourier)

    except RuntimeError as e:
        # Handle the exception (e.g., print an error message)
        print(f"Error: {e}")
        print()


# Plot general stats
# Execution time
plt.figure()
plt.semilogy(jacobi_nsubint, jacobi_texe, label='Jacobi')
plt.semilogy(gs_nsubint, gs_texe, label='Gauss-Seidel')
plt.semilogy(sor_nsubint, sor_texe, label=f'SOR (w={w})')
plt.semilogy(subintervals, lu_texe, label='LU')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$t_{exe}$ [s]', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/texe.png', bbox_inches='tight')


# Niter
plt.figure()
plt.semilogy(jacobi_nsubint, jacobi_niter, label='Jacobi')
plt.semilogy(gs_nsubint, gs_niter, label='Gauss-Seidel')
plt.semilogy(sor_nsubint, sor_niter, label=f'SOR (w={w})')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('# of iterations', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/niter.png', bbox_inches='tight')


# Max error
plt.figure()
plt.semilogy(jacobi_nsubint, jacobi_maxerr, label='Jacobi')
plt.semilogy(gs_nsubint, gs_maxerr, label='Gauss-Seidel')
plt.semilogy(sor_nsubint, sor_maxerr, label=f'SOR (w={w})')
plt.semilogy(subintervals, lu_maxerr, label='LU')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$max(\\varepsilon)$', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/error.png', bbox_inches='tight')
