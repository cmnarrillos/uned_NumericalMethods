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

subintervals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Which methods to run
use_LU = True
use_SOR = True

# Initialize lists to store general vars
if use_LU:
    lu_texe = []
    lu_maxerr = []
if use_SOR:
    sor_nsubint = []
    sor_eigval = []
    sor_texe = []
    sor_niter = []
    sor_maxerr = []
    sor_texe_aitken = []
    sor_niter_aitken = []


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



for n_subint in subintervals:
    # Get the linear system representing the edp in polar coordinates
    print()
    print()
    print(f'Number of subintervals between req points: {n_subint}')
    N = N_latex * n_subint
    M = M_latex * n_subint
    print(f'Initializing cartesian finite differences system: [{N}x{M}] grid')
    A, b = cartesian_laplace_eq_df_system(N, M, radius, boundary_conditions)


    # Solve the system using LU decomposition
    if use_LU:
        print('Solving the system using LU decomposition')
        tinit = time.time()

        L, U = lu_decomposition(A)
        u = lu_solve(L, U, b)

        lu_texe.append(time.time()-tinit)
        print(f' execution time: {lu_texe[-1]} s')

        # Unpack result
        lu_sol = unpack_cartesian(u, N, M, boundary_conditions)

        # Compare wrt analytical solution to get error:
        lu_error = get_error_diff_grids(solution=lu_sol, analytical_sol=analytical_sol,
                                        aim_shape=(M_latex + 1, N_latex + 1))
        lu_maxerr.append(np.max(np.abs(lu_error)))
        print(f' max eror: {lu_maxerr[-1]}')
        print()

        # Document test results
        filename = f'./results/finiteDiff_cartesian_LU_sol_{N}x{M}.txt'
        info = f'Finite differences solution of Laplace eq in cartesian coordinates over a mesh with:\n' \
               f' - {N} evenly spaced intervals ({N+1} points) between [{-radius, radius}] in x\n' \
               f' - {M} evenly spaced intervals ({M+1} points) between [{0, radius}] in y\n' \
               f'Obtained using LU decomposition for linear system solving'
        document_test(filename=filename, solution=lu_sol, info=info, latex_shape=(M_latex, N_latex),
                      analytical_sol=analytical_sol, n_terms=N_Fourier)

    # Apply SOR method
    if use_SOR:
        w = 1.25
        print(f'Solving the system using Succesive Over-Relaxation method (w={w})')
        # Extract max eigval, to check for convergence
        H = sor_matrix(A, w)
        max_abs_eigval = 0.0
        for ii in range(n_tries):  # Make several tries to ensure we get the actual maximum
            eigval, _ = power_method(H, max_iterations=5000)
            if abs(eigval) > abs(max_abs_eigval):
                max_abs_eigval = eigval
        sor_eigval.append(max_abs_eigval)
        print(f' maximum eigenvalue associated to SOR method: {max_abs_eigval}')

        try:
            # Using Aitken's acceleration method
            tinit = time.time()
            x, niter = sor_method(A, b, w=w, max_iterations=50000, tolerance=1e-5, aitken=True)
            sor_nsubint.append(n_subint)
            sor_texe.append(time.time()-tinit)
            sor_niter.append(niter)
            print(f' execution time: {sor_texe[-1]} s')
            print(f' Converged after {niter} iterations')

            # Unpack result
            sor_sol = unpack_cartesian(x, N, M, boundary_conditions)

            # Compare wrt analytical solution to get error:
            sor_error = get_error_diff_grids(solution=sor_sol, analytical_sol=analytical_sol,
                                             aim_shape=(M_latex+1, N_latex+1))
            sor_maxerr.append(np.max(np.abs(sor_error)))
            print(f' max eror: {sor_maxerr[-1]}')
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

# Plot general stats
# Execution time
plt.figure()
if use_SOR:
    plt.semilogy(sor_nsubint, sor_texe, 'b-+', label=f'SOR (w={w})')
if use_LU:
    plt.semilogy(subintervals, lu_texe, 'k-+', label='LU')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$t_{exe}$ [s]', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/cartesian_texe.png', bbox_inches='tight')

# Execution time
plt.figure()
if use_SOR:
    plt.loglog(sor_nsubint, sor_texe, 'b-+', label=f'SOR (w={w})')
if use_LU:
    plt.loglog(subintervals, lu_texe, 'k-+', label='LU')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$t_{exe}$ [s]', fontsize=18)
plt.legend(fontsize=14)
# plt.xlim((subintervals[0]/10, subintervals[-1]*10))
plt.savefig(f'./Figures/cartesian_texe_log.png', bbox_inches='tight')


# Eigvals
plt.figure()
if use_SOR:
    plt.plot(subintervals, sor_eigval, 'b-+', label=f'SOR (w={w})')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$max(|\lambda|)$', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/cartesian_eigval.png', bbox_inches='tight')


# Eigvals abs
plt.figure()
if use_SOR:
    plt.semilogy(subintervals, [1-abs(eigv) for eigv in sor_eigval], 'b-+', label=f'SOR (w={w})')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$1-max(|\lambda|)$', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/cartesian_eigval_abs.png', bbox_inches='tight')


# Niter
plt.figure()
if use_SOR:
    plt.semilogy(sor_nsubint, sor_niter, 'b-+', label=f'SOR (w={w})')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('# of iterations', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/cartesian_niter.png', bbox_inches='tight')


# Niter
plt.figure()
if use_SOR:
    plt.loglog(sor_nsubint, sor_niter, 'b-+', label=f'SOR (w={w})')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('# of iterations', fontsize=18)
plt.legend(fontsize=14)
# plt.xlim((subintervals[0]/10, subintervals[-1]*10))
plt.savefig(f'./Figures/cartesian_niter_log.png', bbox_inches='tight')


# Max error
plt.figure()
if use_SOR:
    plt.semilogy(sor_nsubint, sor_maxerr, 'b-+', label=f'SOR (w={w})')
if use_LU:
    plt.semilogy(subintervals, lu_maxerr, 'k-+', label='LU')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$max(\\varepsilon)$', fontsize=18)
plt.legend(fontsize=14)
plt.xlim((subintervals[0], subintervals[-1]))
plt.savefig(f'./Figures/cartesian_error.png', bbox_inches='tight')


# Max error
plt.figure()
if use_SOR:
    plt.loglog(sor_nsubint, sor_maxerr, 'b-+', label=f'SOR (w={w})')
if use_LU:
    plt.loglog(subintervals, lu_maxerr, 'k-+', label='LU')
plt.grid(which='both')
plt.xlabel('# of subintervals', fontsize=18)
plt.ylabel('$max(\\varepsilon)$', fontsize=18)
plt.legend(fontsize=14)
# plt.xlim((subintervals[0]/10, subintervals[-1]*10))
plt.savefig(f'./Figures/cartesian_error_log.png', bbox_inches='tight')


