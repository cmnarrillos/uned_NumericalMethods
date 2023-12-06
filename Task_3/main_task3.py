import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_3_functions import fourier_series_analytical_sol, polar_laplace_eq_df_system, document_test, \
                             get_error_diff_grids, jacobi_matrix, gs_matrix, sor_matrix, plot_polar_colormap

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
if not os.path.exists('./Figures/cmaps/'):
    os.makedirs('./Figures/cmaps/')
if not os.path.exists('./Figures/cmaps_adim/'):
    os.makedirs('./Figures/cmaps_adim/')
if not os.path.exists('./results/'):
    os.makedirs('./results/')


# Define general parameters of the problem:
N_latex = 3
M_latex = 6
rho_range = (0, 1)
theta_range = (0, np.pi)
boundary_conditions = [0, 1, 0, 0]
N_Fourier = 100001
n_tries = 15

radius = 1
T_0 = -10
T_1 = 90

subintervals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# What to plot
plt_gral = False
plt_output = True
if not plt_gral and plt_output:
    subintervals = [subintervals[-1]]

# Which methods to run
use_LU = True
use_Jacobi = False
use_GS = False
use_SOR = False

# Initialize lists to store general vars
if use_LU:
    lu_texe = []
    lu_maxerr = []
if use_Jacobi:
    jacobi_nsubint = []
    jacobi_eigval = []
    jacobi_texe = []
    jacobi_niter = []
    jacobi_maxerr = []
    jacobi_texe_aitken = []
    jacobi_niter_aitken = []
if use_GS:
    gs_nsubint = []
    gs_eigval = []
    gs_texe = []
    gs_niter = []
    gs_maxerr = []
    gs_texe_aitken = []
    gs_niter_aitken = []
if use_SOR:
    sor_nsubint = []
    sor_eigval = []
    sor_texe = []
    sor_niter = []
    sor_maxerr = []
    sor_texe_aitken = []
    sor_niter_aitken = []
# Needed afterwards
jacobi_failed = True
gs_failed = True
sor_failed = True


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
document_test(filename=filename, solution=analytical_sol, info=info, latex_shape=(N_latex, M_latex))


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
    if use_LU:
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
        document_test(filename=filename, solution=lu_sol, info=info, latex_shape=(N_latex, M_latex),
                      analytical_sol=analytical_sol, n_terms=N_Fourier)


    # Solve the system using Jacobi iterative method
    if use_Jacobi:
        print('Solving the system using Jacobi method')
        # Extract max eigval, to check for convergence
        H = jacobi_matrix(A)
        max_abs_eigval = 0.0
        for ii in range(n_tries):  # Make several tries to ensure we get the actual maximum
            eigval, _ = power_method(H)
            if abs(eigval) > abs(max_abs_eigval):
                max_abs_eigval = eigval
        jacobi_eigval.append(max_abs_eigval)
        print(f' maximum eigenvalue associated to Jacobi method: {max_abs_eigval}')

        try:
            # Using Aitken acceleration method
            tinit = time.time()
            x, niter = jacobi_method(A, b, max_iterations=50000, tolerance=1e-5, aitken=True)
            jacobi_texe_aitken.append(time.time()-tinit)
            jacobi_nsubint.append(n_subint)
            jacobi_niter_aitken.append(niter)
            print(f' execution time: {jacobi_texe_aitken[-1]} s')
            print(f' Converged after {niter} iterations')

            # Not using Aitken's
            tinit = time.time()
            _, niter = jacobi_method(A, b, max_iterations=50000, tolerance=1e-5, aitken=False)
            jacobi_texe.append(time.time()-tinit)
            jacobi_niter.append(niter)
            print(f' Not using Aitken: converged after {niter} iterations ({jacobi_texe[-1]} s)')

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
            document_test(filename=filename, solution=jacobi_sol, info=info, latex_shape=(N_latex, M_latex),
                                analytical_sol=analytical_sol, n_terms=N_Fourier)
            jacobi_failed = False

        except RuntimeError as e:
            # Handle the exception (e.g., print an error message)
            print(f"Error: {e}")
            print()
            jacobi_failed = True

    # Solve the system using Gauss-Seidel iterative method
    if use_GS:
        print('Solving the system using Gauss-Seidel method')
        # Extract max eigval, to check for convergence
        H = gs_matrix(A)
        max_abs_eigval = 0.0
        for ii in range(n_tries):  # Make several tries to ensure we get the actual maximum
            eigval, _ = power_method(H)
            if abs(eigval) > abs(max_abs_eigval):
                max_abs_eigval = eigval
        gs_eigval.append(max_abs_eigval)
        print(f' maximum eigenvalue associated to Gauss-Seidel method: {max_abs_eigval}')

        try:
            # Using Aitken's acceleration method
            tinit = time.time()
            x, niter = gauss_seidel(A, b, max_iterations=50000, tolerance=1e-5, aitken=True)
            gs_nsubint.append(n_subint)
            gs_texe_aitken.append(time.time()-tinit)
            gs_niter_aitken.append(niter)
            print(f' execution time: {gs_texe_aitken[-1]} s')
            print(f' Converged after {niter} iterations')

            # Not using Aitken's
            tinit = time.time()
            _, niter = gauss_seidel(A, b, max_iterations=50000, tolerance=1e-5, aitken=False)
            gs_texe.append(time.time()-tinit)
            gs_niter.append(niter)
            print(f' Not Using Aitken: converged after {niter} iterations ({gs_texe[-1]} s)')

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
            document_test(filename=filename, solution=gs_sol, info=info, latex_shape=(N_latex, M_latex),
                                analytical_sol=analytical_sol, n_terms=N_Fourier)
            gs_failed = False

        except RuntimeError as e:
            # Handle the exception (e.g., print an error message)
            print(f"Error: {e}")
            print()
            gs_failed = True


    # Solve the system using SOR iterative method
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
            sor_texe_aitken.append(time.time()-tinit)
            sor_niter_aitken.append(niter)
            print(f' execution time: {sor_texe_aitken[-1]} s')
            print(f' Converged after {niter} iterations')

            # Not using Aitken's
            tinit = time.time()
            _, niter = sor_method(A, b, w=w, max_iterations=50000, tolerance=1e-5, aitken=False)
            sor_texe.append(time.time()-tinit)
            sor_niter.append(niter)
            print(f' Not Using Aitken: converged after {niter} iterations ({sor_texe[-1]} s)')

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
            document_test(filename=filename, solution=sor_sol, info=info, latex_shape=(N_latex, M_latex),
                                analytical_sol=analytical_sol, n_terms=N_Fourier)
            sor_failed = False

        except RuntimeError as e:
            # Handle the exception (e.g., print an error message)
            print(f"Error: {e}")
            print()
            sor_failed = True


# Plot general stats
if plt_gral & len(subintervals)>1:
    # Execution time
    plt.figure()
    if use_Jacobi:
        plt.semilogy(jacobi_nsubint, jacobi_texe, 'r-+', label='Jacobi')
        plt.semilogy(jacobi_nsubint, jacobi_texe_aitken, 'r--+')
    if use_GS:
        plt.semilogy(gs_nsubint, gs_texe, 'g-+', label='Gauss-Seidel')
        plt.semilogy(gs_nsubint, gs_texe_aitken, 'g--+')
    if use_SOR:
        plt.semilogy(sor_nsubint, sor_texe, 'b-+', label=f'SOR (w={w})')
        plt.semilogy(sor_nsubint, sor_texe_aitken, 'b--+')
    if use_LU:
        plt.semilogy(subintervals, lu_texe, 'k-+', label='LU')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('$t_{exe}$ [s]', fontsize=18)
    plt.legend(fontsize=14)
    plt.xlim((subintervals[0], subintervals[-1]))
    plt.savefig(f'./Figures/texe.png', bbox_inches='tight')

    # Execution time
    plt.figure()
    if use_Jacobi:
        plt.loglog(jacobi_nsubint, jacobi_texe, 'r-+', label='Jacobi')
        plt.loglog(jacobi_nsubint, jacobi_texe_aitken, 'r--+')
    if use_GS:
        plt.loglog(gs_nsubint, gs_texe, 'g-+', label='Gauss-Seidel')
        plt.loglog(gs_nsubint, gs_texe_aitken, 'g--+')
    if use_SOR:
        plt.loglog(sor_nsubint, sor_texe, 'b-+', label=f'SOR (w={w})')
        plt.loglog(sor_nsubint, sor_texe_aitken, 'b--+')
    if use_LU:
        plt.loglog(subintervals, lu_texe, 'k-+', label='LU')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('$t_{exe}$ [s]', fontsize=18)
    plt.legend(fontsize=14)
    # plt.xlim((subintervals[0]/10, subintervals[-1]*10))
    plt.savefig(f'./Figures/texe_log.png', bbox_inches='tight')


    # Eigvals
    plt.figure()
    if use_Jacobi:
        plt.plot(subintervals, jacobi_eigval, 'r-+', label='Jacobi')
    if use_GS:
        plt.plot(subintervals, gs_eigval, 'g-+', label='Gauss-Seidel')
    if use_SOR:
        plt.plot(subintervals, sor_eigval, 'b-+', label=f'SOR (w={w})')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('$max(|\lambda|)$', fontsize=18)
    plt.legend(fontsize=14)
    plt.xlim((subintervals[0], subintervals[-1]))
    plt.savefig(f'./Figures/eigval.png', bbox_inches='tight')


    # Eigvals abs
    plt.figure()
    if use_Jacobi:
        plt.semilogy(subintervals, [1-abs(eigv) for eigv in jacobi_eigval], 'r-+', label='Jacobi')
    if use_GS:
        plt.semilogy(subintervals, [1-abs(eigv) for eigv in gs_eigval], 'g-+', label='Gauss-Seidel')
    if use_SOR:
        plt.semilogy(subintervals, [1-abs(eigv) for eigv in sor_eigval], 'b-+', label=f'SOR (w={w})')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('$1-max(|\lambda|)$', fontsize=18)
    plt.legend(fontsize=14)
    plt.xlim((subintervals[0], subintervals[-1]))
    plt.savefig(f'./Figures/eigval_abs.png', bbox_inches='tight')


    # Niter
    plt.figure()
    if use_Jacobi:
        plt.semilogy(jacobi_nsubint, jacobi_niter, 'r-+', label='Jacobi')
        plt.semilogy(jacobi_nsubint, jacobi_niter_aitken, 'r--+')
    if use_GS:
        plt.semilogy(gs_nsubint, gs_niter, 'g-+', label='Gauss-Seidel')
        plt.semilogy(gs_nsubint, gs_niter_aitken, 'g--+')
    if use_SOR:
        plt.semilogy(sor_nsubint, sor_niter, 'b-+', label=f'SOR (w={w})')
        plt.semilogy(sor_nsubint, sor_niter_aitken, 'b--+')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('# of iterations', fontsize=18)
    plt.legend(fontsize=14)
    plt.xlim((subintervals[0], subintervals[-1]))
    plt.savefig(f'./Figures/niter.png', bbox_inches='tight')


    # Niter
    plt.figure()
    if use_Jacobi:
        plt.loglog(jacobi_nsubint, jacobi_niter, 'r-+', label='Jacobi')
        plt.loglog(jacobi_nsubint, jacobi_niter_aitken, 'r--+')
    if use_GS:
        plt.loglog(gs_nsubint, gs_niter, 'g-+', label='Gauss-Seidel')
        plt.loglog(gs_nsubint, gs_niter_aitken, 'g--+')
    if use_SOR:
        plt.loglog(sor_nsubint, sor_niter, 'b-+', label=f'SOR (w={w})')
        plt.loglog(sor_nsubint, sor_niter_aitken, 'b--+')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('# of iterations', fontsize=18)
    plt.legend(fontsize=14)
    # plt.xlim((subintervals[0]/10, subintervals[-1]*10))
    plt.savefig(f'./Figures/niter_log.png', bbox_inches='tight')


    # Max error
    plt.figure()
    if use_Jacobi:
        plt.semilogy(jacobi_nsubint, jacobi_maxerr, 'r-+', label='Jacobi')
    if use_GS:
        plt.semilogy(gs_nsubint, gs_maxerr, 'g-+', label='Gauss-Seidel')
    if use_SOR:
        plt.semilogy(sor_nsubint, sor_maxerr, 'b-+', label=f'SOR (w={w})')
    if use_LU:
        plt.semilogy(subintervals, lu_maxerr, 'k-+', label='LU')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('$max(\\varepsilon)$', fontsize=18)
    plt.legend(fontsize=14)
    plt.xlim((subintervals[0], subintervals[-1]))
    plt.savefig(f'./Figures/error.png', bbox_inches='tight')


    # Max error
    plt.figure()
    if use_Jacobi:
        plt.loglog(jacobi_nsubint, jacobi_maxerr, 'r-+', label='Jacobi')
    if use_GS:
        plt.loglog(gs_nsubint, gs_maxerr, 'g-+', label='Gauss-Seidel')
    if use_SOR:
        plt.loglog(sor_nsubint, sor_maxerr, 'b-+', label=f'SOR (w={w})')
    if use_LU:
        plt.loglog(subintervals, lu_maxerr, 'k-+', label='LU')
    plt.grid(which='both')
    plt.xlabel('# of subintervals', fontsize=18)
    plt.ylabel('$max(\\varepsilon)$', fontsize=18)
    plt.legend(fontsize=14)
    # plt.xlim((subintervals[0]/10, subintervals[-1]*10))
    plt.savefig(f'./Figures/error_log.png', bbox_inches='tight')






# Plot distribution of temperature
if plt_output:
    rho_vals_distrib = np.linspace(0, 1, N + 1)
    theta_vals_distrib = np.linspace(0, np.pi, M + 1)

    rho_vals_ref = rho_vals
    theta_vals_ref = theta_vals

    # Plot analytical temperature distribution
    filename = f'./Figures/cmaps/polar_T_map_{N_latex}x{M_latex}_{N_Fourier}terms.png'
    title = 'Temperature distribution - Analytical ($T(r,\\theta)$)'
    plot_polar_colormap(radius * rho_vals_ref, theta_vals_ref, solution=T_0 + (T_1 - T_0) * analytical_sol,
                        filename=filename, title=title)

    # Non-dimensional plot
    # Plot Temperature map obtained with LU method
    filename = f'./Figures/cmaps_adim/polar_T_map_{N_latex}x{M_latex}_{N_Fourier}terms.png'
    title = 'Temperature non-dimensional distribution - Analytical ($u(\\rho,\\theta)$)'
    plot_polar_colormap(rho_vals_ref, theta_vals_ref, solution=analytical_sol, filename=filename, title=title)

    if use_LU:
        # Plot Temperature map obtained with LU method
        filename = f'./Figures/cmaps/polar_T_map_LU_{N}x{M}.png'
        title = 'Temperature distribution - LU ($T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_distrib, theta_vals_distrib, solution=T_0 + (T_1 - T_0) * lu_sol,
                            filename=filename, title=title)

        # Plot Error map obtained with LU method
        filename = f'./Figures/cmaps/polar_errT_map_LU_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature error - LU ($\Delta T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_ref, theta_vals_ref, solution=(T_1 - T_0) * lu_error, filename=filename,
                            title=title)

        # Non-dimensional plots
        # Plot Temperature map obtained with LU method
        filename = f'./Figures/cmaps_adim/polar_T_map_LU_{N}x{M}.png'
        title = 'Temperature non-dimensional distribution - LU ($u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_distrib, theta_vals_distrib, solution=lu_sol, filename=filename, title=title)

        # Plot Error map obtained with LU method
        filename = f'./Figures/cmaps_adim/polar_errT_map_LU_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature non-dimensional error - LU ($\Delta u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_ref, theta_vals_ref, solution=lu_error, filename=filename, title=title)

    if use_Jacobi & (not jacobi_failed):
        # Plot Temperature map obtained with Jacobi method
        filename = f'./Figures/cmaps/polar_T_map_Jacobi_{N}x{M}.png'
        title = 'Temperature distribution - Jacobi ($T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_distrib, theta_vals_distrib, solution=T_0 + (T_1 - T_0) * jacobi_sol,
                            filename=filename, title=title)

        # Plot Error map obtained with Jacobi method
        filename = f'./Figures/cmaps/polar_errT_map_Jacobi_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature error - Jacobi ($\Delta T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_ref, theta_vals_ref, solution=(T_1 - T_0) * jacobi_error,
                            filename=filename, title=title)

        # Non-dimensional plots
        # Plot Temperature map obtained with Jacobi method
        filename = f'./Figures/cmaps_adim/polar_T_map_Jacobi_{N}x{M}.png'
        title = 'Temperature non-dimensional distribution - Jacobi ($u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_distrib, theta_vals_distrib, solution=jacobi_sol, filename=filename, title=title)

        # Plot Error map obtained with Jacobi method
        filename = f'./Figures/cmaps_adim/polar_errT_map_Jacobi_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature non-dimensional error - Jacobi ($\Delta u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_ref, theta_vals_ref, solution=jacobi_error, filename=filename, title=title)

    if use_GS & (not gs_failed):
        # Plot Temperature map obtained with GS method
        filename = f'./Figures/cmaps/polar_T_map_GS_{N}x{M}.png'
        title = 'Temperature distribution - GS ($T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_distrib, theta_vals_distrib, solution=T_0 + (T_1 - T_0) * gs_sol,
                            filename=filename, title=title)

        # Plot Error map obtained with GS method
        filename = f'./Figures/cmaps/polar_errT_map_GS_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature error - GS ($\Delta T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_ref, theta_vals_ref, solution=(T_1 - T_0) * gs_error, filename=filename,
                            title=title)

        # Non-dimensional plots
        # Plot Temperature map obtained with GS method
        filename = f'./Figures/cmaps_adim/polar_T_map_GS_{N}x{M}.png'
        title = 'Temperature non-dimensional distribution - GS ($u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_distrib, theta_vals_distrib, solution=gs_sol, filename=filename, title=title)

        # Plot Error map obtained with GS method
        filename = f'./Figures/cmaps_adim/polar_errT_map_GS_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature non-dimensional error - GS ($\Delta u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_ref, theta_vals_ref, solution=gs_error, filename=filename, title=title)

    if use_SOR & (not sor_failed):
        # Plot Temperature map obtained with SOR method
        filename = f'./Figures/cmaps/polar_T_map_SOR_{N}x{M}.png'
        title = 'Temperature distribution - SOR ($T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_distrib, theta_vals_distrib, solution=T_0 + (T_1 - T_0) * sor_sol,
                            filename=filename, title=title)

        # Plot Error map obtained with SOR method
        filename = f'./Figures/cmaps/polar_errT_map_SOR_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature error - SOR ($\Delta T(r,\\theta)$)'
        plot_polar_colormap(radius * rho_vals_ref, theta_vals_ref, solution=(T_1 - T_0) * sor_error, filename=filename,
                            title=title)

        # Non-dimensional plots
        # Plot Temperature map obtained with SOR method
        filename = f'./Figures/cmaps_adim/polar_T_map_SOR_{N}x{M}.png'
        title = 'Temperature non-dimensional distribution - SOR ($u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_distrib, theta_vals_distrib, solution=sor_sol, filename=filename, title=title)

        # Plot Error map obtained with SOR method
        filename = f'./Figures/cmaps_adim/polar_errT_map_SOR_{N}x{M}_vs_{N_latex}x{M_latex}.png'
        title = 'Temperature non-dimensional error - SOR ($\Delta u(\\rho,\\theta)$)'
        plot_polar_colormap(rho_vals_ref, theta_vals_ref, solution=sor_error, filename=filename, title=title)


