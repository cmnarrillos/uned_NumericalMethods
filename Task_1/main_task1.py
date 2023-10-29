import os
import time

import numpy as np
import matplotlib.pyplot as plt

from task_1_functions import beam_momentum_ode, shooting_method
from task_1_functions import beam_momentum_and_deformation_ode
from task_1_functions import p_beam, q_beam, r_beam, finite_diff_order2
from aux_functions import solve_linear_system_with_lu_decomposition


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')

# Specify the blocks to be run
shooting_ref = False
finite_diff_ref = False
difference_btw_refs = False
shooting_convergence = False
finite_diff_convergence = False
shooting_deform = True

# REFERENCE FOR METHODS
print()
print('#-----------------------------------------')
print(' Compute reference to be used in comparisons')
print('#-----------------------------------------')

# Shooting method REF
if shooting_ref:
    x_bc = np.array([-1., 1.])
    y_bc = np.array([[0., 0.], [0., 0.]])
    is_bc = np.array([[True, True], [False, False]])
    N_ref_sh = 1000000
    # N_ref_sh = 10000
    params = {}

    t_init = time.time()
    y0_ref_sh, x_ref_sh, y_ref_sh = shooting_method(beam_momentum_ode, x_bc, y_bc, is_bc, 2*N_ref_sh, params, tol=1e-15)
    t_exe_ref_sh = time.time() - t_init

    print(f' -> Shooting method (N=1e6)')
    print(f'      Reference Initial Condition: y0 = {y0_ref_sh} (N={N_ref_sh})')
    print(f'      Maximum momentum at the beam: M_max = {np.max(np.abs(y_ref_sh[:, 0]))}'
          f' * p_0 * L^2 (y_max = {np.max(np.abs(y_ref_sh[:, 0]))})')
    print(f'      Computing time: {t_exe_ref_sh} s')

# Finite differences REF
if finite_diff_ref:
    x_bc = (-1, 1)
    y_bc = (0, 0)
    N_ref_fd = 10000
    # N_ref_fd = 1000

    t_init = time.time()
    A, b = finite_diff_order2(p=p_beam, q=q_beam, r=r_beam, x_bc=x_bc, y_bc=y_bc, n=2*N_ref_fd)

    y_intermediate = solve_linear_system_with_lu_decomposition(A, b)
    x_ref_fd = np.linspace(-1, 1, 2*N_ref_fd+1)
    y_ref_fd = np.concatenate(([y_bc[0]], y_intermediate, [y_bc[-1]]))

    t_exe_ref_fd = time.time() - t_init

    print()
    print(f' -> Finite differences (N=1e4)')
    print(f'      Maximum momentum at the beam: M_max = {np.max(np.abs(y_ref_fd))} '
          f'* p_0 * L^2 (y_max = {np.max(np.abs(y_ref_fd))})')
    print(f'      Computing time: {t_exe_ref_fd} s')

# Difference between methods
if difference_btw_refs:
    delta_y_methods = np.abs(y_ref_sh[::N_ref_sh//N_ref_fd, 0] - y_ref_fd)
    print()
    print(f' -> Difference between 2 methods:')
    print(f'      Maximum difference: {delta_y_methods.max()}')


# Study of convergence of shooting method
if shooting_convergence:
    print()
    print('#-----------------------------------------')
    print('  Start study on shooting method')
    print('#-----------------------------------------')

    # Initialize params and BCs structures
    x_bc = np.array([-1., 1.])
    y_bc = np.array([[0., 0.], [0., 0.]])
    is_bc = np.array([[True, True], [False, False]])
    params = {}
    n_list = [10, 20, 40, 80, 100, 200, 400, 800, 1000, 2000, 4000, 8000, 10000,
              20000, 40000, 80000, 100000, 200000, 400000]
    # n_list = [1000, 100, 10]

    # Initialize lists to store results
    t_exe_list = []
    x_list = []
    y_list = []
    x_eps_list = []
    eps_list = []
    max_eps = []
    n_eps_list = []

    # Loop over different discretizations
    for nn in n_list:

        t_init = time.time()

        # Compute solution with shooting method
        y0, x, y = shooting_method(beam_momentum_ode, x_bc, y_bc, is_bc, 2*nn, params, tol=1e-15)

        # Compute and store t exe
        t_exe = time.time() - t_init
        t_exe_list.append(t_exe)

        print(f' -> N={nn}')
        print(f'      Initial Condition: y0 = {y0}')
        print(f'      Maximum momentum at the beam: M_max = {np.max(np.abs(y[:, 0]))}'
              f' * p_0 * L^2 (y_max = {np.max(np.abs(y[:, 0]))})')
        print(f'      Computing time: {t_exe} s')

        if nn < 100:
            format_ = 'r+-'
        else:
            format_ = 'r-'

        # Plot the moment superimposed over the reference computed
        if nn < N_ref_sh:
            y_plot = y[:, 0]
            plt.figure()
            plt.plot(x, y_plot, format_, label=f'N={nn}')
            plt.plot(x_ref_sh, y_ref_sh[:, 0], 'k-', label=f'REF (N={N_ref_sh})')
            plt.xlabel('$ x=\\xi/L $')
            plt.ylabel('$ y=M/(p_0 L^2) $')
            plt.xlim(min(x), max(x))
            plt.ylim(min(y_plot)*1.1 - 0.1, max(y_plot)*1.1 + 0.1)
            plt.grid(which='both')
            plt.legend()
            plt.savefig(f'./Figures/shooting_N_{nn}_y.png', bbox_inches='tight')
            plt.close()

        x_list.append(x)
        y_list.append(y[:, 0])

        # Get the difference between the computed one and ref
        # Extract matching data
        if (nn % N_ref_sh) & (N_ref_sh % nn):
            continue  # Skip following part if the vectors cannot be matched
        if nn < N_ref_sh:
            step = N_ref_sh//nn
            x_eps = x
            eps = np.abs(y[:, 0] - y_ref_sh[::step, 0])
        else:
            step = nn//N_ref_sh
            x_eps = x_ref_sh
            eps = np.abs(y[::step, 0] - y_ref_sh[:, 0])
        print(f'    Difference wrt reference:')
        print(f'      Maximum difference: {eps.max()}')

        # Plot the difference between computed and ref
        if nn < N_ref_sh:
            plt.figure()
            plt.semilogy(x_eps, eps, format_, label=f'N={nn}')
            plt.xlabel('$ x=\\xi/L $')
            plt.ylabel('$ \\Delta y $')
            plt.xlim(min(x), max(x))
            plt.ylim(min(eps[1:])*0.1, max(eps)*10)
            plt.grid(which='both')
            plt.legend()
            plt.savefig(f'./Figures/shooting_N_{nn}_err.png', bbox_inches='tight')
            plt.close()

        x_eps_list.append(x_eps)
        eps_list.append(eps)
        n_eps_list.append(nn)
        max_eps.append(eps.max())

    # Plot momentum from different discretizations computed
    plt.figure()
    for ii, nn in enumerate(n_list):
        if nn < 100:
            format_ = '+-'
        else:
            format_ = '-'
        plt.plot(x_list[ii], y_list[ii], format_, label=f'N={nn}')
    plt.plot(x_ref_sh, y_ref_sh[:, 0], 'k-', label=f'REF (N={N_ref_sh})')
    plt.xlabel('$ x=\\xi/L $')
    plt.ylabel('$ y=M/(p_0 L^2) $')
    plt.xlim(min(x_ref_sh), max(x_ref_sh))
    plt.ylim(min(y_ref_sh[:, 0])*1.1 - 0.1, max(y_ref_sh[:, 0])*1.1 + 0.1)
    plt.grid(which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./Figures/shooting_all_y.png', bbox_inches='tight')
    plt.close()

    # Plot error from different sizes computed
    plt.figure()
    for ii, nn in enumerate(n_eps_list):
        if nn < 100:
            format_ = '+-'
        else:
            format_ = '-'
        plt.semilogy(x_eps_list[ii], eps_list[ii], format_, label=f'N={nn}')
    plt.xlabel('$ x=\\xi/L $')
    plt.ylabel('$ \\Delta y $')
    plt.xlim(min(x_eps_list[-1]), max(x_eps_list[-1]))
    plt.ylim(min(eps_list[-1][1:])*0.1, max(eps_list[0])*10)
    plt.grid(which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./Figures/shooting_all_err.png', bbox_inches='tight')
    plt.close()

    # Plot execution time vs N
    n_list.append(N_ref_sh)
    t_exe_list.append(t_exe_ref_sh)

    plt.figure()
    plt.loglog(n_list, t_exe_list, 'k+-')
    plt.xlabel('$ N $')
    plt.ylabel('$ t_{exe} $')
    plt.xlim(min(n_list)*0.3, max(n_list)*3)
    plt.ylim(min(t_exe_list)*0.1, max(t_exe_list)*10)
    plt.grid(which='both')
    plt.savefig(f'./Figures/shooting_t_exe.png', bbox_inches='tight')
    plt.close()

    # Plot max_error vs N
    plt.figure()
    plt.loglog(n_eps_list, max_eps, 'k+-')
    plt.xlabel('$ N $')
    plt.ylabel('$ max(\Delta y) $')
    plt.xlim(min(n_eps_list)*0.3, max(n_eps_list)*3)
    plt.ylim(min(max_eps)*0.1, max(max_eps)*10)
    plt.grid(which='both')
    plt.savefig(f'./Figures/shooting_max_err.png', bbox_inches='tight')
    plt.close()

    # Plot max error vs execution time
    t_exe_eps_list = [t_exe_list[n_list.index(n)] for n in n_eps_list if n in n_list]
    plt.figure()
    plt.loglog(t_exe_eps_list, max_eps, 'k+-')
    plt.xlabel('$ t_{exe}[s] $')
    plt.ylabel('$ max(\Delta y) $')
    plt.xlim(min(t_exe_eps_list)*0.3, max(t_exe_eps_list)*3)
    plt.ylim(min(max_eps)*0.1, max(max_eps)*10)
    plt.grid(which='both')
    plt.savefig(f'./Figures/shooting_max_err_vs_t_exe.png', bbox_inches='tight')
    plt.close()


# Study of convergence of Finite Differences method
if finite_diff_convergence:
    print()
    print('#-----------------------------------------')
    print('  Start study on finite diferences method')
    print('#-----------------------------------------')

    # Initialize params and BCs structures
    x_bc = (-1, 1)
    y_bc = (0, 0)
    n_list = [5, 10, 20, 40, 50, 100, 200, 400, 500, 1000, 2000, 4000, 5000]
    # n_list = [5, 10, 20, 40, 50, 100, 200, 400, 500]

    # Initialize lists to store results
    t_exe_list = []
    x_list = []
    y_list = []
    x_eps_list = []
    eps_list = []
    max_eps = []
    n_eps_list = []

    # Loop over different discretizations
    for nn in n_list:

        t_init = time.time()

        # Compute solution with shooting method
        A, b = finite_diff_order2(p=p_beam, q=q_beam, r=r_beam, x_bc=x_bc, y_bc=y_bc, n=2*nn)

        y_intermediate = solve_linear_system_with_lu_decomposition(A, b)
        x = np.linspace(-1, 1, 2*nn + 1)
        y = np.concatenate(([y_bc[0]], y_intermediate, [y_bc[-1]]))

        # Compute and store t exe
        t_exe = time.time() - t_init
        t_exe_list.append(t_exe)

        print(f' -> N={nn}')
        print(f'      Maximum momentum at the beam: M_max = {np.max(np.abs(y))}'
              f' * p_0 * L^2 (y_max = {np.max(np.abs(y))})')
        print(f'      Computing time: {t_exe} s')

        if nn < 100:
            format_ = 'r+-'
        else:
            format_ = 'r-'

        # Plot the moment superimposed over the reference computed
        if nn < N_ref_fd:
            plt.figure()
            plt.plot(x, y, format_, label=f'N={nn}')
            plt.plot(x_ref_fd, y_ref_fd, 'k-', label=f'REF (N={N_ref_fd})')
            plt.xlabel('$ x=\\xi/L $', fontsize=14)
            plt.ylabel('$ y=M/(p_0 L^2) $', fontsize=14)
            plt.xlim(min(x), max(x))
            plt.ylim(min(y)*1.1 - 0.1, max(y)*1.1 + 0.1)
            plt.grid(which='both')
            plt.legend()
            plt.savefig(f'./Figures/finite_diff_N_{nn}_y.png', bbox_inches='tight')
            plt.close()

        x_list.append(x)
        y_list.append(y)

        # Get the difference between the computed one and ref
        # Extract matching data
        if (nn % N_ref_fd) & (N_ref_fd % nn):
            continue  # Skip following part if the vectors cannot be matched
        if nn < N_ref_fd:
            step = N_ref_fd//nn
            x_eps = x
            eps = np.abs(y - y_ref_fd[::step])
        else:
            step = nn//N_ref_fd
            x_eps = x_ref_fd
            eps = np.abs(y[::step] - y_ref_fd)
        print(f'    Difference wrt reference:')
        print(f'      Maximum difference: {eps.max()}')

        # Plot the difference between computed and ref
        if nn < N_ref_fd:
            plt.figure()
            plt.semilogy(x_eps, eps, format_, label=f'N={nn}')
            plt.xlabel('$ x=\\xi/L $', fontsize=14)
            plt.ylabel('$ \\Delta y $', fontsize=14)
            plt.xlim(min(x), max(x))
            plt.ylim(min(eps[1:])*0.1, max(eps)*10)
            plt.grid(which='both')
            plt.legend()
            plt.savefig(f'./Figures/finite_diff_N_{nn}_err.png', bbox_inches='tight')
            plt.close()

        x_eps_list.append(x_eps)
        eps_list.append(eps)
        n_eps_list.append(nn)
        max_eps.append(eps.max())

    # Plot momentum from different discretizations computed
    plt.figure()
    for ii, nn in enumerate(n_list):
        if nn < 100:
            format_ = '+-'
        else:
            format_ = '-'
        plt.plot(x_list[ii], y_list[ii], format_, label=f'N={nn}')
    plt.plot(x_ref_fd, y_ref_fd, 'k-', label=f'REF (N={N_ref_fd})')
    plt.xlabel('$ x=\\xi/L $', fontsize=14)
    plt.ylabel('$ y=M/(p_0 L^2) $', fontsize=14)
    plt.xlim(min(x_ref_fd), max(x_ref_fd))
    plt.ylim(min(y_ref_fd)*1.1 - 0.1, max(y_ref_fd)*1.1 + 0.1)
    plt.grid(which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./Figures/finite_diff_all_y.png', bbox_inches='tight')
    plt.close()

    # Plot error from different sizes computed
    plt.figure()
    for ii, nn in enumerate(n_eps_list):
        if nn < 100:
            format_ = '+-'
        else:
            format_ = '-'
        plt.semilogy(x_eps_list[ii], eps_list[ii], format_, label=f'N={nn}')
    plt.xlabel('$ x=\\xi/L $', fontsize=14)
    plt.ylabel('$ \\Delta y $', fontsize=14)
    plt.xlim(min(x_eps_list[-1]), max(x_eps_list[-1]))
    plt.ylim(min(eps_list[-1][1:])*0.1, max(eps_list[0])*10)
    plt.grid(which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./Figures/finite_diff_all_err.png', bbox_inches='tight')
    plt.close()

    # Plot execution time vs N
    n_list.append(N_ref_fd)
    t_exe_list.append(t_exe_ref_fd)

    plt.figure()
    plt.loglog(n_list, t_exe_list, 'k+-')
    plt.xlabel('$ N $', fontsize=14)
    plt.ylabel('$ t_{exe} $', fontsize=14)
    plt.xlim(min(n_list)*0.3, max(n_list)*3)
    plt.ylim(min(t_exe_list)*0.1, max(t_exe_list)*10)
    plt.grid(which='both')
    plt.savefig(f'./Figures/finite_diff_t_exe.png', bbox_inches='tight')
    plt.close()

    # Plot max error vs N
    plt.figure()
    plt.loglog(n_eps_list, max_eps, 'k+-')
    plt.xlabel('$ N $', fontsize=14)
    plt.ylabel('$ max(\Delta y) $', fontsize=14)
    plt.xlim(min(n_eps_list)*0.3, max(n_eps_list)*3)
    plt.ylim(min(max_eps)*0.1, max(max_eps)*10)
    plt.grid(which='both')
    plt.savefig(f'./Figures/finite_diff_max_err.png', bbox_inches='tight')
    plt.close()

    # Plot max error vs execution time
    t_exe_eps_list = [t_exe_list[n_list.index(n)] for n in n_eps_list if n in n_list]
    plt.figure()
    plt.loglog(t_exe_eps_list, max_eps, 'k+-')
    plt.xlabel('$ t_{exe}[s] $', fontsize=14)
    plt.ylabel('$ max(\Delta y) $', fontsize=14)
    plt.xlim(min(t_exe_eps_list)*0.3, max(t_exe_eps_list)*3)
    plt.ylim(min(max_eps)*0.1, max(max_eps)*10)
    plt.grid(which='both')
    plt.savefig(f'./Figures/finite_diff_max_err_vs_t_exe.png', bbox_inches='tight')
    plt.close()


# Shooting method for deformation using Euler-Bernouilli beam model
if shooting_deform:
    print()
    print('#-----------------------------------------')
    print('  Propagate the 4th order equation to solve')
    print('  deformation using shooting method')
    print('#-----------------------------------------')
    x_bc = np.array([-1., 1.])
    y_bc = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
    is_bc = np.array([[True, True], [False, False], [True, True], [False, False]])
    N = 1000
    factor_list = [0.1, 0.5, 1, 1.1, 1.15]
    y_list = []
    v_list = []

    for lf in factor_list:
        params = {
            'load_factor': lf
        }

        print(f'   -> Load factor: p_0 * L / P = {params["load_factor"]}')
        t_init = time.time()
        try:
            y0, x, y = shooting_method(beam_momentum_and_deformation_ode, x_bc, y_bc, is_bc, 2*N, params, tol=1e-12)
        except:
            y0, x, y = shooting_method(beam_momentum_and_deformation_ode, x_bc, y_bc, is_bc, 2*N, params)
            print('      Tolerance increased to 1e-6')
        t_exe = time.time() - t_init

        print(f'      Initial condition: {y0}')
        print(f'      Maximum momentum at the beam: M_max = {np.max(np.abs(y[:, 0]))}'
              f' * p_0 * L^2 (y_max = {np.max(np.abs(y[:, 0]))})')
        print(f'      Maximum deformation at the beam: w_max = {np.max(np.abs(y[:, 2]))}'
              f' * L (v_max = {np.max(np.abs(y[:, 2]))})')
        print(f'      Computing time: {t_exe} s')
        plt.figure()
        y_plot = y[:, 0]
        plt.plot(x, y_plot, label=f'LF={lf}')
        plt.xlabel('$ x=\\xi/L $', fontsize=14)
        plt.ylabel('$ y=M/(p_0 L^2) $', fontsize=14)
        plt.xlim(min(x), max(x))
        plt.ylim(min(y_plot)*1.1 - 0.1, max(y_plot)*1.1 + 0.1)
        plt.grid(which='both')
        plt.savefig(f'./Figures/test_shooting_lf_{lf}_N_{N}_momentum.png', bbox_inches='tight')
        plt.close()
        y_list.append(y_plot)

        plt.figure()
        v_plot = y[:, 2]
        plt.plot(x, v_plot, label=f'LF={lf}')
        plt.xlabel('$ x=\\xi/L $', fontsize=14)
        plt.ylabel('$ v=w/L $', fontsize=14)
        plt.xlim(min(x), max(x))
        plt.ylim(min(v_plot)*1.1 - 0.1, max(v_plot)*1.1 + 0.1)
        plt.grid(which='both')
        plt.savefig(f'./Figures/test_shooting_lf_{lf}_N_{N}_deformation.png', bbox_inches='tight')
        plt.close()
        v_list.append(v_plot)

    plt.figure()
    for ii, lf in enumerate(factor_list):
        plt.plot(x, y_list[ii], label=f'LF={lf}')
    plt.xlabel('$ x=\\xi/L $', fontsize=14)
    plt.ylabel('$ y=M/(p_0 L^2) $', fontsize=14)
    plt.xlim(min(x), max(x))
    plt.ylim(min(y_list[-1])*1.1 - 0.1, max(y_list[-1])*1.1 + 0.1)
    plt.grid(which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./Figures/test_shooting_lf_all_N_{N}_momentum.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    for ii, lf in enumerate(factor_list):
        plt.plot(x, v_list[ii], label=f'LF={lf}')
    plt.xlabel('$ x=\\xi/L $', fontsize=14)
    plt.ylabel('$ v=w/L $', fontsize=14)
    plt.xlim(min(x), max(x))
    plt.ylim(min(v_list[-1])*1.1 - 0.1, max(v_list[-1])*1.1 + 0.1)
    plt.grid(which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./Figures/test_shooting_lf_all_N_{N}_deformation.png', bbox_inches='tight')
    plt.close()
