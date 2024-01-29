import os
import time

import numpy as np
import matplotlib.pyplot as plt

from task_5_functions import finite_elements, rhs, analytic_sol

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import solve_linear_system_with_lu_decomposition
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import solve_linear_system_with_lu_decomposition


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')

# Define the problem to solve
u_0 = 0
u_N = np.tan(1) - 1
du_dx_N = 0
domain = (0, 1)
alpha2 = 1.
neumann_cond = True  # Allows to switch between BC types
convergence = True
if convergence:
    num_elems_convergence = [1, 2, 3, 5, 10, 15, 20, 30, 50, 80, 100, 150, 200,
                             300, 500, 800, 1000, 1500, 2000, 3000, 5000]
    if not neumann_cond:
        num_elems_convergence.pop(0)

if neumann_cond:
    fig_path = 'Figures'
else:
    fig_path = 'Figures/Dirichlet_both'
    if not os.path.exists('./Figures/Dirichlet_both/'):
        os.makedirs('./Figures/Dirichlet_both/')

file = open(f'./{fig_path}/tables.txt', 'w')

# Compute analytical solution for plots
x_analytic = np.linspace(domain[0], domain[-1], 1001)
u_analytic = analytic_sol(x_analytic)

# Create lists of configurations
nelem_list = [4, 8, 2, 4]
order_list = [1, 1, 2, 2]

# Initialize variables to store results
x_store = []
u_store = []
err_store = []

for nelem, order in zip(nelem_list, order_list):
    file.write('\n')
    file.write(f'{nelem} Elements, Order {order}\n')
    file.write('\n\n')
    # Create generic system for the function given by the rhs and alpha^2
    A_out, F_out, x = finite_elements(alpha2=alpha2, f=rhs,
                                      num_elem=nelem, order=order,
                                      lims=domain, analytical=True)

    if neumann_cond:
        # Apply Dirichlet boundary condition u(x=0) = u_0 = 0
        A_syst = A_out[1:, 1:]
        F_syst = F_out[1:] - A_out[1:, 0] * u_0
        # Apply Neumann boundary condition du/dx(x=1) = du_dx_N = 0
        A_syst[-1, -1] += du_dx_N
    else:
        # Apply 2 Dirichlet boundary conditions u(x=0) = u_0 = 0; u(x=1) = u_N
        A_syst = A_out[1:-1, 1:-1]
        F_syst = F_out[1:-1] - A_out[1:-1, 0] * u_0 - A_out[1:-1, -1] * u_N

    # Solve the reduced system
    u_syst = solve_linear_system_with_lu_decomposition(A_syst, F_syst)

    # Cast system result and boundary conditions
    u_sol = np.zeros(F_out.shape)
    if neumann_cond:
        u_sol[0] = u_0
        u_sol[1:] = u_syst
    else:
        u_sol[0] = u_0
        u_sol[1:-1] = u_syst
        u_sol[-1] = u_N

    # Compute error wrt analytical solution
    u_ref = analytic_sol(x)
    err = u_ref - u_sol[:]
    file.write('\\begin{tabular}{|c||c|c|c|}\n')
    file.write('\hline\n')
    file.write('$x$ & $u_{ref}(x)$ & $u_{FE}(x)$ & $\\varepsilon(x)$ \\\\\n')
    file.write('\hline\hline\n')
    for ii in range(x.shape[0]):
        file.write(f'{x[ii]} & {round(u_ref[ii], 6)} & '
                   f'{round(u_sol[ii], 6)} & {round(err[ii], 6)} \\\\\n')
        file.write('\hline\n')
    file.write('\\end{tabular}\n')
    file.write('\n\n\n\n\n')


    x_store.append(x)
    u_store.append(u_sol)
    err_store.append(err)

    # Plot results vs analytical solution
    plt.figure(figsize=(12, 8))
    plt.plot(x_analytic, u_analytic, '-k', linewidth=2.5, label='analytical')
    plt.plot(x, u_sol, '--xr', markersize=8, linewidth=1, label='FEM')
    plt.xlim(domain)
    plt.ylim(0)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$u(x)$', fontsize=14)
    plt.title(f'FEM - {nelem} elements, order {order}', fontsize=18)
    plt.grid(which='both')
    plt.legend(fontsize=14)
    plt.savefig(f'./{fig_path}/order_{order}_nelem_{nelem}_sol.png',
                bbox_inches='tight')
    plt.close()

    # Plot error
    plt.figure(figsize=(12, 8))
    plt.plot(x, err, '--xr', markersize=8, linewidth=1, label='error')
    plt.xlim(domain)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
    plt.title(f'FEM - {nelem} elements, order {order}', fontsize=18)
    plt.grid(which='both')
    plt.legend(fontsize=14)
    plt.savefig(f'./{fig_path}/order_{order}_nelem_{nelem}_err.png',
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.semilogy(x, np.abs(err), '--xr', markersize=8, linewidth=1,
                 label='|error|')
    plt.xlim(domain)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
    plt.title(f'FEM - {nelem} elements, order {order}', fontsize=18)
    plt.grid(which='both')
    plt.legend(fontsize=14)
    plt.savefig(f'./{fig_path}/order_{order}_nelem_{nelem}_err_log.png',
                bbox_inches='tight')
    plt.close()


# Plot all the results together

plt.figure(figsize=(12, 8))
plt.plot(x_analytic, u_analytic, '-k', linewidth=2.5, label='analytical')
ii = 0
for nelem, order in zip(nelem_list, order_list):
    plt.plot(x_store[ii], u_store[ii], '--x', markersize=8, linewidth=1,
             label=f'FEM - {nelem} elements, order {order}')
    ii += 1
plt.xlim(domain)
plt.ylim(0)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$u(x)$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/all_cases_sol.png',
            bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 8))
ii = 0
for nelem, order in zip(nelem_list, order_list):
    plt.plot(x_store[ii], err_store[ii], '--x', markersize=8, linewidth=1,
             label=f'FEM - {nelem} elements, order {order}')
    ii += 1
plt.xlim(domain)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/all_cases_err.png',
            bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 8))
ii = 0
for nelem, order in zip(nelem_list, order_list):
    plt.semilogy(x_store[ii], err_store[ii], '--x', markersize=8, linewidth=1,
                 label=f'FEM - {nelem} elements, order {order}')
    ii += 1
plt.xlim(domain)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/all_cases_err_log.png',
            bbox_inches='tight')
plt.close()


# Analyze convergence
h_o1 = []
h_o2 = []
num_nodes_o1 = []
num_nodes_o2 = []
t_exe_o1 = []
t_exe_o2 =[]
avg_err_o1 = []
avg_err_o2 = []
avg_abs_err_o1 = []
avg_abs_err_o2 = []
max_err_o1 = []
max_err_o2 = []

if convergence:
    for nelem in num_elems_convergence:
        for order in [1, 2]:
            # Create generic system for the function given by the rhs and alpha^2
            tinit = time.time()
            A_out, F_out, x = finite_elements(alpha2=alpha2, f=rhs,
                                              num_elem=nelem, order=order,
                                              lims=domain, analytical=True)

            if neumann_cond:
                # Apply Dirichlet boundary condition u(x=0) = u_0 = 0
                A_syst = A_out[1:, 1:]
                F_syst = F_out[1:] - A_out[1:, 0] * u_0
                # Apply Neumann boundary condition du/dx(x=1) = du_dx_N = 0
                A_syst[-1, -1] += du_dx_N
            else:
                # Apply 2 Dirichlet boundary conditions u(x=0) = u_0 = 0; u(x=1) = u_N
                A_syst = A_out[1:-1, 1:-1]
                F_syst = F_out[1:-1] - A_out[1:-1, 0] * u_0 - A_out[1:-1, -1] * u_N

            # Solve the reduced system
            u_syst = solve_linear_system_with_lu_decomposition(A_syst, F_syst)

            # Cast system result and boundary conditions
            u_sol = np.zeros(F_out.shape)
            if neumann_cond:
                u_sol[0] = u_0
                u_sol[1:] = u_syst
            else:
                u_sol[0] = u_0
                u_sol[1:-1] = u_syst
                u_sol[-1] = u_N

            # Compute error wrt analytical solution
            u_ref = analytic_sol(x)
            err = u_ref - u_sol[:]

            # Store results for plotting
            if order == 1:
                t_exe_o1.append(time.time() - tinit)
                num_nodes_o1.append(nelem + 1)
                h_o1.append((x[-1] - x[0])/nelem)
                avg_err_o1.append(np.mean(err))
                avg_abs_err_o1.append(np.mean(np.abs(err)))
                max_err_o1.append(np.max(np.abs(err)))
            elif order == 2:
                t_exe_o2.append(time.time() - tinit)
                num_nodes_o2.append(2*nelem + 1)
                h_o2.append((x[-1] - x[0])/nelem)
                avg_err_o2.append(np.mean(err))
                avg_abs_err_o2.append(np.mean(np.abs(err)))
                max_err_o2.append(np.max(np.abs(err)))


# Plot results for convergence
plt.figure(figsize=(12, 8))
plt.loglog(h_o1, avg_err_o1, 'x-r', markersize=8, linewidth=1, label='Order 1 (avg)')
plt.loglog(h_o1, max_err_o1, 's-.r', markersize=4, linewidth=1, label='Order 1 (max)')
plt.loglog(h_o2, avg_err_o2, 'x-b', markersize=8, linewidth=1, label='Order 2 (avg)')
plt.loglog(h_o2, max_err_o2, 's-.b', markersize=4, linewidth=1, label='Order 2 (max)')
# plt.xlim(domain)
plt.xlabel('$h$', fontsize=14)
plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/convergence_h.png',
            bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 8))
plt.loglog(num_nodes_o1, avg_err_o1, 'x-r', markersize=8, linewidth=1, label='Order 1 (avg)')
plt.loglog(num_nodes_o1, max_err_o1, 's-.r', markersize=4, linewidth=1, label='Order 1 (max)')
plt.loglog(num_nodes_o2, avg_err_o2, 'x-b', markersize=8, linewidth=1, label='Order 2 (avg)')
plt.loglog(num_nodes_o2, max_err_o2, 's-.b', markersize=4, linewidth=1, label='Order 2 (max)')
# plt.xlim(domain)
plt.xlabel('$Nodes$', fontsize=14)
plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/convergence_num_nodes.png',
            bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 8))
plt.loglog(num_elems_convergence, avg_err_o1, 'x-r', markersize=8, linewidth=1, label='Order 1 (avg)')
plt.loglog(num_elems_convergence, max_err_o1, 's-.r', markersize=4, linewidth=1, label='Order 1 (max)')
plt.loglog(num_elems_convergence, avg_err_o2, 'x-b', markersize=8, linewidth=1, label='Order 2 (avg)')
plt.loglog(num_elems_convergence, max_err_o2, 's-.b', markersize=4, linewidth=1, label='Order 2 (max)')
# plt.xlim(domain)
plt.xlabel('$Elements$', fontsize=14)
plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/convergence_num_elems.png',
            bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 8))
plt.loglog(t_exe_o1, avg_err_o1, 'x-r', markersize=8, linewidth=1, label='Order 1 (avg)')
plt.loglog(t_exe_o1, max_err_o1, 's-.r', markersize=4, linewidth=1, label='Order 1 (max)')
plt.loglog(t_exe_o2, avg_err_o2, 'x-b', markersize=8, linewidth=1, label='Order 2 (avg)')
plt.loglog(t_exe_o2, max_err_o2, 's-.b', markersize=4, linewidth=1, label='Order 2 (max)')
# plt.xlim(domain)
plt.xlabel('Execution time', fontsize=14)
plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/convergence_t_exe.png',
            bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 8))
plt.semilogy(t_exe_o1, avg_err_o1, 'x-r', markersize=8, linewidth=1, label='Order 1 (avg)')
plt.semilogy(t_exe_o1, max_err_o1, 's-.r', markersize=4, linewidth=1, label='Order 1 (max)')
plt.semilogy(t_exe_o2, avg_err_o2, 'x-b', markersize=8, linewidth=1, label='Order 2 (avg)')
plt.semilogy(t_exe_o2, max_err_o2, 's-.b', markersize=4, linewidth=1, label='Order 2 (max)')
# plt.xlim(domain)
plt.xlabel('Execution time', fontsize=14)
plt.ylabel('$\\varepsilon[u(x)]$', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/convergence_t_exe_semilog.png',
            bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 8))
plt.loglog(h_o1, t_exe_o1, 'x-r', markersize=8, linewidth=1, label='Order 1')
plt.loglog(h_o1, t_exe_o2, 'x-b', markersize=8, linewidth=1, label='Order 2')
# plt.xlim(domain)
plt.xlabel('$h$', fontsize=14)
plt.ylabel('Execution time', fontsize=14)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./{fig_path}/convergence_t_exe_vs_h.png',
            bbox_inches='tight')
plt.close()

file.close()
