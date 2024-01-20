import os

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

if neumann_cond:
    fig_path = 'Figures'
else:
    fig_path = 'Figures/Dirichlet_both'
    if not os.path.exists('./Figures/Dirichlet_both/'):
        os.makedirs('./Figures/Dirichlet_both/')

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
    # Create generic system for the function given by the rhs and alpha^2
    A_out, F_out, x = finite_elements(alpha2=alpha2, f=rhs,
                                      num_elem=nelem, order=order,
                                      lims=domain, analytical=False)

    if neumann_cond:
        # Apply Dirichlet boundary condition u(x=0) = u_0 = 0
        A_syst = A_out[1:, 1:]
        F_syst = F_out[1:, 0] - A_out[1:, 0] * u_0
        # Apply Neumann boundary condition du/dx(x=1) = du_dx_N = 0
        A_syst[-1, -1] += du_dx_N
    else:
        # Apply 2 Dirichlet boundary conditions u(x=0) = u_0 = 0; u(x=1) = u_N
        A_syst = A_out[1:-1, 1:-1]
        F_syst = F_out[1:-1, 0] - A_out[1:-1, 0] * u_0 - A_out[1:-1, -1] * u_N

    # Solve the reduced system
    u_syst = solve_linear_system_with_lu_decomposition(A_syst, F_syst)

    # Cast system result and boundary conditions
    u_sol = np.zeros(F_out.shape[0])
    if neumann_cond:
        u_sol[0] = u_0
        u_sol[1:] = u_syst
    else:
        u_sol[0] = u_0
        u_sol[1:-1] = u_syst
        u_sol[-1] = u_N

    # Compute error wrt analytical solution
    err = analytic_sol(x) - u_sol[:]

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
