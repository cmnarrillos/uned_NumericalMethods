import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_4_functions import iterative_pde_solver, theta_method_neutron, prog_diff_method
try:
    from aux_functions import dirac_delta, integrate_trapezoidal
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import dirac_delta, integrate_trapezoidal


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')


# Initialize parameters of the method
L = 1
D = 1
C = 0

# Discretization
n_x = 99
d_t = 0.005
n_t = 200
t = np.linspace(0, (n_t+1)*d_t, n_t+1)

# Boundary conditions
u_BC = np.array([0, 0])

# Initial condition
x = np.linspace(-L/2, L/2, n_x+2)
d_x = L/(n_x+1)
u_0 = np.array([dirac_delta(x_val, tol=d_x) for x_val in x])


# Propagate evolution of the system

# Progressive differences
# Build the matrices involved
A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=0)

# Propagate
tinit = time.time()
u_pd, u_pd_evolution = prog_diff_method(u_0=u_0, u_BC=u_BC, B=B, D_rhs=D_rhs, n_t=n_t)
print(f'Direct progresive differences propagated in {time.time() - tinit} s')

# Get avg neutron density
neutron_avg_pd = np.array([integrate_trapezoidal(x, u_step, lims=(-L/2, L/2)) for u_step in u_pd_evolution])


# Regressive differences
# Build the matrices involved
A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=1)

# Propagate
tinit = time.time()
u_rd, u_rd_evolution = iterative_pde_solver(u_0=u_0, u_BC=u_BC, A=A, B=B, D_rhs=D_rhs, D_lhs=D_lhs, n_t=n_t)
print(f'Regressive differences propagated in {time.time() - tinit} s')

# Get avg neutron density
neutron_avg_rd = np.array([integrate_trapezoidal(x, u_step, lims=(-L/2, L/2)) for u_step in u_rd_evolution])


# Crank-Nicholson differences
# Build the matrices involved
A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=0.5)

# Propagate
tinit = time.time()
u_cn, u_cn_evolution = iterative_pde_solver(u_0=u_0, u_BC=u_BC, A=A, B=B, D_rhs=D_rhs, D_lhs=D_lhs, n_t=n_t)
print(f'Progresive differences propagated in {time.time() - tinit} s')

# Get avg neutron density
neutron_avg_cn = np.array([integrate_trapezoidal(x, u_step, lims=(-L/2, L/2)) for u_step in u_cn_evolution])


# Theoretical neutron density
neutron_avg_th = np.exp((C - D*np.pi**2/L**2)*t)
neutron_avg_th_adj = neutron_avg_cn[-1] / neutron_avg_th[-1] * neutron_avg_th

# Plot neutron density
plt.figure()
if not np.isnan(neutron_avg_pd).any() and \
        not (np.max(neutron_avg_pd) > 1e3*np.max(neutron_avg_th)):
    plt.semilogy(t, neutron_avg_pd, 'r-x', linewidth=0.75, markersize=1.5, label='Prog Diff')
plt.semilogy(t, neutron_avg_rd, 'b-x', linewidth=0.75, markersize=1.5, label='Reg Diff')
plt.semilogy(t, neutron_avg_cn, 'g-x', linewidth=0.75, markersize=1.5, label='C-N')
plt.semilogy(t, neutron_avg_th, 'k--', linewidth=2, label='Theoretical')
plt.semilogy(t, neutron_avg_th_adj, 'k-.', linewidth=1, label='Th adjusted')
plt.xlabel('$t(s)$', fontsize=18)
plt.xlim((0, t[-1]))
plt.ylabel('$\overline{n}$', fontsize=18)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./Figures/log_neutron_density_C_{C}_dx_{d_x}_dt_{d_t}.png', bbox_inches='tight')
plt.yscale('linear')
plt.ylim(0)
plt.savefig(f'./Figures/neutron_density_C_{C}_dx_{d_x}_dt_{d_t}.png', bbox_inches='tight')
