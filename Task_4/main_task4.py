import os
import time
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from task_4_functions import iterative_pde_solver, theta_method_neutron, prog_diff_method, \
                             plot_cmap, plot_linear
try:
    from aux_functions import dirac_delta, integrate_trapezoidal
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import dirac_delta, integrate_trapezoidal

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')
if not os.path.exists(f'./Figures/{now}/'):
    os.makedirs(f'./Figures/{now}/')


# Initialize parameters of the method
L = 1
D = 1
C = 1

# Discretization
n_x = 19
d_t = 0.002
n_t = 500
t = np.linspace(0, n_t*d_t, n_t+1)

# Boundary conditions
u_BC = np.array([0, 0])

# Initial condition
x = np.linspace(-L/2, L/2, n_x+2)
d_x = L/(n_x+1)
u_0 = np.array([dirac_delta(x_val, tol=d_x) for x_val in x])

print()
print('Problem initialized')
print(f'L = {L}, D = {D}, C ={C}')
print(f'dx = {d_x}, dt = {d_t}')
print()

# Check stability conditions:
pd_stable = D*d_t/d_x**2 - C/2*d_t < 1/2
if C < 1e-7:
    rd_stable = True
    cn_stable = True
else:
    rd_stable = d_t < 1/C
    cn_stable = d_t < 2/C


# Propagate evolution of the system

# Progressive differences
if pd_stable:
    # Build the matrices involved
    A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=0)

    # Propagate
    tinit = time.time()
    u_pd, u_pd_evolution = prog_diff_method(u_0=u_0, u_BC=u_BC, B=B, D_rhs=D_rhs, n_t=n_t)
    print(f'Progresive differences propagated in {time.time() - tinit} s')

    # Get avg neutron density
    neutron_avg_pd = np.array([integrate_trapezoidal(x, u_step, lims=(-L/2, L/2)) for u_step in u_pd_evolution])
else:
    print('Progressive differences method cannot be applied for this relation of dt, dx')
    u_pd = None
    u_pd_evolution = np.array([None for _ in x])
    neutron_avg_pd = None


# Regressive differences
if rd_stable:
    # Build the matrices involved
    A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=1)

    # Propagate
    tinit = time.time()
    u_rd, u_rd_evolution = iterative_pde_solver(u_0=u_0, u_BC=u_BC, A=A, B=B, D_rhs=D_rhs, D_lhs=D_lhs, n_t=n_t)
    print(f'Regressive differences propagated in {time.time() - tinit} s')

    # Get avg neutron density
    neutron_avg_rd = np.array([integrate_trapezoidal(x, u_step, lims=(-L/2, L/2)) for u_step in u_rd_evolution])
else:
    print('Regressive differences method cannot be applied for this dt given param C of the problem')
    u_rd = None
    u_rd_evolution = np.array([None for _ in x])
    neutron_avg_rd = None


# Crank-Nicholson differences
if cn_stable:
    # Build the matrices involved
    A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=0.5)

    # Propagate
    tinit = time.time()
    u_cn, u_cn_evolution = iterative_pde_solver(u_0=u_0, u_BC=u_BC, A=A, B=B, D_rhs=D_rhs, D_lhs=D_lhs, n_t=n_t)
    print(f'Crank-Nicolson propagated in {time.time() - tinit} s')

    # Get avg neutron density
    neutron_avg_cn = np.array([integrate_trapezoidal(x, u_step, lims=(-L/2, L/2)) for u_step in u_cn_evolution])
else:
    print('Regressive differences method cannot be applied for this dt given param C of the problem')
    u_cn = None
    u_cn_evolution = np.array([None for _ in x])
    neutron_avg_cn = None


# Neutron density

# Theoretical neutron density
neutron_avg_th = np.exp((C - D*np.pi**2/L**2)*t)

if cn_stable:
    factor = neutron_avg_cn[-1] / neutron_avg_th[-1]
elif rd_stable:
    factor = neutron_avg_rd[-1] / neutron_avg_th[-1]
elif pd_stable:
    factor = neutron_avg_pd[-1] / neutron_avg_th[-1]
else:
    factor = 1
neutron_avg_th_adj = factor * neutron_avg_th
print()
print(f'Adjustment factor: {factor}')
print(f'Equivalent delay: {-np.log(factor)/(C - D*np.pi**2/L**2)}')


# Plot neutron density
plt.figure(figsize=(12, 8))
if pd_stable:
    plt.semilogy(t, neutron_avg_pd, 'r-x', linewidth=0.75, markersize=1.5, label='Prog Diff')
if rd_stable:
    plt.semilogy(t, neutron_avg_rd, 'b-x', linewidth=0.75, markersize=1.5, label='Reg Diff')
if cn_stable:
    plt.semilogy(t, neutron_avg_cn, 'g-x', linewidth=0.75, markersize=1.5, label='C-N')
plt.semilogy(t, neutron_avg_th, 'k--', linewidth=2, label='Trend')
plt.semilogy(t, neutron_avg_th_adj, 'k-.', linewidth=1, label='Tr. adjusted')
plt.xlim((0, t[-1]))
plt.xlabel('$t(s)$', fontsize=14)
plt.ylabel('$\\tilde{n}$', fontsize=14)
plt.title('Average neutron density', fontsize=18)
plt.grid(which='both')
plt.legend(fontsize=14)
plt.savefig(f'./Figures/{now}/log_neutron_density_C_{C}_dx_{d_x}_dt_{d_t}.png', bbox_inches='tight')
plt.yscale('linear')
plt.ylim(0)
plt.savefig(f'./Figures/{now}/neutron_density_C_{C}_dx_{d_x}_dt_{d_t}.png', bbox_inches='tight')


# # Plot color maps
domain = (-L/2 - d_x/2, L/2 + d_x/2, t[-1]+d_t/2, -d_t/2)

# Progressive Diff
if pd_stable:
    filename = f'./Figures/{now}/cmap_PD_{C}_dx_{d_x}_dt_{d_t}.png'
    plot_cmap(u_evol=u_pd_evolution, domain=domain, title='Progressive Differences', filename=filename,
              xlims=(-L/2, L/2), ylims=(0, t[-1]))


# Regressive Diff
if rd_stable:
    filename = f'./Figures/{now}/cmap_RD_{C}_dx_{d_x}_dt_{d_t}.png'
    plot_cmap(u_evol=u_rd_evolution, domain=domain, title='Regressive Differences', filename=filename,
              xlims=(-L/2, L/2), ylims=(0, t[-1]))

# Crank-Nicolson
if cn_stable:
    filename = f'./Figures/{now}/cmap_CN_{C}_dx_{d_x}_dt_{d_t}.png'
    plot_cmap(u_evol=u_rd_evolution, domain=domain, title='Crank-Nicolson', filename=filename,
              xlims=(-L/2, L/2), ylims=(0, t[-1]))


# Plot final state
filename = f'./Figures/{now}/solution_end_C_{C}_dx_{d_x}_dt_{d_t}.png'

plot_linear(x=x, y_pd=u_pd, y_rd=u_rd, y_cn=u_cn, title=f'Solution at t={t[-1]}', filename=filename,
            xlims=(-L/2, L/2), ylims=0, x_label='$x/L$', y_label='$n(x)$')


# Plot evolution at fixed x

# x = 0.1
idx = int((1 + 0.1 / (L/2)) * (n_x+1)/2)
filename = f'./Figures/{now}/neutron_evol_x1_C_{C}_dx_{d_x}_dt_{d_t}.png'

plot_linear(x=t, y_pd=u_pd_evolution.T[idx], y_rd=u_rd_evolution.T[idx], y_cn=u_cn_evolution.T[idx],
            title=f'Solution at x={round(x[idx],2)}', filename=filename,
            xlims=(0, t[-1]), ylims=0, x_label='$t$', y_label='$n(t)$')

# x = 0.2
idx = int((1 + 0.2 / (L/2)) * (n_x+1)/2)
filename = f'./Figures/{now}/neutron_evol_x2_C_{C}_dx_{d_x}_dt_{d_t}.png'

plot_linear(x=t, y_pd=u_pd_evolution.T[idx], y_rd=u_rd_evolution.T[idx], y_cn=u_cn_evolution.T[idx],
            title=f'Solution at x={round(x[idx],2)}', filename=filename,
            xlims=(0, t[-1]), ylims=0, x_label='$t$', y_label='$n(t)$')

# x = 0.3
idx = int((1 + 0.3 / (L/2)) * (n_x+1)/2)
filename = f'./Figures/{now}/neutron_evol_x3_C_{C}_dx_{d_x}_dt_{d_t}.png'

plot_linear(x=t, y_pd=u_pd_evolution.T[idx], y_rd=u_rd_evolution.T[idx], y_cn=u_cn_evolution.T[idx],
            title=f'Solution at x={round(x[idx],2)}', filename=filename,
            xlims=(0, t[-1]), ylims=0, x_label='$t$', y_label='$n(t)$')

# x = 0.4
idx = int((1 + 0.4 / (L/2)) * (n_x+1)/2)
filename = f'./Figures/{now}/neutron_evol_x4_C_{C}_dx_{d_x}_dt_{d_t}.png'

plot_linear(x=t, y_pd=u_pd_evolution.T[idx], y_rd=u_rd_evolution.T[idx], y_cn=u_cn_evolution.T[idx],
            title=f'Solution at x={round(x[idx],2)}', filename=filename,
            xlims=(0, t[-1]), ylims=0, x_label='$t$', y_label='$n(t)$')
