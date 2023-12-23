import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_4_functions import iterative_pde_solver, theta_method_neutron
try:
    from aux_functions import dirac_delta, integrate_trapezoidal
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import dirac_delta, integrate_trapezoidal



# Initialize parameters of the method
L = 1
D = 1
C = 0

# Discretization
n_x = 9
d_t = 0.0049
n_t = 100

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
u_pd, u_pd_evolution = iterative_pde_solver(u_0=u_0, u_BC=u_BC, A=A, B=B, D_rhs=D_rhs, D_lhs=D_lhs, n_t=n_t)
# Get avg neutron density
neutron_avg_pd = np.array([integrate_trapezoidal(x, u_step, lims=(0, L/2)) for u_step in u_pd_evolution])
print(f'Progressive diff solution: {u_pd}')


# Regressive differences
# Build the matrices involved
A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=1)
# Propagate
u_rd, u_rd_evolution = iterative_pde_solver(u_0=u_0, u_BC=u_BC, A=A, B=B, D_rhs=D_rhs, D_lhs=D_lhs, n_t=n_t)
# Get avg neutron density
neutron_avg_rd = np.array([integrate_trapezoidal(x, u_step, lims=(0, L/2)) for u_step in u_rd_evolution])
print(f'Regressive diff solution: {u_rd}')


# Crank-Nicholson differences
# Build the matrices involved
A, B, D_rhs, D_lhs = theta_method_neutron(n_x=n_x, dt=d_t, dx=d_x, d=D, c=C, theta=0.5)
# Propagate
u_cn, u_cn_evolution = iterative_pde_solver(u_0=u_0, u_BC=u_BC, A=A, B=B, D_rhs=D_rhs, D_lhs=D_lhs, n_t=n_t)
# Get avg neutron density
neutron_avg_cn = np.array([integrate_trapezoidal(x, u_step, lims=(0, L/2)) for u_step in u_cn_evolution])
print(f'Crank-Nicholson diff solution: {u_cn}')

