import os
import time
import numpy as np
import matplotlib.pyplot as plt

from task_4_functions import iterative_pde_solver, theta_method_neutron
try:
    from aux_functions import dirac_delta
except ImportError:
    import sys
    sys.path.append(os.path.abspath('..'))
    from aux_functions import dirac_delta



# Initialize parameters of the method
L = 1
D = 1
C = 0

# Discretization
n_x = 11
d_t = 0.0049
n_t = 100

# Initial condition
x = np.linspace(-L/2, L/2, n_x)
u_0 = np.array([dirac_delta(x_val, tol=1.e-3) for x_val in x])

# Progressive differences
A, B = theta_method_neutron(n_x=n_x, dt=d_t, dx=L/(n_x-1), d=D, c=C, theta=0)

u_pd, u_pd_evolution = iterative_pde_solver(u_0=u_0, A=A, B=B, n_t=n_t)
print(f'Progressive diff solution: {u_pd}')

# Regressive differences
A, B = theta_method_neutron(n_x=n_x, dt=d_t, dx=L/(n_x-1), d=D, c=C, theta=1)

u_rd, u_rd_evolution = iterative_pde_solver(u_0=u_0, A=A, B=B, n_t=n_t)
print(f'Regressive diff solution: {u_rd}')

# Crank-Nicholson differences
A, B = theta_method_neutron(n_x=n_x, dt=d_t, dx=L/(n_x-1), d=D, c=C, theta=0.5)

u_cn, u_cn_evolution = iterative_pde_solver(u_0=u_0, A=A, B=B, n_t=n_t)
print(f'Crank-Nicholson diff solution: {u_cn}')

