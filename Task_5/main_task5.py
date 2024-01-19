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

def f(x, params={}):
    return np.zeros(x.shape)


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')

u_0 = 0
u_N = np.tan(1) - 1
du_dx_N = 0

# Create generic system for the function given by the rhs and alpha^2=1
A_out, F_out, x = finite_elements(alpha2=1, f=rhs, num_elem=4, order=2, lims=(0, 1), analytical=False)

# Apply Dirichlet boundary condition u(x=0) = 0
A_syst = A_out[1:, 1:]
F_syst = F_out[1:, 0] - A_out[1:, 0] * u_0
# Apply Neumann boundary condition du/dx(x=1) = 0
A_syst[-1, -1] += du_dx_N

# # # Apply 2 Dirichlet boundary conditions
# A_syst = A_out[1:-1, 1:-1]
# F_syst = F_out[1:-1, 0] - A_out[1:-1, 0] * u_0 - A_out[1:-1, -1] * u_N

u_syst = solve_linear_system_with_lu_decomposition(A_syst, F_syst)
u_sol = np.zeros(F_out.shape)
u_sol[0, 0] = u_0
u_sol[1:, 0] = u_syst

# u_sol[1:-1, 0] = u_syst
# u_sol[-1, 0] = u_N


x_vect = np.linspace(0, 1, 101)
y_vect = -x_vect**3/6
# y_vect = np.exp(3*x_vect)
# y_vect = np.sin(3*x_vect)
y_vect = analytic_sol(x_vect)

plt.figure()
plt.plot(x, u_sol, 'xr')
plt.plot(x_vect, y_vect, '-k')
plt.savefig('./Figures/analytic_vs_num.png')
plt.close()
