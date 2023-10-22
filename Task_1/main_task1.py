import os

import numpy as np
import matplotlib.pyplot as plt

from task_1_functions import beam_momentum_ode, shooting_method
from task_1_functions import p_beam, q_beam, r_beam, finite_diff_order2
from aux_functions import solve_linear_system_with_lu_decomposition


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')


# Shooting method
x_bc = np.array([-1, 1])
y_bc = np.array([[0, 0], [0, 0]])
is_bc = np.array([[True, True], [False, False]])
N = 100
params = []

y0, x, y = shooting_method(beam_momentum_ode, x_bc, y_bc, is_bc, N, params)

print(f'Initial condition: {y0}')
y_plot = y[:, 0]
plt.plot(x, y_plot, label='shooting')
plt.xlabel('$ x=\\xi/L $')
plt.ylabel('$ y=M/(p_0 L^2) $')
plt.xlim(min(x), max(x))
plt.ylim(min(y_plot)*1.1 - 0.1, max(y_plot)*1.1 + 0.1)
plt.grid('minor')
plt.savefig(f'./Figures/test_shooting_N_{N}.png')

# Finite differences
x_bc = (-1, 1)
y_bc = (0, 0)
A, b = finite_diff_order2(p=p_beam, q=q_beam, r=r_beam, x_bc=x_bc, y_bc=y_bc, n=N)

y_intermediate = solve_linear_system_with_lu_decomposition(A, b)
x = np.linspace(-1, 1, N+1)
y = np.concatenate(([y_bc[0]], y_intermediate, [y_bc[-1]]))

plt.plot(x, y, label='finite_diff')
plt.legend()
plt.xlabel('$ x=\\xi/L $')
plt.ylabel('$ y=M/(p_0 L^2) $')
plt.xlim(min(x), max(x))
plt.ylim(min(y)*1.1 - 0.1, max(y)*1.1 + 0.1)
plt.grid('minor')
plt.savefig(f'./Figures/test_finite_diff_vs_shooting_N_{N}.png')

plt.figure()

plt.plot(x, y, label='finite_diff')
plt.legend()
plt.xlabel('$ x=\\xi/L $')
plt.ylabel('$ y=M/(p_0 L^2) $')
plt.xlim(min(x), max(x))
plt.ylim(min(y)*1.1 - 0.1, max(y)*1.1 + 0.1)
plt.grid('minor')
plt.savefig(f'./Figures/test_finite_diff_N_{N}.png')
