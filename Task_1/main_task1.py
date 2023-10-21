import os

import numpy as np
import matplotlib.pyplot as plt

from task_1_functions import beam_momentum_ode, shooting_method


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')

x_bc = np.array([-1, 1])
y_bc = np.array([[0, 0], [0, 0]])
is_bc = np.array([[True, True], [False, False]])
N = 100

y0, x, y = shooting_method(beam_momentum_ode, x_bc, y_bc, is_bc, N)

print(f'Initial condition: {y0}')

plt.plot(x[0,:], y[0,:])
plt.savefig('./Figures/test.png')