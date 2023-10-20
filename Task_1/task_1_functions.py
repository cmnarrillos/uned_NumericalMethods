import numpy as np
from aux_functions import rk4, newton_method_vect


def beam_momentum_ode(x, y):
    """
    Right hand side of the ODE representing the

    Args:
        x (float): Independent variable of the equation.
        y (numpy.ndarray): Dependent variable and sucesive derivatives: y = [y, y', y'', ...].

    Returns:
        y_dot (numpy.ndarray): Derivative of the second input: y_dot = [y', y'', y''', ...].
    """
    y_dot = np.zeros(y.shape())
    y_dot[0] = y[1]
    y_dot[1] = -(1+x**2)*y[0] - 1

    return y_dot

def shooting_method(f, x_bc, y_bc, is_bc, N):
    """
    Shooting method for solving boundary condition ODE. Requires:
        - A propagator, in this case order 4 Runge-Kutta (rk4) will be used
        - A root-finder, here a vectorial Newton-Raphson (newton_method_vect)

    Args:
        f (function): ODE to solve with the given BCs.
        x_bc (numpy.ndarray): Positions where the Boundary Conditions are applied.
        y_bc (numpy.ndarray): Values of the dependent variable and its derivatives at BCs.
        is_bc (numpy.ndarray, boolean): Whether corresponding values from the previous
                                        input are actually fixed or not.
        N (int): Number of steps to divide each subinterval.

    Returns:
        y0 (numpy.ndarray): Initial condition which is equivalent to the given BCs.
        x (numpy.ndarray): Vector of positions where the ODE has been propagated.
        y (numpy.ndarray): Matrix of the dependent variable and its derivatives which
                           has been propagated.
    """
    # Define the stepsize for propagation
    h = (x_bc[1] - x_bc[0])/N

    # Create an initial condition
    x0 = x_bc[0]
    y0 = y_bc[:, 0].copy()

    # Define root function which will be used to find the IC equivalent to the BCs provided
    def shooting_function(y_IC):
        # Block the components of the Initial Condition which are actual BCs
        for ii in range(len(y_IC)):
            if is_bc[ii, 0]:
                y_IC[i] = y_bc[ii, 0]

        # Propagate the equation using Runge-Kutta method
        x = np.zeros([1, N+1])
        y = np.zeros([len(y_IC), N+1])
        x[0] = x0
        y[:, 0] = y_IC
        for ii in range(N+1):
            x[ii+1], y[:, ii+1] = rk4(f, x[ii], y[:, ii], h)

        # Check difference in the boundary condition
        diff = 0
        for ii in range(len(y_IC)):
            if is_bc[ii, 1]:
                diff += y[ii, N+1] - y_bc[ii, 1]

        return diff

    # Find the roots of the shooting function, that is, the initial condition
    # which makes the final state match the boundary condition
    y0_root = newton_method_vect(shooting_function, y0)
    y0 = y0_root

    # Propagate the equation with that initial condition using Runge-Kutta method
    x = np.zeros([1, N + 1])
    y = np.zeros([len(y0), N + 1])
    x[0] = x0
    y[:, 0] = y0
    for ii in range(N + 1):
        x[ii + 1], y[:, ii + 1] = rk4(f, x[ii], y[:, ii], h)

    return y0, x, y
