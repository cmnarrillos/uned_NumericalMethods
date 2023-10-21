import numpy as np
from aux_functions import rk4, newton_method, newton_method_vect


def beam_momentum_ode(x, y):
    """
    Right hand side of the ODE representing the momentum of a variable stiffness beam compressed
    by a longitudinal force under a transversal distributed load: y'' = -(1+x^2) y - 1

    Args:
        x (float): Independent variable of the equation.
        y (numpy.ndarray): Dependent variable and sucesive derivatives: y = [y, y', y'', ...].

    Returns:
        y_dot (numpy.ndarray): Derivative of the second input: y_dot = [y', y'', y''', ...].
    """
    y_dot = np.zeros(y.shape)
    y_dot[0] = y[1]
    y_dot[1] = -(1 + x**2) * y[0] - 1

    return y_dot


def p_beam(x):
    """
    1-variable function, multiplying y' in the ODE y'' = p(x)*y' + q(x)*y + r(x).

    Args:
        x(np.ndarray): Vector of independent variable discretized in a set of positions.

    Returns:
        p(x)=0 (np.ndarray): Values of p(x) for the beam problem in the given positions.

    """
    return np.zeros(x.shape)


def q_beam(x):
    """
    1-variable function, multiplying y in the ODE y'' = p(x)*y' + q(x)*y + r(x).

    Args:
        x(np.ndarray): Vector of independent variable discretized in a set of positions.

    Returns:
        q(x)=-(1 + x^2) (np.ndarray): Values of q(x) for the beam problem in the set.

    """
    return -(1 + x**2)


def r_beam(x):
    """
    1-variable function, independent term in the ODE y'' = p(x)*y' + q(x)*y + r(x).

    Args:
        x(np.ndarray): Vector of independent variable discretized in a set of positions.

    Returns:
        r(x)=-1 (np.ndarray): Values of r(x) for the beam problem in the set.

    """
    return -np.ones(x.shape)


def shooting_method(f, x_bc, y_bc, is_bc, N):
    """
    Shooting method for solving boundary condition ODE. Requires:
        - A propagator, in this case order 4 Runge-Kutta (rk4) will be used
        - A root-finder, here a vectorial Newton-Raphson (newton_method/newton_method_vect)

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
                y_IC[ii] = y_bc[ii, 0]

        # Propagate the equation using Runge-Kutta method
        x = np.zeros([1, N+1])
        y = np.zeros([len(y_IC), N+1])
        x[:, 0] = x0
        y[:, 0] = y_IC
        for ii in range(N):
            x[:, ii+1], y[:, ii+1] = rk4(f, x[:, ii], y[:, ii], h)

        # Check difference in the boundary condition
        diff = 0
        for ii in range(len(y_IC)):
            if is_bc[ii, 1]:
                diff += y[ii, N] - y_bc[ii, 1]

        return diff

    # Find the roots of the shooting function, that is, the initial condition
    # which makes the final state match the boundary condition
    if len(is_bc[:, 0]) - sum(is_bc[:, 0]) > 1:
        y0_root, niter = newton_method_vect(shooting_function, y0)
    else:
        y0_root, niter = newton_method(shooting_function, y0)
    y0 = y0_root

    # Propagate the equation with that initial condition using Runge-Kutta method
    x = np.zeros([1, N + 1])
    y = np.zeros([len(y0), N + 1])
    x[:, 0] = x0
    y[:, 0] = y0
    for ii in range(N):
        x[:, ii+1], y[:, ii+1] = rk4(f, x[:, ii], y[:, ii], h)

    return y0, x, y


def finite_diff_order2(p, q, r, x_bc, y_bc, N):
    """
    Function which creates the linear system Ay=b obtained when applying finite difference method
    to the ODE: y'' = p(x)*y' + q(x)*y + r(x) with BCs imposed on y_0, y_{N+1}

    Args:
        p (function): 1-variable function, multiplying y' in the ODE.
        q (function): 1-variable function, multiplying y in the ODE.
        r (function): 1-variable function, the independent term of the ODE.
        x_bc (tuple): Pair of values of x where BCs are applied.
        y_bc (tuple): Boundary Conditions fixed for the equation.
        N (int): Number of steps to discretize the interval given at x_BC, also dictates the
                 size of the resulting matrix and vector.

    Returns:
        A (np.ndarray): (N-1)x(N-1) matrix with the coefficients of the resulting linear system.
        b (np.ndarray): (N-1) vector with the independent terms of the resulting linear system.
    """
    # Discretize the interval and obtain the stepsize
    x = np.linspace(x_bc[0], x_bc[-1], N+1)
    h = x[1]-x[0]

    # Generate the matrix A
    A = np.zeros([N-1, N-1])
    for ii in range(N-1):
        A[ii, ii] = 2 + h**2 * q(x[ii])
        if ii > 0:
            A[ii, ii-1] = -1 - h/2 * p(x[ii])
        if ii < N-2:
            A[ii, ii+1] = -1 + h/2 * p(x[ii])

    # Generate the vector b
    b = -h**2 * r(x[1:-1])
    b[0] += (1 + h/2 * p(x[1])) * y_bc[0]
    b[-1] += (1 + h/2 * p(x[-2])) * y_bc[-1]

    return A, b

