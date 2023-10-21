import numpy as np


def rk4(f, t0, y0, h):
    """
    Apply the Runge-Kutta method of order 4 to solve a first-order ordinary
    differential equation of the form dy/dt = f(t, y).

    Parameters:
        f (function): The function representing the differential equation.
            It should take two arguments: t (current time) and y (current value).
        t0 (float): The initial time.
        y0 (numpy.ndarray): The initial value of y at time t0.
        h (float): The step size.

    Returns:
        t_next (float): The updated time
        y_next (numpy.ndarray): The next value of y.
    """
    k1 = h * f(t0, y0)
    k2 = h * f(t0 + 0.5 * h, y0 + 0.5 * k1)
    k3 = h * f(t0 + 0.5 * h, y0 + 0.5 * k2)
    k4 = h * f(t0 + h, y0 + k3)

    y_next = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t_next = t0 + h

    return t_next, y_next


def newton_method(f, x0, tol=1e-6, max_iter=100, eps=1e-6):
    """
    Newton's method to find the root of a function.

    Parameters:
        f (function): The target function for which you want to find the root.
        x0 (float): Initial guess for the root.
        tol (float, optional): Tolerance for stopping criterion (default: 1e-6).
        max_iter (int, optional): Maximum number of iterations (default: 100).
        eps (float, optional): Perturbation to apply when computing gradient
            (default: 1e-6).

    Returns:
        root (float): Approximated root of the function.
        iterations (int): Number of iterations performed.
    """
    x = x0
    for ii in range(max_iter):
        fx = f(x)
        # Approximate the derivative using finite differences
        dx = eps  # Small perturbation for finite differences
        derivative = (f(x + dx) - fx) / dx

        if abs(fx) < tol:
            return x, ii
        x = x - fx / derivative

    raise Exception("Newton's method did not converge within max iterations")


def lu_decomposition(A):
    """
    Perform LU decomposition on the given coefficient matrix A.

    Args:
        A (numpy.ndarray): The coefficient matrix of the linear system.

    Returns:
        L (numpy.ndarray): The lower triangular matrix.
        U (numpy.ndarray): The upper triangular matrix.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.copy(A)

    for jj in range(n):
        L[jj, jj] = 1.0
        for ii in range(jj + 1, n):
            factor = U[ii, jj] / U[jj, jj]
            L[ii, jj] = factor
            U[ii, jj:] -= factor * U[jj, jj:]

    return L, U


def lu_solve(L, U, b):
    """
    Solve a linear system using LU decomposition.

    Args:
        L (numpy.ndarray): Lower triangular matrix from LU decomposition.
        U (numpy.ndarray): Upper triangular matrix from LU decomposition.
        b (numpy.ndarray): Right-hand side vector of the linear system.

    Returns:
        x (numpy.ndarray): Solution vector.
    """
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    # Solve Ly = b for y
    for ii in range(n):
        y[ii] = b[ii] - np.dot(L[ii, :ii], y[:ii])

    # Solve Ux = y for x
    for ii in range(n - 1, -1, -1):
        x[ii] = (y[ii] - np.dot(U[ii, ii + 1:], x[ii + 1:])) / U[ii, ii]

    return x


def solve_linear_system_with_lu_decomposition(A, b):
    """
    Solve a linear system Ax = b using LU decomposition.

    Args:
        A (numpy.ndarray): The coefficient matrix of the linear system.
        b (numpy.ndarray): The right-hand side vector of the linear system.

    Returns:
        x (numpy.ndarray): The solution vector that satisfies Ax = b.
    """
    # Perform LU decomposition
    L, U = lu_decomposition(A)

    # Solve the linear system using LU decomposition
    solution = lu_solve(L, U, b)

    return solution


def newton_method_vect(f, x0, tol=1e-6, max_iter=100, eps=1e-6):
    """
    Newton's method to find the root of a vectorial function.

    Parameters:
        f (function): The target function for which you want to find the root.
        x0 (np.ndarray): Initial guess for the root.
        tol (float, optional): Tolerance for stopping criterion (default: 1e-6).
        max_iter (int, optional): Maximum number of iterations (default: 100).
        eps (float, optional): Perturbation to apply when computing gradient
            (default: 1e-6).

    Returns:
        root (np.ndarray): Approximated root of the function.
        iterations (int): Number of iterations performed.
    """
    x = x0
    num_vars = len(x)
    grad_fx = np.zeros(x.shape)

    for iter in range(max_iter):
        fx = f(x)

        # Compute the gradient (derivative) of f at x
        for ii in range(num_vars):
            x_eps = x.copy()
            x_eps[ii] += eps

            grad_fx[ii] = (f(x_eps) - fx) / eps

        if np.all(np.abs(fx) < tol):
            return x, iter

        # Use the gradient to update x
        x = x - solve_linear_system_with_lu_decomposition(np.diag(grad_fx), fx)

    raise Exception("Newton's method did not converge within max iterations")
