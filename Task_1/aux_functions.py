def rk4(f, y0, t0, h):
    """
    Apply the Runge-Kutta method of order 4 to solve a first-order ordinary
    differential equation of the form dy/dt = f(t, y).

    Parameters:
    - f (function): The function representing the differential equation.
        It should take two arguments: t (current time) and y (current value).
    - y0 (float): The initial value of y at time t0.
    - t0 (float): The initial time.
    - h (float): The step size.

    Returns:
    - t_next (float): The updated time
    - y_next (float): The next value of y.
    """
    k1 = h * f(t0, y0)
    k2 = h * f(t0 + 0.5 * h, y0 + 0.5 * k1)
    k3 = h * f(t0 + 0.5 * h, y0 + 0.5 * k2)
    k4 = h * f(t0 + h, y0 + k3)

    y_next = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t_next = t0 + h

    return t_next, y_next


def newton_method(f, x0, tol=1e-6, max_iter=100):
    """
    Newton's method to find the root of a function.

    Parameters:
    - f: The target function for which you want to find the root.
    - x0: Initial guess for the root.
    - tol: Tolerance for stopping criterion (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).

    Returns:
    - root: Approximated root of the function.
    - iterations: Number of iterations performed.
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        # Approximate the derivative using finite differences
        dx = 1e-6  # Small perturbation for finite differences
        derivative = (f(x + dx) - fx) / dx

        if abs(fx) < tol:
            return x, i
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

    for k in range(n):
        L[k, k] = 1.0
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

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
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y for x
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

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