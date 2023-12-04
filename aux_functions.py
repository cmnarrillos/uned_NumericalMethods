import numpy as np


def rk4(f, t0, y0, h, params):
    """
    Apply the Runge-Kutta method of order 4 to solve a first-order ordinary
    differential equation of the form dy/dt = f(t, y).

    Args:
        f (function): The function representing the differential equation.
            It should take two arguments: t (current time) and y (current value).
        t0 (float): The initial time.
        y0 (numpy.ndarray): The initial value of y at time t0.
        h (float): The step size.
        params (dict): Used to pass params to the ODE which is being propagated.

    Returns:
        t_next (float): The updated time
        y_next (numpy.ndarray): The next value of y.
    """
    k1 = h * f(t0, y0, params)
    k2 = h * f(t0 + 0.5 * h, y0 + 0.5 * k1, params)
    k3 = h * f(t0 + 0.5 * h, y0 + 0.5 * k2, params)
    k4 = h * f(t0 + h, y0 + k3, params)

    y_next = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t_next = t0 + h

    return t_next, y_next


def newton_method(f, x0, tol=1e-6, max_iter=100, epsilon=1e-6):
    """
    Newton's method to find the root of a function.

    Args:
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
        dx = epsilon  # Small perturbation for finite differences
        derivative = (f(x + dx) - fx) / dx

        if np.all(abs(fx) < tol):
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


def newton_method_vect(f, x0, tol=1e-6, max_iter=100, epsilon=1e-6):
    """
    Newton's method to find the root of a vectorial function.

    Args:
        f (function): The target function for which you want to find the root.
        x0 (np.ndarray): Initial guess for the root.
        tol (float, optional): Tolerance for stopping criterion (default: 1e-6).
        max_iter (int, optional): Maximum number of iterations (default: 100).
        epsilon (float, optional): Perturbation to apply when computing gradient
            (default: 1e-6).

    Returns:
        root (np.ndarray): Approximated root of the function.
        iterations (int): Number of iterations performed.
    """
    x = x0
    num_vars = len(x)
    grad_fx = np.zeros([x.shape[0], x.shape[0]])

    for iter in range(max_iter):
        fx = f(x)

        # Compute the gradient (derivative) of f at x
        for ii in range(num_vars):
            eps = np.zeros(x.shape)
            eps[ii] = epsilon

            fx_eps = f(x + eps)
            grad_fx[ii] = (fx_eps - fx) / epsilon

        if np.all(np.abs(fx) < tol):
            return x, iter

        # Remove zero rows and columns from grad_fx
        reduced_grad_fx, removed_rows, removed_cols = remove_zero_rows_columns(grad_fx)

        # Remove the corresponding elements from fx
        reduced_fx = np.delete(fx, removed_cols)

        # Use the gradient to update x
        dx = solve_linear_system_with_lu_decomposition(reduced_grad_fx, reduced_fx)
        jj = 0
        for ii in range(x.size):
            if ii not in removed_rows:
                x[ii] = x[ii] - dx[jj]
                jj += 1

    raise Exception("Newton's method did not converge within max iterations")


def remove_zero_rows_columns(matrix):
    """
    Remove zero rows and columns from a given matrix.

    Args:
        matrix (np.ndarray): The input matrix from which to remove zero rows and columns.

    Returns:
        reduced_matrix (np.ndarray): The matrix with zero rows and columns removed.
        removed_rows (np.ndarray): Indices of the removed rows.
        removed_columns (np.ndarray): Indices of the removed columns.
    """
    non_zero_rows = np.any(matrix != 0, axis=1)
    non_zero_columns = np.any(matrix != 0, axis=0)
    reduced_matrix = matrix[non_zero_rows][:, non_zero_columns]
    removed_rows = np.where(~non_zero_rows)[0]
    removed_columns = np.where(~non_zero_columns)[0]
    return reduced_matrix, removed_rows, removed_columns


def calculate_norm(vector):
    """
    Computes the norm of a given vector.

    Args:
        vector (np.ndarray): The input vector whose norm will be computed.

    Returns:
        norm (float): Norm of the vector.
    """
    return np.sqrt(np.sum(vector**2))


def jacobi_method(A, b, x0=None, max_iterations=1000, tolerance=1e-10, aitken=True):
    """
    Solve a linear system Ax = b using the Jacobi iterative method.

    Parameters:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        x0: Initial guess for the solution (default is None, which initializes to zeros)
        max_iterations: Maximum number of iterations (default is 1000)
        tolerance: Convergence tolerance (default is 1e-10)
        aitken: boolean telling whether to use Aitken's acceleration method (default is True)

    Returns:
        x: Solution vector (n x 1)
        iterations: Number of iterations performed
    """

    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for iteration in range(max_iterations):
        x_old = x.copy()  # Store the previous iteration for convergence check

        for i in range(n):
            sigma = np.dot(A[i, :i], x_old[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sigma) / A[i, i]

        # Combine using Aitken's acc method
        x_out = x.copy()
        if aitken:
            if iteration > 1:
                for i in range(n):
                    x_out[i] = x[i] - (x[i]-x_old[i])**2 / (x[i] - 2*x_old[i] + x_prev[i])

            x_prev = x_old.copy()

        # Check for convergence
        residual = calculate_norm(np.dot(A, x_out) - b)
        if residual < tolerance:
            return x_out, iteration + 1

    raise RuntimeError("Jacobi method did not converge within the specified number of iterations.")


def gauss_seidel(A, b, x0=None, max_iterations=1000, tolerance=1e-10, aitken=True):
    """
    Solve a linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        x0: Initial guess for the solution (default is None, which initializes to zeros)
        max_iterations: Maximum number of iterations (default is 1000)
        tolerance: Convergence tolerance (default is 1e-10)
        aitken: boolean telling whether to use Aitken's acceleration method (default is True)

    Returns:
        x: Solution vector (n x 1)
        iterations: Number of iterations performed
    """

    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for iteration in range(max_iterations):
        x_old = x.copy()  # Store the previous iteration for convergence check

        for i in range(n):
            sigma_forward = np.dot(A[i, :i], x[:i])
            sigma_backward = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sigma_forward - sigma_backward) / A[i, i]

        # Combine using Aitken's acc method
        x_out = x.copy()
        if aitken:
            if iteration > 1:
                for i in range(n):
                    x_out[i] = x[i] - (x[i]-x_old[i])**2 / (x[i] - 2*x_old[i] + x_prev[i])

            x_prev = x_old.copy()

        # Check for convergence
        residual = calculate_norm(np.dot(A, x_out) - b)
        if residual < tolerance:
            return x_out, iteration + 1

    raise RuntimeError("Gauss-Seidel method did not converge within the specified number of iterations.")


def sor_method(A, b, w, x0=None, max_iterations=1000, tolerance=1e-10, aitken=True):
    """
    Solve a linear system Ax = b using the Successive Over-Relaxation (SOR) iterative method.

    Parameters:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        w: Relaxation parameter (0 < w < 2 for convergence)
        x0: Initial guess for the solution (default is None, which initializes to zeros)
        max_iterations: Maximum number of iterations (default is 1000)
        tolerance: Convergence tolerance (default is 1e-10)
        aitken: boolean telling whether to use Aitken's acceleration method (default is True)

    Returns:
        x: Solution vector (n x 1)
        iterations: Number of iterations performed
    """

    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for iteration in range(max_iterations):
        x_old = x.copy()  # Store the previous iteration for convergence check

        for i in range(n):
            sigma_forward = np.dot(A[i, :i], x[:i])
            sigma_backward = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (1 - w) * x_old[i] + (w / A[i, i]) * (b[i] - sigma_forward - sigma_backward)

        # Combine using Aitken's acc method
        x_out = x.copy()
        if aitken:
            if iteration > 1:
                for i in range(n):
                    x_out[i] = x[i] - (x[i]-x_old[i])**2 / (x[i] - 2*x_old[i] + x_prev[i])

            x_prev = x_old.copy()

        # Check for convergence
        residual = calculate_norm(np.dot(A, x_out) - b)
        if residual < tolerance:
            return x_out, iteration + 1

    raise RuntimeError("SOR method did not converge within the specified number of iterations.")


def inverse_lower_triangular(L):
    """
    Compute the inverse of a lower triangular matrix using back substitution.

    Parameters:
        L: Lower triangular matrix (n x n)

    Returns:
        L_inv: Inverse of the lower triangular matrix (n x n)
    """
    n = L.shape[0]
    if L.shape[0] != L.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Initialize the inverse matrix as an identity matrix
    L_inv = np.eye(n)

    for i in range(n):
        for j in range(i):
            L_inv[i, :] -= L_inv[j, :] * L[i, j]

        # Divide the entire row by the diagonal element
        L_inv[i, :] /= L[i, i]

    return L_inv
