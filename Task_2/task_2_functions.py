import numpy as np
import warnings

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import calculate_norm, solve_linear_system_with_lu_decomposition
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import calculate_norm, solve_linear_system_with_lu_decomposition


def power_method(matrix, initial_vector=None, tolerance=1e-6, max_iterations=1000):
    """
    Compute the maximum eigenvalue of a square matrix using the power method.

    Args:
        matrix (numpy.array): The input square matrix (NxN).
        initial_vector (numpy.array, optional): The initial vector for iteration.
            If not provided, a vector of N random real numbers between 0 and 1 will be used.
            If provided, must be a vector of the same size as the matrix.
        tolerance (float, optional): Tolerance for convergence (default is 1e-6).
        max_iterations (int, optional): Maximum number of iterations before raising a warning (default is 1000).

    Returns:
        eigenvalue (float): Maximum eigenvalue of the matrix.
        eigenvector (numpy.ndarray): Eigenvector associated to the maximum eigenvalue.

    Raises:
        ValueError: If the matrix provided is rectangular.
        ValueError: If the size of the provided initial vector is different from the matrix size.
        ValueError: If initial vector provided is virtually 0.
        UserWarning: If the iteration does not converge within the specified maximum number of iterations.
    """

    # Check matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")

    # Set the initial vector if not provided
    if initial_vector is None:
        initial_vector = np.random.rand(matrix.shape[0])

    # Check the size of the initial vector
    if initial_vector.shape[0] != matrix.shape[0]:
        raise ValueError("Size of the initial vector must match the matrix size")

    # Check initial vector is not null
    if max(np.abs(initial_vector)) < tolerance:
        raise ValueError("Initial vector must be a non-0 vector")

    # Normalize the initial vector
    initial_vector /= calculate_norm(initial_vector)

    # Power method iteration
    eigenvalue_old = 0.0
    for iteration in range(max_iterations):
        # Power iteration
        eigenvector = np.dot(matrix, initial_vector)
        eigenvalue = np.dot(initial_vector, eigenvector)

        # Normalize the eigenvector
        eigenvector /= calculate_norm(eigenvector)

        # Check for convergence
        if np.abs(eigenvalue - eigenvalue_old) < tolerance:
            return eigenvalue, eigenvector

        # Update for next step
        eigenvalue_old = eigenvalue
        initial_vector = eigenvector

    # If the iteration does not converge, raise a warning
    warnings.warn("Power method did not converge within the specified maximum number of iterations")

    return eigenvalue, eigenvector


def inverse_power_method(matrix, eigenvalue_est, initial_vector=None, tolerance=1e-6, max_iterations=1000):
    """
    Compute the eigenvalue and eigenvector of a square matrix using the inverse power method.

    Parameters:
        matrix (numpy.array): The input square matrix (NxN).
        eigenvalue_est (float): Initial estimation for the eigenvalue.
        initial_vector (numpy.array, optional): The initial vector for iteration.
            If not provided, a vector of N random real numbers between 0 and 1 will be used.
            If provided, must be a vector of the same size as the matrix.
        tolerance (float, optional): Tolerance for convergence (default is 1e-6).
        max_iterations (int, optional): Maximum number of iterations before raising a warning (default is 1000).

    Returns:
        eigenvalue (float): Closest eigenvalue of the matrix to the input eigenvalue_est.
        eigenvector (numpy.ndarray): Eigenvector associated to the found eigenvalue.

    Raises:
        ValueError: If the matrix provided is rectangular.
        ValueError: If the size of the provided initial vector is different from the matrix size.
        ValueError: If initial vector provided is virtually 0.
        UserWarning: If the iteration does not converge within the specified maximum number of iterations.
    """

    # Check matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")

    # Set the initial vector if not provided
    if initial_vector is None:
        initial_vector = np.random.rand(matrix.shape[0])

    # Check the size of the initial vector
    if initial_vector.shape[0] != matrix.shape[0]:
        raise ValueError("Size of the initial vector must match the matrix size")

    # Check the size of the initial vector
    if initial_vector.shape[0] != matrix.shape[0]:
        raise ValueError("Size of the initial vector must match the matrix size")

    # Normalize the initial vector
    initial_vector /= calculate_norm(initial_vector)

    # Inverse power method iteration
    for iteration in range(max_iterations):
        # Create auxiliary matrix
        aux_matrix = matrix - eigenvalue_est * np.identity(matrix.shape[0])
        # Solve the linear system using LU decomposition
        solve = solve_linear_system_with_lu_decomposition(aux_matrix, initial_vector)

        # Normalize the result
        solve /= calculate_norm(solve)

        # Update the estimation of the eigenvalue
        eigenvalue_est = np.dot(initial_vector, np.dot(matrix, initial_vector)) / np.dot(initial_vector, initial_vector)

        # Check for convergence
        if calculate_norm(initial_vector - solve) < tolerance \
                or calculate_norm(initial_vector + solve) < tolerance:
            return eigenvalue_est, solve

        initial_vector = solve

    # If the iteration does not converge, raise a warning
    warnings.warn("Inverse power method did not converge within the specified maximum number of iterations")

    return eigenvalue_est, initial_vector


def is_tri_diagonal(matrix, tolerance=1e-6):
    """
    Check if a matrix is tri-diagonal.

    Parameters:
        matrix (numpy.array): The input square matrix (NxN).
        tolerance (float, optional): Tolerance for condition. (default is 1e-6).

    Returns:
        bool: True if the matrix is tri-diagonal, False otherwise.

    Raises:
        ValueError: If the matrix provided is rectangular.
    """

    # Check matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")

    n = matrix.shape[0]

    # Check off-diagonal elements
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(matrix[i, j]) > tolerance:
                return False

    return True


def rotation_matrix(n, indexes, c, s):
    """
    Create an NxN rotation matrix.

    Parameters:
        n (int): Size of the matrix.
        indexes (tuple of 2 ints): Rows/cols different from the identity matrix.
        c (float): Real number.
        s (float): Real number.

    Returns:
        rot_matrix (numpy.array): NxN rotation matrix.
    """

    # Check that c^2 + s^2 = 1 (normalize if needed)
    norm_factor = np.sqrt(c**2 + s**2)
    if norm_factor != 1:
        c /= norm_factor
        s /= norm_factor

    # Initialize the rotation matrix as an identity matrix
    rot_matrix = np.eye(n)

    # Set values for the specified rows/cols
    i, j = indexes
    rot_matrix[i, i] = c
    rot_matrix[i, j] = s
    rot_matrix[j, i] = -s
    rot_matrix[j, j] = c

    return rot_matrix


def qr_method(matrix, tolerance=1e-6, max_iterations=1000):
    """
    Args:
        matrix (numpy.array): The input square matrix (NxN).
        tolerance (float, optional): Tolerance for convergence (default is 1e-6).
        max_iterations (int, optional): Maximum number of iterations before raising a warning (default is 1000).

    Returns:
        eigenvalues (list): List containing eigenvalues of the matrix.

    Raises:
        ValueError: If the matrix provided is rectangular.
        ValueError: If the matrix provided is not tri-diagonal.
        ValueError: If the matrix provided is not symmetric.
    """

    # Check matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")

    # Check matrix is tri-diagonal
    if not is_tri_diagonal(matrix, tolerance=1e-9):
        raise ValueError("Input matrix must be tri-diagonal")

    # Check matrix is symmetric
    if (abs(matrix - matrix.T) > tolerance).any():
        raise ValueError("Input matrix must be symmetric")

    # Get size of matrix
    n = matrix.shape[0]

    # Initialize empty list to store eigenvalues
    eigenvalues_list = []

    # Return the only eigenvalue if matrix is [1x1]
    if n == 1:
        eigenvalues_list.append(matrix[0, 0])
        return eigenvalues_list

    # QR method iteration
    for iteration in range(max_iterations):

        # Check if eigenvalues can be directly extracted
        # - 1st eigenvalue
        if abs(matrix[0, 1]) < tolerance:
            reduced_matrix = matrix[1:, 1:]
            eigenvalues_list = qr_method(reduced_matrix, tolerance=tolerance, max_iterations=max_iterations-iteration)
            eigenvalues_list.append(matrix[0, 0])
            return eigenvalues_list

        # - Last eigenvalue
        if abs(matrix[n-2, n-1]) < tolerance:
            reduced_matrix = matrix[:n-1, :n-1]
            eigenvalues_list = qr_method(reduced_matrix, tolerance=tolerance, max_iterations=max_iterations-iteration)
            eigenvalues_list.append(matrix[n-1, n-1])
            return eigenvalues_list

        # Divide in 2 tri-diagonal matrices if possible
        for k in range(1, n-1):
            if abs(matrix[k, k+1]) < tolerance:
                matrix_1 = matrix[:k, :k]
                matrix_2 = matrix[k:, k:]
                eig_val_list_1 = qr_method(matrix_1, tolerance=tolerance, max_iterations=max_iterations-iteration)
                eig_val_list_2 = qr_method(matrix_2, tolerance=tolerance, max_iterations=max_iterations-iteration)
                eigenvalues_list = eig_val_list_1 + eig_val_list_2
                return eigenvalues_list

        # Actual method
        # - Compute rotation matrices for building Q
        #       Initialize vectors to store sin/cos
        s = np.zeros(n-1)
        c = np.zeros(n-1)
        # - Compute R
        r_matrix = np.zeros(matrix.shape)
        #       Initialize aux variables (x,y) to use while iterating over matrix elements (temporal elements in row k)
        x = matrix[0, 0]
        y = matrix[0, 1]
        for k in range(n-1):
            b = matrix[k, k+1]
            a = matrix[k+1, k+1]
            # Compute and store rotations at step k
            s[k] = b/np.sqrt(b**2 + x**2)
            c[k] = x/np.sqrt(b**2 + x**2)
            # Update row k in R matrix
            r_matrix[k, k] = c[k]*x + s[k]*b
            r_matrix[k, k+1] = c[k]*y + s[k]*a
            if k < n-2:
                r_matrix[k, k+2] = s[k]*b
            # Update aux variables (x,y)
            x = c[k]*a - s[k]*y
            y = c[k]*b
        r_matrix[n-1, n-1] = x

        # - Build Q from the stored info for rotation matrices
        q_matrix = np.eye(n)
        for k in range(n-1):
            rot_matrix = rotation_matrix(n, (k, k+1), c[k], -s[k])
            q_matrix = np.dot(q_matrix, rot_matrix)

        # - Update matrix
        matrix = np.dot(np.dot(q_matrix.T, matrix), q_matrix)

    # If the iteration does not converge, raise a warning
    warnings.warn("QR method did not converge within the specified maximum number of iterations")

    return eigenvalues_list
