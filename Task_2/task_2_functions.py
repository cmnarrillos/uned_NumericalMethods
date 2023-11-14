import numpy as np
import warnings

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import calculate_norm
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import calculate_norm


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
        eigenvalue (numpy.ndarray): Eigenvector associated to the maximum eigenvalue.

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

