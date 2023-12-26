import numpy as np
import matplotlib.pyplot as plt
import warnings

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import check_tridiagonal, crout_tridiagonal_solver, solve_linear_system_with_lu_decomposition
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import check_tridiagonal, crout_tridiagonal_solver,solve_linear_system_with_lu_decomposition


def iterative_pde_solver(u_0, u_BC, A, B, D_rhs, D_lhs, n_t):
    """
    Solve a PDE using finite differences general form given by:
    A*u(i+1) = B*u(i)

    Args:
        u_0 (np.ndarray): vector with the initial condition
        u_BC (np.ndarray): list the boundary conditions
        A (np.ndarray): matrix of coefficients on the left hand side
        B (np.ndarray): matrix of coefficients on the right hand side
        D_rhs (np.ndarray): matrix of boundary conditions on the left hand side
        D_lhs (np.ndarray): matrix of boundary conditions on the right hand side
        n_t (int): number of timesteps to perform

    Returns:
        u (np.ndarray): vector at the end of the propagation
        u_t (np.ndarray): matrix containing the evolution of u through the simulation
    """
    # copy/ store initial condition
    u = u_0.copy()
    u_t = [u_0]
    # Remove boundaries, which are fixed
    u = u[1:-1]

    # Extract diags from A to solve the equation
    if check_tridiagonal(A):
        diag = np.diag(A)
        low = np.diag(A, k=-1)
        up = np.diag(A, k=1)

        # Propagate solving the equation for n steps
        for _ in range(n_t):
            b = np.dot(B, u) + np.dot(D_rhs, u_BC) - np.dot(D_lhs, u_BC)
            u = crout_tridiagonal_solver(diag, low, up, b)
            u_t.append(np.append(np.insert(u, 0, u_BC[0]), u_BC[-1]))

    else:
        b = np.dot(B, u) + np.dot(D_rhs, u_BC) - np.dot(D_lhs, u_BC)
        u = solve_linear_system_with_lu_decomposition(A, b)
        u_t.append(np.append(np.insert(u, 0, u_BC[0]), u_BC[-1]))

    return np.append(np.insert(u, 0, u_BC[0]), u_BC[-1]), np.array(u_t, dtype=float)


def prog_diff_method(u_0, u_BC, B, D_rhs, n_t):
    """
    Solve a PDE using finite differences general form given by:
    A*u(i+1) = B*u(i)

    Args:
        u_0 (np.ndarray): vector with the initial condition
        u_BC (np.ndarray): list the boundary conditions
        B (np.ndarray): matrix of coefficients on the right hand side
        D_lhs (np.ndarray): matrix of boundary conditions on the right hand side
        n_t (int): number of timesteps to perform

    Returns:
        u (np.ndarray): vector at the end of the propagation
        u_t (np.ndarray): matrix containing the evolution of u through the simulation
    """
    # copy/ store initial condition
    u = u_0.copy()
    u_t = [u_0]
    # Remove boundaries, which are fixed
    u = u[1:-1]

    # Propagate the equation for n steps
    for _ in range(n_t):
        u = np.dot(B, u) + np.dot(D_rhs, u_BC)
        u_t.append(np.append(np.insert(u, 0, u_BC[0]), u_BC[-1]))

    return np.append(np.insert(u, 0, u_BC[0]), u_BC[-1]), np.array(u_t, dtype=float)


def theta_method_neutron(n_x, dt, dx, c, d, theta=0.):
    """
    Function which creates the functions of the system

    Args:
        n_x (int): number of elements of the vector u(t) - matrices are n_x*n_x
        dt (float): interval in t
        dx (float): interval in x
        c (float): coefficient of neutron creation associated to reaction
        d (float): coefficient of diffusion
        theta (float): allows to select different methods:
            - theta = 0 --> Progressive differences
            - theta = 1 --> Regressive differences (unconditionally stable)
            - theta = 0.5 --> Crank-Nicholson (unconditionally stable, 2nd order in t)

    Returns:
        A (np.ndarray): matrix of coefficients on the left hand side
        B (np.ndarray): matrix of coefficients on the right hand side
        D_rhs (np.ndarray): matrix of boundary conditions on the left hand side
        D_lhs (np.ndarray): matrix of boundary conditions on the right hand side
    """
    # Create elements of the diagonal
    A = np.eye(n_x) * (1 - theta*c*dt + 2*theta*d*dt/dx**2)
    B = np.eye(n_x) * (1 + (1-theta)*c*dt - 2*(1-theta)*d*dt/dx**2)

    # Create out-of-diagonal elements
    for ii in range(n_x-1):
        A[ii, ii+1] -= theta*d*dt/dx**2
        A[ii+1, ii] -= theta*d*dt/dx**2
        B[ii, ii+1] += (1-theta)*d*dt/dx**2
        B[ii+1, ii] += (1-theta)*d*dt/dx**2

    # Create arrays to include boundary conditions
    D_lhs = np.zeros(shape=(n_x, 2))
    D_rhs = np.zeros(shape=(n_x, 2))
    # Left boundary
    D_lhs[0, 0] -= theta*d*dt/dx**2
    D_rhs[0, 0] += (1-theta)*d*dt/dx**2
    # Right boundary
    D_lhs[-1, -1] -= theta*d*dt/dx**2
    D_rhs[-1, -1] += (1-theta)*d*dt/dx**2

    return A, B, D_rhs, D_lhs

