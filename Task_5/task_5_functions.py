import numpy as np
import warnings

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import integrate_trapezoidal
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import integrate_trapezoidal


def analytic_sol(x):
    """
    Analytical solution of the ode d2u/dx2 + u = -x
    with BCS: u(0)=0, du/dx(1)=0

    Args:
        x (float/np.ndarray): independent variable

    Returns:
        y(x) = sin(x)/cos(1) - x
    """
    return np.sin(x)/np.cos(1) - x


def rhs(x, params={}):
    """
    Right hand side of the differential equation d2u/dx2 + alpha u = f(x)

    Args:
        x (float/np.ndarray): independent variable
        params (dict): dictionary of parameters (empty)

    Returns:
        value of f(x) = -x
    """
    return -x


def local_interpolator(x, params):
    """
    Local Interpolating functions

    Args:
        x (float/np.ndarray): independent variable
        params (dict): dictionary of parameters:
          - order (int): order of the local interpolator
          - index (int): index (between 1 and order+1)
          - h (float): width of the element

    Returns:
        f_x (float/np.ndarray): evaluation of the interpolator at x point(s)

    Raises:
        ValueError if invalid index is passed
        ValueError if order not implemented
    """
    order = params['order']
    index = params['index']
    h = params['h']

    if index > order + 1 or index < 1:
        raise ValueError('The index provided is not valid')

    if order == 1:
        if index == 1:
            f_x = 1 - x / h
        elif index == 2:
            f_x = x / h
        else:
            raise ValueError('The index provided is not valid')
    elif order == 2:
        if index == 1:
            f_x = 1 - 3 * x / h + 2 * x ** 2 / h ** 2
        elif index == 2:
            f_x = 4 * x / h * (1 - x / h)
        elif index == 3:
            f_x = 2 * x ** 2 / h ** 2 - x / h
        else:
            raise ValueError('The index provided is not valid')
    else:
        raise ValueError('Local interpolator has not been implemented up to required order')

    return f_x


def local_interpolator_derivative(x, params):
    """
    Local Interpolating functions 1st derivative

    Args:
        x (float/np.ndarray): independent variable
        params (dict): dictionary of parameters:
          - order (int): order of the local interpolator
          - index (int): index (between 1 and order+1)
          - h (float): width of the element

    Returns:
        df_x (float/np.ndarray): evaluation of df(x)

    Raises:
        ValueError if invalid index is passed
        ValueError if order not implemented
    """
    order = params['order']
    index = params['index']
    h = params['h']

    if index > order + 1 or index < 1:
        raise ValueError('The index provided is not valid')

    if order == 1:
        if index == 1:
            df_x = -np.ones(shape=x.shape) / h
        elif index == 2:
            df_x = np.ones(shape=x.shape) / h
        else:
            raise ValueError('The index provided is not valid')
    elif order == 2:
        if index == 1:
            df_x = -3 / h + 4 * x / h ** 2
        elif index == 2:
            df_x = 4 / h - 8 * x / h ** 2
        elif index == 3:
            df_x = 4 * x / h ** 2 - 1 / h
        else:
            raise ValueError('The index provided is not valid')
    else:
        raise ValueError('Local interpolator has not been implemented up to required order')

    return df_x


def local_fe_matrix_numerical(alpha2, h, order):
    """
    Builds the system associated to the application of FEM method to the ODE
    d2u/dx2 + alpha^2 * u = f(x)
    computes numerically the involved integrals

    Args:
        alpha2 (float): parameter involved in the ODE to be solved (alpha^2)
        h (float): size of the element
        order (int): order of the interpolators used

    Returns:
        A_local (np.ndarray): matrix of coefficients of the element in local coords
    """

    # Initialize vector for numerical integration
    x_local = np.linspace(0, h, 10001)

    # Initialize matrices
    A_local = np.zeros((order+1, order+1))

    for ii in range(order+1):
        params_ii = {'order': order,
                     'index': ii+1,
                     'h': h}
        phi_ii = local_interpolator(x_local, params_ii)
        dphi_ii = local_interpolator_derivative(x_local, params_ii)

        # A local:
        # - diagonal elements
        A_local[ii, ii] = integrate_trapezoidal(x=x_local, fx=dphi_ii**2, lims=(0, h)) - \
            alpha2 * integrate_trapezoidal(x=x_local, fx=phi_ii**2, lims=(0, h))

        # - out of diagonal elements
        for jj in range(ii):
            params_jj = {'order': order,
                         'index': jj+1,
                         'h': h}
            phi_jj = local_interpolator(x_local, params_jj)
            dphi_jj = local_interpolator_derivative(x_local, params_jj)
            A_local[ii, jj] = integrate_trapezoidal(x=x_local, fx=dphi_ii * dphi_jj, lims=(0, h)) - \
                alpha2 * integrate_trapezoidal(x=x_local, fx=phi_ii * phi_jj, lims=(0, h))
            A_local[jj, ii] = A_local[ii, jj]

    return A_local


def local_fe_force_numerical(f, lims, order):
    """
    Builds the system associated to the application of FEM method to the ODE
    d2u/dx2 + alpha^2 * u = f(x)
    computes numerically the involved integrals

    Args:
        f (function): right hand side of the ODE: f(x, params)
        lims (list/tuple): limits of the element in global coords
        order (int): order of the interpolators used

    Returns:
        A_local (np.ndarray): matrix of coefficients of the element in local coords
    """
    h = lims[-1] - lims[0]

    # Initialize vector for numerical integration
    x_local = np.linspace(0, h, 10001)
    x_global = np.linspace(lims[0], lims[-1], 10001)
    f_x = f(x_global, params={})

    # Initialize array
    F_local = np.zeros((order+1, 1))

    for ii in range(order+1):
        params_ii = {'order': order,
                     'index': ii+1,
                     'h': h}
        phi_ii = local_interpolator(x_local, params_ii)

        # F local
        F_local[ii] = -integrate_trapezoidal(x=x_local, fx=phi_ii*f_x, lims=(0, h))

    return F_local


def local_fe_matrix_analytical(alpha2, h, order):
    """
    Builds the system associated to the application of FEM method to the ODE
    d2u/dx2 + alpha^2 * u = f(x)
    previously solved analytically

    Args:
        alpha2 (float): parameter involved in the ODE to be solved (alpha^2)
        h (float): size of the element
        order (int): order of the interpolators used

    Returns:
        A_local (np.ndarray): matrix of coefficients of the element in local coords

    Raises:
        ValueError if order not implemented
    """

    # Initialize
    A_local = np.zeros((order+1, order+1))

    if order == 1:
        # A
        # - Diagonal
        A_local[0, 0] = 1/h - alpha2 * h/3
        A_local[1, 1] = 1/h - alpha2 * h/3
        # - Out of diagonal
        A_local[1, 0] = -1/h - alpha2 * h/6
        A_local[0, 1] = -1/h - alpha2 * h/6

    elif order == 2:
        # A
        # - Diagonal
        A_local[0, 0] = 7/(3*h) - alpha2 * 2*h/15
        A_local[1, 1] = 16/(3*h) - alpha2 * 8*h/15
        A_local[2, 2] = 7/(3*h) - alpha2 * 2*h/15
        # - Out of diagonal
        A_local[1, 0] = -8/(3*h) - alpha2 * h/15
        A_local[0, 1] = -8/(3*h) - alpha2 * h/15
        A_local[2, 0] = 1/(3*h) + alpha2 * h/30
        A_local[0, 2] = 1/(3*h) + alpha2 * h/30
        A_local[2, 1] = -8/(3*h) - alpha2 * h/15
        A_local[1, 2] = -8/(3*h) - alpha2 * h/15

    else:
        raise ValueError('Analytical local matrix and force vector are not implemented for the required order')

    return A_local


def local_fe_force_analytical(f, lims, order):
    """
    Builds the system associated to the application of FEM method to the ODE
    d2u/dx2 + alpha^2 * u = f(x)
    computes numerically the involved integrals

    Args:
        f (function): right hand side of the ODE: f(x, params)
        lims (list/tuple): limits of the element in global coords
        order (int): order of the interpolators used

    Returns:
        A_local (np.ndarray): matrix of coefficients of the element in local coords
    """
    h = lims[-1] - lims[0]

    # Extract averaged force in the element
    x_global = np.linspace(lims[0], lims[-1], 10001)
    f_x = f(x_global, params={})

    f_avg = integrate_trapezoidal(x_global, f_x, lims) / h

    # Initialize array
    F_local = np.zeros((order+1, 1))

    if order == 1:
        F_local[0] = -h/2 * f_avg
        F_local[1] = -h/2 * f_avg

    else:
        raise ValueError('Analytical local matrix and force vector are not implemented for the required order')

    return F_local


def finite_elements(alpha2, f, num_elem, order, lims, analytical=False):
    """
    Builds the system associated to the application of FEM method to the ODE
    d2u/dx2 + alpha^2 * u = f(x)
    computes numerically the involved integrals

    Args:
        alpha2 (float): parameter involved in the ODE to be solved (alpha^2)
        f (function): right hand side of the ODE: f(x, params)
        num_elem (int): number of finite elements to divide the interval
        order (int): order of the interpolators used
        lims (tuple/list): limits of the domain where the ODE is to be solved
        analytical (bool): whether to use analytical (T) or numerical (F)
                           approach for building the system (default=False)

    Returns:
        A (np.ndarray): matrix of coefficients which allows to solve the ODE
        F (np.ndarray): vector involved in the system generated
        x_i (np.ndarray): locations of points where the solution would be obtained

        A * u(x_i) = F
        system can be solved to provide the solution approximation given by u(x_i)
    """

    # Points where u is evaluated when applying the method
    x_i = np.linspace(lims[0], lims[-1], order*num_elem+1)

    # Interval size
    h = (lims[-1] - lims[0]) / num_elem

    # Get local system of an individual element
    if analytical:
        try:
            A_loc = local_fe_matrix_analytical(alpha2=alpha2, h=h, order=order)
        except ValueError:
            print('Order required not implemented analytically, solving using numerical methods')
            A_loc = local_fe_matrix_numerical(alpha2=alpha2, h=h, order=order)
    else:
        A_loc = local_fe_matrix_numerical(alpha2=alpha2, h=h, order=order)

    # Build global matrix & vector
    A = np.zeros((order*num_elem + 1, order*num_elem + 1))
    F = np.zeros((order*num_elem + 1, 1))

    # Sum over elements
    for ee in range(num_elem):
        # Generate mapping of current element
        mapping = np.zeros((order + 1, order*num_elem + 1))
        for ii in range(order+1):
            mapping[ii, ee*order + ii] = 1

        # Compute local force
        elem_lims = (lims[0] + ee*h, lims[0] + (ee+1)*h)
        if analytical:
            try:
                F_loc = local_fe_force_analytical(f=f, lims=elem_lims, order=order)
            except ValueError:
                print('Order required not implemented analytically, solving using numerical methods')
                F_loc = local_fe_force_numerical(f=f, lims=elem_lims, order=order)
        else:
            F_loc = local_fe_force_numerical(f=f, lims=elem_lims, order=order)

        # Add contribution to global system
        A += np.matmul(mapping.T, np.matmul(A_loc, mapping))
        F += np.matmul(mapping.T, F_loc)

    return A, F, x_i


