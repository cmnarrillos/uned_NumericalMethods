import os
import numpy as np
import matplotlib.pyplot as plt

from task_2_functions import qr_method, inverse_power_method, power_method


if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')
if not os.path.exists('./results/'):
    os.makedirs('./results/')

# General parameters of the program
tol = 1e-9
max_iter = 10000
max_tries_inv_pow = 10
dist_btw_eigvals = 1e-4

# Size of the matrix
N = 10

# List of lambda values to test:
lambda_list = np.linspace(0, 1, 21)
# Remove the first and last elements (0 and 1 are not usable)
lambda_list = lambda_list[1:-1]

for lambda_ in lambda_list:
    # Open file and write preamble
    file = open(f'./results/result_lambda_{lambda_}.txt', 'w')
    print('--------------------------------------------------------------------------------', file=file)
    print(f'lambda = {lambda_}', file=file)
    print('--------------------------------------------------------------------------------', file=file)
    print(file=file)
    # Print in Console too
    print()
    print(f'lambda = {lambda_}')
    print()

    # Create the matrix
    A = np.eye(N)
    for ii in range(N):
        A[ii, ii] = 1-2*lambda_
        if ii > 0:
            A[ii, ii-1] = lambda_
            A[ii-1, ii] = lambda_

    # Document matrix created
    print(file=file)
    print('A = ', file=file)
    print(A, file=file)

    # Extract the maximum eigenvalue with the power method
    max_eigval, eigvec = power_method(A, tolerance=tol, max_iterations=max_iter)

    # Document power method
    print(file=file)
    print('Power method result:', file=file)
    print(f'maximum eigenvalue = {max_eigval}', file=file)
    print(f'associated eigenvector = {eigvec}', file=file)
    # Print in Console too
    print()
    print('Power method result:')
    print(f'maximum eigenvalue = {max_eigval}')

    # Extract all the eigenvalues with the Q-R method
    eigenvalues_QR = qr_method(A, tolerance=tol, max_iterations=max_iter)

    # Document QR method
    print(file=file)
    print('Q-R method result:', file=file)
    print(f'List of eigenvalues {eigenvalues_QR}', file=file)
    # Print in Console too
    print()
    print('Q-R method result:')
    print(f'List of eigenvalues {eigenvalues_QR}')

    # Get the Orthonormal matrix formed by the eigenvectors: Q / D = Q^T * A * Q
    for iteration in range(max_tries_inv_pow):
        q_matrix = []
        eigenvalues_iPower = []
        print(file=file)
        print('Inverse power method applied to each eigenvalue:', file=file)
        # Print in Console too
        print()
        print('Inverse power method applied to each eigenvalue:')
        restart = False

        # Initialize for each eigenvalue found by QR method
        for eigenvalue_qr in eigenvalues_QR:
            for ii in range(max_tries_inv_pow):
                # Apply inverse power method to find eigenvector using eigval extracted with QR method as initial est
                eigenvalue_ip, eigenvector = inverse_power_method(A, eigenvalue_qr,
                                                                  tolerance=tol, max_iterations=max_iter)
                # Try again if eigenvalue found was already stored
                if all(abs(eigenvalue_ip - existing_eigenvalue) > dist_btw_eigvals
                       for existing_eigenvalue in eigenvalues_iPower):
                    break  # Stop the loop if the condition is met: the found eigenvalue is valid

            # If was not possible finding all the N eigenvalues after max_tries_inv_pow, repeat from the beginning
            if ii == max_tries_inv_pow-1:
                print('Failed to find all the different eigenvalues with inverse power', file=file)
                print('Let\'s try again from the beginning', file=file)
                print('Failed to find all the different eigenvalues with inverse power')
                print('Let\'s try again from the beginning')
                restart = True
                break

            # Document inverse power method in result.txt file
            print(f'eigenvalue = {eigenvalue_ip}', file=file)
            print(f'associated eigenvector = {eigenvector}', file=file)

            # Include found eigenvector to q matrix and eigenvalue_ip to eigenvalues_iPower list
            q_matrix.append(eigenvector)
            eigenvalues_iPower.append(eigenvalue_ip)

        if not restart:
            # If the method has been completed, transform the list of eigenvectors to Q matrix
            if iteration < max_tries_inv_pow-1:
                q_matrix = np.array(q_matrix).T
                break

    if restart:
        # Not capable of finding all N different eigenvalues of the matrix
        # avoid errors due to shape mixmatch
        print(file=file)
        print('It was not possible finding the Q matrix', file=file)
        print()
        print('It was not possible finding the Q matrix')
        file.close()
        continue

    # Export Q to file
    print(file=file)
    print('Q = ', file=file)
    print(q_matrix, file=file)

    # Verify Q is orthogonal
    print(file=file)
    print('Q * Q^T = ', file=file)
    print(np.dot(q_matrix, q_matrix.T), file=file)

    print(file=file)
    print('max(abs( Q * Q^T - I ) ) = ', file=file)
    print(np.max(np.abs(np.dot(q_matrix, q_matrix.T) - np.eye(N))), file=file)
    # Print in Console too
    print()
    print('max(abs( Q * Q^T - I ) ) = ')
    print(np.max(np.abs(np.dot(q_matrix, q_matrix.T) - np.eye(N))))

    # Verify that Q^T * A * Q provides the diagonal matrix with eigenvalues in main diag
    print(file=file)
    print('Q^T * A * Q = ', file=file)
    print(np.dot(q_matrix.T, np.dot(A, q_matrix)), file=file)

    print(file=file)
    print('max(abs( Q^T * A * Q - Diag(lambda) ) ) = ', file=file)
    print(np.max(np.abs(np.dot(q_matrix.T, np.dot(A, q_matrix)) -
                        np.diag(eigenvalues_iPower))), file=file)
    # Print in Console too
    print()
    print('max(abs( Q^T * A * Q - Diag(lambda) ) ) = ')
    print(np.max(np.abs(np.dot(q_matrix.T, np.dot(A, q_matrix)) -
                        np.diag(eigenvalues_iPower))))

    # Close file
    file.close()

    # Plotting the results

    # Get and plot Gerschgorin circles
    theta = np.linspace(0, 2*np.pi, 100)
    x_1 = 1-2*lambda_ + lambda_*np.cos(theta)
    y_1 = lambda_*np.sin(theta)
    x_2 = 1-2*lambda_ + 2*lambda_*np.cos(theta)
    y_2 = 2*lambda_*np.sin(theta)

    plt.figure()
    # Plot axes
    plt.plot((-4, 2), (0, 0), 'k-.')
    plt.plot((0, 0), (-2.5, 2.5), 'k-.')
    # Plot Gerschgorin circles
    plt.plot(x_1, y_1, 'k-', label='Gerschgorin circle')
    plt.plot(x_2, y_2, 'k-')
    plt.plot(1-2*lambda_, 0, '+k')
    # Plot eigenvalues obtained with QR method
    plt.plot(eigenvalues_QR[0], 0, 'og', label='QR')
    for eigval in eigenvalues_QR[1:]:
        plt.plot(eigval, 0, 'og')
    # Plot eigenvalues obtained with inverse Power method
    plt.plot(eigenvalues_iPower[0], 0, 'xr', label='inverse Power')
    for eigval in eigenvalues_iPower[1:]:
        plt.plot(eigval, 0, 'xr')
    plt.plot(max_eigval, 0, '+b', label='Power method (max eigval)')

    plt.grid(which='both')
    plt.xlabel('$\mathcal{Re}(\lambda)$', fontsize=18)
    plt.ylabel('$\mathcal{Im}(\lambda)$', fontsize=18)
    plt.legend(fontsize=14)

    plt.xlim((1-5*lambda_, 1+lambda_))
    plt.ylim((-2.5*lambda_, 2.5*lambda_))
    plt.savefig(f'./Figures/Gerschgorin_circle_lambda_{lambda_}.png', bbox_inches='tight')

    plt.xlim((-4, 2))
    plt.ylim((-2.5, 2.5))
    plt.savefig(f'./Figures/Gerschgorin_circle_lambda_{lambda_}_grl.png', bbox_inches='tight')

    plt.close()
