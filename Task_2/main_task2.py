import numpy as np

from task_2_functions import qr_method, inverse_power_method, power_method


# Size of the matrix
N = 10

# First version: lambda = 1/5
lambda_ = 0.2
# Second version: lambda = 4/5
lambda_ = 0.8

# Create the matrix
A = np.eye(N)
for ii in range(N):
    A[ii, ii] = 1-2*lambda_
    if ii > 0:
        A[ii, ii-1] = lambda_
        A[ii-1, ii] = lambda_

# Extract the maximum eigenvalue with the power method
max_eigval, eigvec = power_method(A)
print()
print('Power method result:')
print(f'maximum eigenvalue = {max_eigval}')
print(f'associated eigenvector = {eigvec}')

# Extract all the eigenvalues with the Q-R method
eigvals = qr_method(A)

print()
print('Q-R method result:')
print(f'List of eigenvalues {eigvals}')

# Get the
q_matrix = []
print()
print('Inverse power method applied to each eigenvalue:')
for eigenvalue in eigvals:
    eigenvalue, eigenvector = inverse_power_method(A, eigenvalue + 1e-6)
    print(f'eigenvalue = {eigenvalue}')
    print(f'associated eigenvector = {eigenvector}')
    q_matrix.append(eigenvector)

q_matrix = np.array(q_matrix).T

print()
print('Q = ')
print(q_matrix)

print()
print('Q^T * A * Q = ')
print(np.dot(q_matrix.T, np.dot(A, q_matrix)))
