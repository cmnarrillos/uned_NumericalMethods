#  Code used for "Advanced Numerical Methods" subject at UNED

This repository contains code samples for the different
tasks made associated to the subject ["Metodos Numericos Avanzados"](https://portal.uned.es/portal/page?_pageid=93,71759483&_dad=portal&_schema=PORTAL&idAsignatura=21156030&idTitulacion=215801)
at Universidad Nacional de Educación a Distancia (UNED) 
for the academic year 2023-2024.



Inside this repository the source code needed to perform the following tasks can be found:

- Task_1: Solving boundary condition problems for ODEs using both finite differences and shooting method
- Task_2: Implementation and application of iterative methods to extract eigenvalues and eigenvectors of a matrix: power, inverse power and QR methods
- Task_3: Solving elliptic PDEs using finite differences in a semicircular domain (using polar coordinates or a boundary-adjustment method with cartesian coords). Different iterative methods have been included to solve the linear system (Jacobi, G-S, SOR)
- Task_4: Solving hyperbolic and parabolic PDEs using 1st (Euler and backwards differentiation) and 2nd order algorithms (Crank-Nicolson)
- Task_5: Solving 2nd order ODE using FEM. Both linear and quadratic local interpolators have been implemented for FEM method.

Also a file (`aux_functions.py`) with auxiliary user defined functions is common all the tasks. Some of the functions which can be found there are:
- rk4: 4th order Runge-Kutta method
- newton_method: Newton-Raphson root finder
- lu_decomposition: factorization of a matrix into L(lower triangular) and U (upper triangular) factors
- lu_solve: solves a linear system given the L, U factors of the matrix A in Ax=b
- solve_linear_system_with_lu_decomposition: linear system solver
- newton_method_vect: extension of Newton-Raphson root finder to a vectorial function
- remove_zero_rows_columns
- calculate_norm: quadratic norm of a vector
- jacobi_method: linear system solver
- gauss_seidel: linear system solver
- sor_method: linear system solver
- inverse_lower_triangular
- check_tridiagonal
- crout_tridiagonal_solver
- dirac_delta
- integrate_trapezoidal