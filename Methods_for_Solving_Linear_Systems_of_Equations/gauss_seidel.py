'''import numpy as np
from numpy.linalg import norm

from colors import bcolors
from matrix_utility import is_diagonally_dominant


def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == '__main__':

    A = np.array([[3, -1, 1], [0, 1, -1], [1, 1, -2]])
    b = np.array([4, -1, -3])
    X0 = np.zeros_like(b)

    solution =gauss_seidel(A, b, X0)
    print(bcolors.OKBLUE,"\nApproximate solution:", solution)


'''




'''  AFTER REFACTOR'''



import numpy as np
from numpy.linalg import norm
from colors import bcolors
from matrix_utility import is_diagonally_dominant


class IterativeSolver:
    """Generic class for iterative solvers like Gauss-Seidel."""

    def __init__(self, A, b, X0=None, tol=1e-16, max_iter=200):
        """
        Initializes the solver.
        :param A: Coefficient matrix
        :param b: Right-hand side vector
        :param X0: Initial guess (defaults to zeros)
        :param tol: Convergence tolerance
        :param max_iter: Maximum number of iterations
        """
        self.A = A
        self.b = b
        self.n = len(A)
        self.tol = tol
        self.max_iter = max_iter
        self.X0 = np.zeros_like(b, dtype=np.double) if X0 is None else X0

    def solve(self):
        """Placeholder method to be implemented in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class GaussSeidelSolver(IterativeSolver):
    """Implements the Gauss-Seidel iterative method."""

    def solve(self):
        """Solves Ax = b using the Gauss-Seidel method."""
        x = np.zeros(self.n, dtype=np.double)

        if is_diagonally_dominant(self.A):
            print('Matrix is diagonally dominant - performing Gauss-Seidel algorithm\n')

        print("Iteration" + "\t\t".join([f" {var:>12}" for var in [f"x{i+1}" for i in range(self.n)]]))
        print("-" * 80)

        for k in range(1, self.max_iter + 1):
            for i in range(self.n):
                sigma = sum(self.A[i][j] * x[j] for j in range(self.n) if j != i)
                x[i] = (self.b[i] - sigma) / self.A[i][i]

            print(f"{k:<10} " + "\t\t".join(["{:<12.6f}".format(val) for val in x]))

            # Check for convergence
            if norm(x - self.X0, np.inf) < self.tol:
                print("\nConverged successfully!")
                return tuple(x)

            self.X0 = x.copy()

        print("\n Solving Problem 2 using Gauss-Seidel Method: ")
        print("\nMaximum number of iterations exceeded, solution may not have converged.")
        return tuple(x)


if __name__ == '__main__':
    #  Problem 2: System Ax = b from the PDF
    A = np.array([[-1, -1, 2], [2, -1, 1], [2, 2, 2]], dtype=np.double)
    b = np.array([-5, -1, 4], dtype=np.double)
    X0 = np.zeros_like(b, dtype=np.double)

    print("\n Solving Problem 2 using Gauss-Seidel Method...")

    #  Use Gauss-Seidel solver
    solver = GaussSeidelSolver(A, b, X0)
    solution = solver.solve()

    #  Extract and print value of 'c'
    c_value = solution[2]

    print(bcolors.OKBLUE, f"\n Approximate value of c: {c_value}", bcolors.ENDC)
