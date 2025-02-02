'''
import jacobi_utilities
from sympy import *

x = Symbol('x')


def natural_cubic_spline(f, x0):
    h = list()
    for i in range(len(f) - 1):
        h.append(f[i + 1][0] - f[i][0])

    g = list()
    g.append(0)  # g0
    for i in range(1, len(f) - 1):
        g.append(h[i] / (h[i] + h[i - 1]))
    g.append(0)  # gn

    m = list()
    m.append(0)
    for i in range(1, len(f)):
        m.append(1 - g[i])

    d = list()
    d.append(0)  # d0=0
    for i in range(1, len(f) - 1):
        d.append((6 / (h[i - 1] + h[i])) * (((f[i + 1][1] - f[i][1]) / h[i]) - ((f[i][1] - f[i - 1][1]) / h[i - 1])))
    d.append(0)  # dn

    # building the matrix
    mat = list()

    # first row
    mat.append(list())
    mat[0].append(2)
    for j in range(len(f) - 1):
        mat[0].append(0)

    for i in range(1, len(f) - 1):
        mat.append(list())
        for j in range(len(f)):
            if j == i - 1:  # put miu
                mat[i].append(m[i])
            elif j == i:
                mat[i].append(2)
            elif j == i + 1:  # put lambda
                mat[i].append(g[i])
            else:
                mat[i].append(0)

    # last row
    mat.append(list())
    for j in range(len(f) - 1):
        mat[len(f) - 1].append(0)
    mat[len(f) - 1].append(2)

    print("matrix: " + str(mat))
    print("vector b: " + str(d))

    # get m vector
    print("\nJacobi middle results: ")
    M = (jacobi_utilities.Jacobi(mat, d))
    print("\nvector M: " + str(list(map(float, M))))

    # find S:
    for loc in range(1, len(f)):
        s = (((f[loc][0] - x) ** 3) * M[loc - 1] + ((x - f[loc - 1][0]) ** 3) * M[loc]) / (6 * h[loc - 1])
        s += (((f[loc][0] - x) * f[loc - 1][1]) + ((x - f[loc - 1][0]) * f[loc][1])) / h[loc - 1]
        s -= (((f[loc][0] - x) * M[loc - 1] + (x - f[loc - 1][0]) * M[loc]) * h[loc - 1]) / 6
        print("s" + str(loc - 1) + "(x) = " + str(s))

    # find the location of x0:
    loc = 0
    for i in range(1, len(f)):
        if x0 < f[i][0] and x0 > f[i - 1][0]:
            loc = i
            break

    if loc == 0:
        print("no range found for x0")
        return

    s = (((f[loc][0] - x) ** 3) * M[loc - 1] + ((x - f[loc - 1][0]) ** 3) * M[loc]) / (6 * h[loc - 1])
    s += (((f[loc][0] - x) * f[loc - 1][1]) + ((x - f[loc - 1][0]) * f[loc][1])) / h[loc - 1]
    s -= (((f[loc][0] - x) * M[loc - 1] + (x - f[loc - 1][0]) * M[loc]) * h[loc - 1]) / 6

    print("\nx0 between f(x" + str(loc - 1) + ") = " + str(f[loc - 1][0]) + " and f(x" + str(loc) + ") = " + str(
        f[loc][0]) + " so:")
    print("s" + str(loc - 1) + "(" + str(x0) + ") = " + str(float(s.subs(x, x0))))


if __name__ == '__main__':
    f = [(1, 1), (2, 2), (3, 1), (4, 1.5), (5, 1)]
    x0 = 6

    print("func: " + str(f))
    print("x0 = " + str(x0) + "\n")
    natural_cubic_spline(f, x0)

'''




'''  AFTER REFACTOR'''


import numpy as np
from sympy import Symbol
from Interpolation_and_Polynomial_Approximation import jacobi_utilities
from colors import bcolors


class TridiagonalMatrixSolver:
    """Solves a tridiagonal matrix system using the Jacobi method."""

    @staticmethod
    def solve(matrix, vector):
        """Solves Ax = b using the Jacobi method."""
        print("\nSolving the tridiagonal system using Jacobi method...")
        solution = jacobi_utilities.Jacobi(matrix, vector)
        return list(map(float, solution))  # Convert SymPy values to floats


class CubicSplineInterpolator:
    """Computes the natural cubic spline interpolation for given points."""

    def __init__(self, points):
        """
        Initialize with the given points.

        :param points: List of tuples (x_i, y_i) representing known data points.
        """
        self.points = points
        self.x = Symbol('x')
        self.h = self._compute_h()
        self.g, self.m, self.d = self._compute_coefficients()
        self.matrix, self.b_vector = self._build_matrix()

    def _compute_h(self):
        """Computes step sizes (h) between adjacent points."""
        return [self.points[i + 1][0] - self.points[i][0] for i in range(len(self.points) - 1)]

    def _compute_coefficients(self):
        """Computes lambda (g), mu (m), and d vectors for the cubic spline equations."""
        g, m, d = [0], [0], [0]

        for i in range(1, len(self.points) - 1):
            g.append(self.h[i] / (self.h[i] + self.h[i - 1]))
            m.append(1 - g[i])
            d.append(
                (6 / (self.h[i - 1] + self.h[i])) *
                (((self.points[i + 1][1] - self.points[i][1]) / self.h[i]) -
                 ((self.points[i][1] - self.points[i - 1][1]) / self.h[i - 1]))
            )

        g.append(0)
        m.append(0)
        d.append(0)
        return g, m, d

    def _build_matrix(self):
        """Builds the tridiagonal matrix for solving the system."""
        n = len(self.points)
        mat = [[0] * n for _ in range(n)]
        mat[0][0] = 2
        mat[-1][-1] = 2

        for i in range(1, n - 1):
            mat[i][i - 1] = self.m[i]
            mat[i][i] = 2
            mat[i][i + 1] = self.g[i]

        return mat, self.d

    def compute_spline(self, x0):
        """Computes and evaluates the cubic spline at a given x0."""
        print("Matrix:", self.matrix)
        print("Vector b:", self.b_vector)

        M = TridiagonalMatrixSolver.solve(self.matrix, self.b_vector)

        # Compute spline functions
        splines = self._compute_splines(M)

        # Find the correct segment for x0
        segment = self._find_segment(x0)
        if segment is None:
            print(" No valid range found for x0!")
            return None

        # Evaluate the spline function at x0
        s_x0 = splines[segment].subs(self.x, x0)
        print(f"\n x0 lies between f(x{segment}) = {self.points[segment][0]} and f(x{segment + 1}) = {self.points[segment + 1][0]}:")
        print(f" s{segment}({x0}) = {float(s_x0)}")

        return float(s_x0)

    def _compute_splines(self, M):
        """Computes the spline functions for each interval."""
        splines = []
        for i in range(1, len(self.points)):
            s = (((self.points[i][0] - self.x) ** 3) * M[i - 1] + ((self.x - self.points[i - 1][0]) ** 3) * M[i]) / (6 * self.h[i - 1])
            s += (((self.points[i][0] - self.x) * self.points[i - 1][1]) + ((self.x - self.points[i - 1][0]) * self.points[i][1])) / self.h[i - 1]
            s -= (((self.points[i][0] - self.x) * M[i - 1] + (self.x - self.points[i - 1][0]) * M[i]) * self.h[i - 1]) / 6
            splines.append(s)
        return splines

    def _find_segment(self, x0):
        """Finds the interval [xi, xi+1] where x0 is located."""
        for i in range(1, len(self.points)):
            if self.points[i - 1][0] <= x0 <= self.points[i][0]:
                return i - 1
        return None


if __name__ == '__main__':
    #  Problem 3: Given data points
    table_points = [
        (0.35, -213.5991), (0.4, -204.4416), (0.55, -194.9375),
        (0.65, -185.0256), (0.7, -174.6711), (0.85, -163.8656), (0.9, -152.6271)
    ]
    x = 0.75  # The point we need to approximate

    print(bcolors.OKBLUE, "----------------- Cubic Spline Interpolation -----------------\n", bcolors.ENDC)
    print(bcolors.OKBLUE, "Table Points:", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation for x =", bcolors.ENDC, x, '\n')

    spline = CubicSplineInterpolator(table_points)
    result_spline = spline.compute_spline(x)

    print(bcolors.OKBLUE, "\n------------------------------------------------------------\n", bcolors.ENDC)


