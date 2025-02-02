'''
from colors import bcolors
from matrix_utility import *


def GaussJordanElimination(matrix, vector):
    """
    Function for solving a linear equation using gauss's elimination method
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=A(-1)b
    """
    # Pivoting process
    matrix, vector = RowXchange(matrix, vector)
    # Inverse matrix calculation
    invert = InverseMatrix(matrix, vector)
    return MulMatrixVector(invert, vector)




def UMatrix(matrix,vector):
    """
    :param matrix: Matrix nxn
    :return:Disassembly into a  U matrix
    """
    # result matrix initialized as singularity matrix
    U = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i])/matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    # U matrix is a doubling of elementary matrices that we used to reset organs under the pivot
    U = MultiplyMatrix(U, matrix)
    return U


def LMatrix(matrix, vector):
    """
       :param matrix: Matrix nxn
       :return:Disassembly into a  L matrix
       """
    # Initialize the result matrix
    L = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i])/matrix[i][i]
            # L matrix is a doubling of inverse elementary matrices
            L[j][i] = (matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)

    return L


def SolveLU(matrix, vector):
    """
    Function for deconstructing a linear equation by ungrouping LU
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=U(-1)L(-1)b
    """
    matrixU = UMatrix(matrix)
    matrixL = LMatrix(matrix)
    return MultiplyMatrix(InverseMatrix(matrixU), MultiplyMatrix(InverseMatrix(matrixL), vector))


def solveMatrix(matrixA,vectorb):
    detA = Determinant(matrixA, 1)
    print(bcolors.YELLOW, "\nDET(A) = ", detA)

    if detA != 0:
        print("CondA = ", Cond(matrixA, InverseMatrix(matrixA, vectorb)), bcolors.ENDC)
        print(bcolors.OKBLUE, "\nnon-Singular Matrix - Perform GaussJordanElimination",bcolors.ENDC)
        result = GaussJordanElimination(matrixA, vectorb)
        print(np.array(result))
        return result
    else:
        print("Singular Matrix - Perform LU Decomposition\n")
        print("Matrix U: \n")
        print(np.array(UMatrix(matrixA, vectorb)))
        print("\nMatrix L: \n")
        print(np.array(LMatrix(matrixA, vectorb)))
        print("\nMatrix A=LU: \n")
        result = MultiplyMatrix(LMatrix(matrixA, vectorb), UMatrix(matrixA, vectorb))
        print(np.array(result))
        return result


def polynomialInterpolation(table_points, x):
    matrix = [[point[0] ** i for i in range(len(table_points))] for point in table_points] # Makes the initial matrix

    b = [[point[1]] for point in table_points]

    print(bcolors.OKBLUE, "The matrix obtained from the points: ", bcolors.ENDC,'\n', np.array(matrix))
    print(bcolors.OKBLUE, "\nb vector: ", bcolors.ENDC,'\n',np.array(b))
    matrixSol = solveMatrix(matrix, b)

    result = sum([matrixSol[i][0] * (x ** i) for i in range(len(matrixSol))])
    print(bcolors.OKBLUE, "\nThe polynom:", bcolors.ENDC)
    print('P(X) = '+'+'.join([ '('+str(matrixSol[i][0])+') * x^' + str(i) + ' ' for i in range(len(matrixSol))])  )
    print(bcolors.OKGREEN, f"\nThe Result of P(X={x}) is:", bcolors.ENDC)
    print(result)
    return result


if __name__ == '__main__':

    table_points = [(0, 0), (1, 0.8415), (2, 0.9093), (3, 0.1411), (4, -0.7568), (5, -0.9589), (6, -0.2794)]
    x = 1.28
    print(bcolors.OKBLUE, "----------------- Interpolation & Extrapolation Methods -----------------\n", bcolors.ENDC)
    print(bcolors.OKBLUE, "Table Points: ", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation to the point: ", bcolors.ENDC, x,'\n')
    polynomialInterpolation(table_points, x)
    print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n", bcolors.ENDC)

'''




'''  AFTER REFACTOR'''



import numpy as np
from colors import bcolors
from matrix_utility import *


class GaussJordanSolver:
    """Handles solving a system of equations using Gauss-Jordan Elimination."""

    @staticmethod
    def solve(matrix, vector):
        """Solves Ax = b using Gauss-Jordan elimination."""
        matrix, vector = RowXchange(matrix, vector)
        inverse_matrix = InverseMatrix(matrix, vector)
        return MulMatrixVector(inverse_matrix, vector)


class LUSolver:
    """Handles solving Ax = b using LU decomposition."""

    @staticmethod
    def solve(matrix, vector):
        """Performs LU decomposition and solves Ax = b."""
        U = MakeIMatrix(len(matrix), len(matrix))
        for i in range(len(matrix[0])):
            matrix, vector = RowXchageZero(matrix, vector)
            for j in range(i + 1, len(matrix)):
                elementary = MakeIMatrix(len(matrix[0]), len(matrix))
                elementary[j][i] = -matrix[j][i] / matrix[i][i]
                matrix = MultiplyMatrix(elementary, matrix)
        return matrix


class LinearSystemSolver:
    """Selects the best method to solve a linear system."""

    @staticmethod
    def solve(A, b):
        detA = Determinant(A, 1)
        print(bcolors.YELLOW, "\nDET(A) =", detA)

        if detA != 0:
            print("CondA =", Cond(A, InverseMatrix(A, b)), bcolors.ENDC)
            print(bcolors.OKBLUE, "\nNon-Singular Matrix - Performing Gauss-Jordan Elimination", bcolors.ENDC)
            return GaussJordanSolver.solve(A, b)
        else:
            print("Singular Matrix - Performing LU Decomposition\n")
            return LUSolver.solve(A, b)


class PolynomialInterpolator:
    """Handles polynomial interpolation based on given points."""

    def __init__(self, table_points):
        """Initialize with a set of points."""
        self.table_points = table_points

    def interpolate(self, x):
        """Interpolates a polynomial and approximates the value at x."""
        matrix = [[point[0] ** i for i in range(len(self.table_points))] for point in self.table_points]
        b = [[point[1]] for point in self.table_points]

        print(bcolors.OKBLUE, "Matrix from points:", bcolors.ENDC, '\n', np.array(matrix))
        print(bcolors.OKBLUE, "\nb vector:", bcolors.ENDC, '\n', np.array(b))

        matrix_sol = LinearSystemSolver.solve(matrix, b)
        result = sum(matrix_sol[i][0] * (x ** i) for i in range(len(matrix_sol)))

        print(bcolors.OKBLUE, "\nThe polynomial:", bcolors.ENDC)
        print('P(X) = ' + ' + '.join([f'({matrix_sol[i][0]}) * x^{i}' for i in range(len(matrix_sol))]))
        print(bcolors.OKGREEN, f"\nResult of P(X={x}):", bcolors.ENDC)
        print(result)

        return result


if __name__ == '__main__':
    #  Problem 3: Given data points
    table_points = [
        (0.35, -213.5991), (0.4, -204.4416), (0.55, -194.9375),
        (0.65, -185.0256), (0.7, -174.6711), (0.85, -163.8656), (0.9, -152.6271)
    ]
    x = 0.75  # The point we need to approximate

    print(bcolors.OKBLUE, "----------------- Polynomial Interpolation -----------------\n", bcolors.ENDC)
    print(bcolors.OKBLUE, "Table Points:", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation for x =", bcolors.ENDC, x, '\n')

    interpolator = PolynomialInterpolator(table_points)
    result = interpolator.interpolate(x)

    print(bcolors.OKBLUE, "\n-----------------------------------------------------------\n", bcolors.ENDC)

