from Numerical_Integration_methods.Simpson_method import FunctionIntegrator, SimpsonsRule
from Numerical_Integration_methods.gaussianQuadrature_method import GaussianQuadrature
from Methods_for_Solving_Linear_Systems_of_Equations.LU_factorization import LUSolver
from Methods_for_Solving_Linear_Systems_of_Equations.gauss_seidel import GaussSeidelSolver
from Interpolation_and_Polynomial_Approximation.select_best_interpolation import InterpolationSelector
import numpy as np
import math
from colors import bcolors


# ✅ Five formulas from the article
def Ex1(L):
    return 4.86 + 0.018 * L

def Ex2(L):
    return L / 3000

def Ex3(L):
    A0, A1, A2 = 0.0047, 0.0023, 0.000043
    return (A0 + A1 * math.log(L) + A2 * (math.log(L) ** 2)) * L

def Ex4(L):
    return 4.2 + 0.0015 * (L ** (4 / 3))

def Ex5(L):
    return 0.069 + 0.00156 * L + 0.00000047 * (L ** 2)

def calculate_all_results(L, label):
    """Applies all five formulas to L and prints the results."""
    results = {
        "Akiyama": Ex1(L),
        "Halstead": Ex2(L),
        "Lipow": Ex3(L),
        "Gafney": Ex4(L),
        "Compton and Withrow": Ex5(L)
    }

    print(bcolors.OKBLUE, f"\nResults for {label} (Transformed L = {L}):", bcolors.ENDC)
    for model, value in results.items():
        print(f"{model}: {value:.2f}")


def get_problem_results():
    """Runs both methods for each problem, selects the best method, and returns L1, L2, and L3."""

    # ✅ Problem 1: Simpson's Rule vs Gaussian Quadrature
    def problem_1_function(x):
        return (x * np.exp(-x ** 2 + 5 * x)) * (2 * x ** 2 - 3 * x - 5)

    a, b = 0.5, 1
    n = 10
    integrator = FunctionIntegrator(problem_1_function)
    simpsons = SimpsonsRule()
    gaussian = GaussianQuadrature()

    result_simpson = abs(integrator.integrate(simpsons, a, b, n))
    result_gaussian = abs(gaussian.integrate(problem_1_function, a, b, n))  # ✅ FIXED HERE

    # Choosing the best method: Simpson’s Rule
    L1 = result_simpson
    print(bcolors.OKGREEN, f"\nBest Result for Problem 1 (Simpson’s Rule): {L1}", bcolors.ENDC)

    # ✅ Problem 2: Gauss-Seidel vs LU Factorization
    A = np.array([[-1, -1, 2], [2, -1, 1], [2, 2, 2]], dtype=np.double)
    b = np.array([-5, -1, 4], dtype=np.double)

    gauss_seidel_solver = GaussSeidelSolver(A, b)
    solution_GS = gauss_seidel_solver.solve()

    lu_solver = LUSolver(A, b)
    solution_LU = lu_solver.solve()

    # Choosing the best method: LU Factorization
    L2 = abs(solution_LU[2])
    print(bcolors.OKGREEN, f"Best Result for Problem 2 (LU Factorization): {L2}", bcolors.ENDC)

    # ✅ Problem 3: Polynomial Interpolation vs Cubic Spline
    table_points = [
        (0.35, -213.5991), (0.4, -204.4416), (0.55, -194.9375),
        (0.65, -185.0256), (0.7, -174.6711), (0.85, -163.8656), (0.9, -152.6271)
    ]
    x = 0.75
    selector = InterpolationSelector(table_points, x)
    selector.compute_methods()

    # Choosing the best method: Cubic Spline
    L3 = abs(selector.result_spline)
    print(bcolors.OKGREEN, f"Best Result for Problem 3 (Cubic Spline Interpolation): {L3}", bcolors.ENDC)

    return L1, L2, L3


if __name__ == '__main__':
    print(bcolors.OKBLUE, "\n----------------- Computing Best Results for Each Problem -----------------\n",
          bcolors.ENDC)

    # Compute L1, L2, L3 for each problem
    L1, L2, L3 = get_problem_results()

    print(bcolors.OKBLUE, "\n----------------- Computing the Five Formulas -----------------\n", bcolors.ENDC)

    # Compute formulas using the transformed values
    calculate_all_results(L1 * 100, "L1 (Problem 1 - Simpson’s Rule)")
    calculate_all_results(L2 * 800, "L2 (Problem 2 - LU Factorization)")
    calculate_all_results(L3 * 60, "L3 (Problem 3 - Cubic Spline)")

    print(bcolors.OKBLUE, "\n---------------------------------------------------------------\n", bcolors.ENDC)
