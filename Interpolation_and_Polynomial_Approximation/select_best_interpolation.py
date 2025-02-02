from Interpolation_and_Polynomial_Approximation.polynomial_interpolation import PolynomialInterpolator
from Interpolation_and_Polynomial_Approximation.cubicSpline import CubicSplineInterpolator
import numpy as np
from colors import bcolors


class InterpolationSelector:
    """Compares interpolation methods and selects the best one."""

    def __init__(self, table_points, x):
        self.table_points = table_points
        self.x = x
        self.result_poly = None
        self.result_spline = None

    def compute_methods(self):
        """Computes both Polynomial and Cubic Spline Interpolation results."""
        polynomial = PolynomialInterpolator(self.table_points)
        self.result_poly = polynomial.interpolate(self.x)

        spline = CubicSplineInterpolator(self.table_points)
        self.result_spline = spline.compute_spline(self.x)

    def compare_methods(self):
        """Compares results and determines the best method."""
        print(bcolors.OKBLUE, f"\nPolynomial Interpolation Result: {self.result_poly}")
        print(bcolors.OKBLUE, f"Cubic Spline Interpolation Result: {self.result_spline}")

        if abs(self.result_poly - self.result_spline) > 15:
            print(
                bcolors.WARNING,
                "\nLarge difference detected between the two methods.",
                "Cubic Spline is usually more stable and accurate for non-polynomial data.",
                bcolors.ENDC,
            )
            best_method = "Cubic Spline Interpolation"
        else:
            best_method = "Polynomial Interpolation (if lower degree is used)"

        print(bcolors.OKGREEN, f"\nBest method for this problem: {best_method}", bcolors.ENDC)


if __name__ == '__main__':
    table_points = [
        (0.35, -213.5991), (0.4, -204.4416), (0.55, -194.9375),
        (0.65, -185.0256), (0.7, -174.6711), (0.85, -163.8656), (0.9, -152.6271)
    ]
    x = 0.75

    print(bcolors.OKBLUE, "\n----------------- Selecting the Best Interpolation Method -----------------\n", bcolors.ENDC)

    selector = InterpolationSelector(table_points, x)
    selector.compute_methods()
    selector.compare_methods()

    print(bcolors.OKBLUE, "\n---------------------------------------------------------------------\n", bcolors.ENDC)
