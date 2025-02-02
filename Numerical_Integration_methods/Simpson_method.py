'''import math
import numpy as np
import matplotlib.pyplot as plt

import sympy as sp

from colors import bcolors
from sympy.utilities.lambdify import lambdify
x = sp.symbols('x')
def simpsons_rule(f, a, b, n):
    """
    Simpson's Rule for Numerical Integration

    Parameters:
    f (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals (must be even).

    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's Rule.")

    h = (b - a) / n

    integral = f(a) + f(b)  # Initialize with endpoints

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            integral += 2 * f(x_i)
        else:
            integral += 4 * f(x_i)

    integral *= h / 3

    return integral


if __name__ == '__main__':
    f = lambda x: math.e ** (x ** 2)
    n = 10
    a=0
    b=1

    print( f" Division into n={n} sections ")
    integral = simpsons_rule(f, 0, 1, n)
    print(bcolors.OKBLUE, f"Numerical Integration of definite integral in range [{a},{b}] is {integral}", bcolors.ENDC)'''



'''   AFTER REFACTORING'''

import math
import sympy as sp
from colors import bcolors
from Numerical_Integration_methods.gaussianQuadrature_method import GaussianQuadrature

x = sp.symbols('x')


class FunctionIntegrator:
    """Handles numerical integration for a given function."""

    def __init__(self, function):
        """Initialize with a function to be integrated."""
        self.function = function

    def integrate(self, method, a, b, n):
        """Perform numerical integration using a given method."""
        return method.apply(self.function, a, b, n)


class SimpsonsRule:
    """Implementation of Simpson's Rule for numerical integration."""

    def apply(self, f, a, b, n):
        """Compute the integral using Simpson's Rule."""
        if n % 2 != 0:
            raise ValueError("Simpson's Rule requires an even number of intervals.")

        h = (b - a) / n
        integral = f(a) + f(b)

        for i in range(1, n):
            x_i = a + i * h
            if i % 2 == 0:
                integral += 2 * f(x_i)
            else:
                integral += 4 * f(x_i)

        return integral * (h / 3)


if __name__ == '__main__':
    #  Function from the PDF
    def problem_function(x):
        return (x * math.exp(-x**2 + 5*x)) * (2*x**2 - 3*x - 5)

    #  Problem 1 parameters
    a, b = 0.5, 1  # Integration range
    n = 10  # Must be even for Simpson's Rule
    n_gaussian = 5  # Gaussian Quadrature points

    print(f"Computing integrals for Problem 1")

    #  Compute integral using Simpson's Rule
    integrator = FunctionIntegrator(problem_function)
    simpsons = SimpsonsRule()
    result_simpson = integrator.integrate(simpsons, a, b, n)

    #  Compute integral using Gaussian Quadrature
    gaussian = GaussianQuadrature()
    result_gaussian = gaussian.integrate(problem_function, a, b, n_gaussian)

    #  Compute the final averaged result
    final_result = (result_simpson + result_gaussian) / 2

    print(bcolors.OKBLUE,
          f"Final averaged result for Problem 1: {final_result}",
          bcolors.ENDC)


