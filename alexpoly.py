from fractions import Fraction

from jaxgeom import get_knot
from sympy import symbols, simplify
import sympy as sp

from pyknotid.spacecurves import Knot


def divided_diff(x, y):
    n = len(y)
    coef = [Fraction(yi) for yi in y]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef

def newton_interp(coef, x_data, x):
    n = len(x_data) - 1
    result = coef[n]
    for i in range(n - 1, -1, -1):
        result = result * (x - x_data[i]) + coef[i]
    return result


def get_poly(x_data, y_data):
    coef = divided_diff(x_data, y_data)
    x = symbols('x')
    polynomial = coef[0]
    for i in range(1, len(coef)):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        polynomial += term
    simplified_poly = simplify(polynomial)
    return simplified_poly


def get_alex_poly(func, numpts = 100):
   foo = get_knot(func)
   frog = Knot(foo)
   apoly = frog.alexander_polynomial()
   gcode = frog.gauss_code()
   if len(gcode) < 3:
      return 1
   x = list(range(1, numpts+1))
   y = [frog.alexander_polynomial(variable = xx) for xx in x]
   return get_poly(x, y)
