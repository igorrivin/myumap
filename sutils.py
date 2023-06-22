from statsmodels.distributions.empirical_distribution import ECDF
import cvxpy as cp
import numpy as np

def fit_poly(data, degree=10, eps = 1e-6):
    ecdf = ECDF(data)
    x = ecdf.x[1:]
    y = ecdf.y[1:]

    V = np.vander(x, degree+1)
    coeffs = cp.Variable(degree+1)
    objective = cp.Minimize(cp.sum_squares(V @ coeffs - y))
    constraints = [coeffs >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
   
    c = np.where(np.abs(coeffs.value) > eps, coeffs.value, 0)
    p = np.poly1d(c)

    return p, c


