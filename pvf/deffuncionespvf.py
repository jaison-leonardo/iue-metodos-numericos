import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp
import params_pvf as params

x = sp.symbols('x')

# Coeficiente de la segunda derivada
def A(x_):
    expr = sp.sympify(params.AX)
    return sp.lambdify(x, expr, "numpy")(x_)

# Coeficiente de la primera derivada
def B(x_):
    expr = sp.sympify(params.BX)
    return sp.lambdify(x, expr, "numpy")(x_)

# Coeficiente de la función y
def C(x_):
    expr = sp.sympify(params.CX)
    return sp.lambdify(x, expr, "numpy")(x_)

# Función de términos independientes
def f(x_):
    expr = sp.sympify(params.FX)
    return sp.lambdify(x, expr, "numpy")(x_)

# Solución exacta
def funcsolex(x_):
    expr = sp.sympify(params.FUNCSOLEX)
    return sp.lambdify(x, expr, "numpy")(x_)

# Solución de referencia utilizando solve_bvp
def funcsolref(x, ysr, y_0, y_n):
    def ode(x, y):
        return np.array([y[1], -(B(x)/A(x))*y[1] - (C(x)/A(x))*y[0] + f(x)/A(x)])

    def bc(ya, yb):
        return np.array([ya[0] - y_0, yb[0] - y_n])

    sol = solve_bvp(ode, bc, x, ysr)
    return sol
