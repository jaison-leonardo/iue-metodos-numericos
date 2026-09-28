"""
Módulo en el que se definen las funciones que se utilizan
en los métodos de solución de PVIs.
* funced: función presente en la ecuación diferencial.
* funcsolex: solución exacta del PVI en caso de existir.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import params_pvi1 as params

t, y = sp.symbols('t y')

# Función f de la EDO (y' = f(t,y))
def funced(t_, y_):
    expr = sp.sympify(params.FUNCED)
    return sp.lambdify((t, y), expr, "numpy")(t_, y_)

# Solución exacta
def funcsolex(t_):
    expr = sp.sympify(params.FUNCSOLEX)
    return sp.lambdify(t, expr, "numpy")(t_)

# Solución de referencia del PVI
def funcsolref(t_0, T, y_0):
    sol = solve_ivp(lambda t_, y_: funced(t_, y_[0]), [t_0, T], [y_0], rtol = 1e-10)
    return sol
