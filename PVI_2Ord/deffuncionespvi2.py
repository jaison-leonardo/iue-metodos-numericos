"""
Módulo en el que se definen las funciones que se utilizan
en los métodos de solución de PVIs.
* funced: función presente en la ecuación diferencial.
* funcsolex: solución exacta del PVI en caso de existir.
* funsolref: solución de referencia del PVI utilizando solve_ivp.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import params_pvi2 as params

t, y0, y1 = sp.symbols('t y0 y1')

# Función f de la EDO (y'' = g(t,y,y'))
def funced(t_, y_):
    expr = sp.sympify(params.FUNCED)
    return sp.lambdify((t, y0, y1), expr, "numpy")(t_, y_[0], y_[1])

# Solución exacta
def funcsolex(t_):
    expr = sp.sympify(params.FUNCSOLEX)
    return sp.lambdify(t, expr, "numpy")(t_)

# Solución de referencia del PVI
def funcsolref(t_0, T, y_0):
    sol = solve_ivp(lambda t_, y_: [y_[1], funced(t_, y_)], [t_0, T], y_0, rtol = 1e-10)
    return sol
