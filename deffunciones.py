"""
Módulo en el que se definen las funciones que se utilizan
en los métodos de búsqueda de raíces.
* funcraiz: función a la que se le calculan las raices.
* funcraizder: derivada de funcraiz.
"""

import numpy as np
import sympy as sp
import params

fx = sp.symbols('x')

# Función a la que se le busca la raíz
def funcraiz(x):
    f = sp.sympify(params.FX)
    return sp.lambdify(fx, f, "numpy")(x)

# Derivada de funcraiz
def funcraizder(x):
    f = sp.sympify(params.FX)
    df = sp.diff(f, fx)
    return sp.lambdify(fx, df, "numpy")(x)

if __name__ == "__main__":
    print(funcraizder(-2))