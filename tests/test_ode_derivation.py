"""
Pruebas de assistant/ode_derivation.py (sesion 8): eliminacion
simbolica de variables auxiliares para reducir la tasa de cambio de un
PVI a una unica expresion ode_rhs(t, y). Esto reemplaza pedirle al LLM
que haga el algebra el mismo (fuente real de un bug: se quedaba sin
tokens "pensando" en la sustitucion, ver test_assistant_pipeline.py
para el bug de MAX_TOKENS en gemini_provider.py).
"""

import math
import os
import sys

import pytest
import sympy as sp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from assistant.ode_derivation import derive_ode_rhs, OdeDerivationError


def test_gota_que_se_evapora_caso_real():
    # Ejercicio real de Jaison: dV/dt=-k*A, A=4*pi*r**2, V=(4/3)*pi*r**3.
    resultado = derive_ode_rhs(
        rate_expr="-k*A",
        auxiliary_relations=["A = 4*pi*r**2", "V = (4/3)*pi*r**3"],
        state_var="V",
        parameters={"k": 0.08},
    )
    obtenido = sp.sympify(resultado)
    y_sym = next(iter(obtenido.free_symbols))
    f_obtenido = sp.lambdify(y_sym, obtenido, "numpy")
    f_esperado = lambda v: -0.08 * 6**(2/3) * math.pi**(1/3) * v**(2/3)
    for v in (1.0, 20.58, 65.45):
        assert abs(f_obtenido(v) - f_esperado(v)) < 1e-9

    # Verificacion fisica independiente: si dV/dt=-k*A y V=(4/3)pi r^3,
    # entonces (por regla de la cadena) dr/dt = -k EXACTAMENTE (constante).
    # Con r0=2.5 y k=0.08 durante 10 min, r_final debe ser 2.5-0.8=1.7.
    f = f_obtenido
    r0 = 2.5
    V0 = (4/3) * math.pi * r0**3
    # Un paso de Euler bien fino para verificar la direccion/magnitud
    # (no es la prueba de precision numerica -- esa vive en
    # test_assistant_pipeline.py via el pipeline completo).
    V, dt, n = V0, 10/100000, 100000
    for _ in range(n):
        V += dt * f(V)
    r_final = (3*V/(4*math.pi))**(1/3)
    assert abs(r_final - 1.7) < 1e-3


def test_sin_variables_auxiliares_que_eliminar_devuelve_igual():
    # rate_expr ya estaba solo en terminos de la variable de estado: no
    # hay nada que sustituir, se devuelve (renombrada a "y").
    resultado = derive_ode_rhs(
        rate_expr="-k*V",
        auxiliary_relations=["A = 4*pi*r**2"],  # relacion irrelevante, no se usa
        state_var="V",
        parameters={"k": 0.175},
    )
    assert sp.simplify(sp.sympify(resultado) - sp.sympify("-0.175*y")) == 0


def test_sin_relaciones_auxiliares_da_error_claro():
    with pytest.raises(OdeDerivationError, match="relaciones auxiliares"):
        derive_ode_rhs(rate_expr="-k*A", auxiliary_relations=[], state_var="V")


def test_relacion_mal_formada_da_error_claro():
    with pytest.raises(OdeDerivationError, match="lhs = rhs"):
        derive_ode_rhs(
            rate_expr="-k*A",
            auxiliary_relations=["A 4*pi*r**2"],  # falta el "="
            state_var="V",
            parameters={"k": 0.08},
        )


def test_variable_de_estado_no_aparece_da_error_claro():
    with pytest.raises(OdeDerivationError, match="no aparece"):
        derive_ode_rhs(
            rate_expr="-k*A",
            auxiliary_relations=["A = 4*pi*r**2"],
            state_var="Q",  # Q no aparece en ninguna relacion
            parameters={"k": 0.08},
        )


def test_relaciones_insuficientes_dejan_variable_sin_eliminar():
    # Falta la relacion que conecta r con V: A queda en terminos de r,
    # que a su vez nunca se relaciona con V. Debe fallar de forma clara,
    # no devolver una expresion con "r" colado.
    with pytest.raises(OdeDerivationError):
        derive_ode_rhs(
            rate_expr="-k*A",
            auxiliary_relations=["A = 4*pi*r**2"],  # falta V=(4/3)*pi*r**3
            state_var="V",
            parameters={"k": 0.08},
        )


def test_funcion_no_permitida_es_rechazada():
    with pytest.raises(OdeDerivationError):
        derive_ode_rhs(
            rate_expr="-k*A",
            auxiliary_relations=["A = __import__('os').system('echo hola')"],
            state_var="V",
            parameters={"k": 0.08},
        )
