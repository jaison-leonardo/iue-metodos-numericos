"""
assistant/verifier.py
=======================
Verifica que una solución (analítica, generada o sugerida por un LLM)
sea realmente correcta, usando SymPy, ANTES de que se considere una
fuente de verdad (sección 10 de la especificación: "Una solución
analítica generada por Claude nunca debe considerarse válida
únicamente por haber sido generada por el LLM").
"""

from __future__ import annotations

import sympy as sp

from .analytical import safe_sympify, UnsafeExpressionError


def verify_ode_solution(ode_rhs_str: str, solution_str: str, t0: float, y0: float, tol: float = 1e-8) -> dict:
    """
    Verifica que y(t) = solution_str cumpla:
        dy/dt == f(t, y)   (sustituyendo y por la solución propuesta)
        y(t0) == y0
    Retorna un dict {"verified": bool, "details": str}.
    """
    t = sp.symbols("t")
    y_sym = sp.symbols("y")

    try:
        rhs = safe_sympify(ode_rhs_str, [t, y_sym])
        sol = safe_sympify(solution_str, [t])
    except (UnsafeExpressionError, sp.SympifyError) as e:
        return {"verified": False, "details": f"Expresión inválida: {e}"}

    lhs = sp.diff(sol, t)
    rhs_sustituida = rhs.subs(y_sym, sol)

    diferencia = sp.simplify(lhs - rhs_sustituida)
    cumple_ode = diferencia == 0

    try:
        y_en_t0 = float(sol.subs(t, t0))
        cumple_ci = abs(y_en_t0 - y0) <= tol
    except Exception as e:
        cumple_ci = False
        y_en_t0 = None

    verified = bool(cumple_ode and cumple_ci)
    details = (
        f"dy/dt - f(t,y(t)) simplificado = {diferencia} (¿0? {cumple_ode}); "
        f"y({t0}) = {y_en_t0}, esperado {y0} (¿coincide? {cumple_ci})"
    )
    return {"verified": verified, "details": details}


def verify_root(fx_str: str, root_value: float, tol: float = 1e-6) -> dict:
    """Verifica que |f(root_value)| esté dentro de una tolerancia razonable."""
    x = sp.symbols("x")
    try:
        expr = safe_sympify(fx_str, [x])
    except (UnsafeExpressionError, sp.SympifyError) as e:
        return {"verified": False, "details": f"Expresión inválida: {e}"}

    residual = abs(float(expr.subs(x, root_value)))
    verified = residual <= max(tol, 1e-4)
    return {"verified": verified, "details": f"|f({root_value})| = {residual}"}
