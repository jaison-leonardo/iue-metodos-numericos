"""
assistant/comparator.py
=========================
Comparación numérica y cálculo de errores (secciones 17 y 19 de la
especificación). Toma valores ya calculados de forma determinista
(por executor.py / SymPy / solve_ivp) y produce métricas objetivas.
No inventa valores: si falta la referencia, lo indica explícitamente
en vez de aproximar en silencio.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .analytical import safe_sympify, UnsafeExpressionError
from .schema import ComparisonResult


def compare_root(numeric_root: float, fx_str: str, tol: float, reference_root: float = None) -> ComparisonResult:
    """
    Para búsqueda de raíces: si se da una raíz de referencia (analítica
    o de otro método más preciso), calcula error absoluto/relativo
    contra ella; si no, reporta el residuo |f(r)| como única medida
    disponible (fuente = "residual").
    """
    x = sp.symbols("x")
    try:
        expr = safe_sympify(fx_str, [x])
        residual = abs(float(expr.subs(x, numeric_root)))
    except (UnsafeExpressionError, sp.SympifyError):
        residual = None

    if reference_root is not None:
        abs_err = abs(numeric_root - reference_root)
        rel_err = abs_err / abs(reference_root) if reference_root != 0 else None
        return ComparisonResult(
            numeric_value=numeric_root,
            reference_value=reference_root,
            reference_source="analytical",
            absolute_error=abs_err,
            relative_error=rel_err,
            converged=(residual is not None and residual <= tol) if residual is not None else abs_err <= tol,
        )

    return ComparisonResult(
        numeric_value=numeric_root,
        reference_value=None,
        reference_source="residual",
        absolute_error=residual,
        relative_error=None,
        converged=(residual is not None and residual <= tol),
    )


def compare_pvi(numeric_final: float, analytical_final: float = None, reference_final: float = None) -> ComparisonResult:
    """
    Para PVI: compara el valor final numérico contra, en orden de
    prioridad, la solución analítica (si existe y fue verificada) o la
    solución de referencia de alta precisión (solve_ivp). Nunca
    considera "convergido" si no hay ninguna referencia disponible.
    """
    if analytical_final is not None:
        ref, fuente = analytical_final, "analytical"
    elif reference_final is not None:
        ref, fuente = reference_final, "solve_ivp"
    else:
        return ComparisonResult(
            numeric_value=numeric_final, reference_value=None, reference_source=None,
            absolute_error=None, relative_error=None, converged=False,
        )

    abs_err = abs(numeric_final - ref)
    rel_err = abs_err / abs(ref) if ref != 0 else None
    return ComparisonResult(
        numeric_value=numeric_final, reference_value=ref, reference_source=fuente,
        absolute_error=abs_err, relative_error=rel_err, converged=True,
    )


def linear_regression_slope(x_values, y_values) -> dict:
    """
    Pendiente numérica obtenida por regresión lineal (sección 17:
    comparar pendiente analítica vs. pendiente de los datos numéricos,
    p.ej. para análisis semilogarítmico ln(y) vs. t).
    """
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    return {"slope": float(slope), "intercept": float(intercept)}
