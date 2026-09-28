"""
assistant/validator.py
========================
Validación DETERMINISTA de un problema estructurado, antes de tocar
SymPy "en serio" o ejecutar cualquier método numérico (sección 7 de la
especificación). Nunca ejecuta una interpretación matemática dudosa:
si algo no se puede verificar, se devuelve un ValidationResult con
ok=False y la lista de errores, sin intentar adivinar.
"""

from __future__ import annotations

import sympy as sp

from .analytical import safe_sympify, UnsafeExpressionError
from .schema import RootFindingProblem, PVIProblem, ValidationResult, ROOT_METHODS, PVI1_METHODS


def validate_root_finding_problem(problem: RootFindingProblem) -> ValidationResult:
    errors = []

    if problem.method not in ROOT_METHODS:
        errors.append(f"Método '{problem.method}' no reconocido. Válidos: {ROOT_METHODS}")

    x = sp.symbols("x")
    fx_expr = None
    try:
        fx_expr = safe_sympify(problem.fx, [x])
    except UnsafeExpressionError as e:
        errors.append(f"f(x) inválida/insegura: {e}")
    except sp.SympifyError as e:
        errors.append(f"f(x) no es una expresión matemática válida: {e}")

    if problem.tol is None or problem.tol <= 0:
        errors.append("La tolerancia (tol) debe ser un número positivo.")

    if problem.maxit is None or int(problem.maxit) <= 0:
        errors.append("El máximo de iteraciones (maxit) debe ser un entero positivo.")

    if problem.method in ("biseccion", "falsapos"):
        if problem.a0 is None or problem.b0 is None:
            errors.append(f"El método '{problem.method}' requiere intervalo inicial a0 y b0.")
        elif fx_expr is not None:
            try:
                fa = float(fx_expr.subs(x, problem.a0))
                fb = float(fx_expr.subs(x, problem.b0))
                if fa * fb >= 0:
                    errors.append(
                        f"No hay cambio de signo en [{problem.a0}, {problem.b0}]: "
                        f"f(a0)={fa}, f(b0)={fb}. No se puede garantizar una raíz en ese intervalo."
                    )
            except Exception as e:
                errors.append(f"No se pudo evaluar f(x) en a0/b0: {e}")
    elif problem.method == "newton":
        if problem.x0 is None:
            errors.append("El método 'newton' requiere una aproximación inicial x0.")
    elif problem.method == "secante":
        if problem.x0 is None or problem.x1 is None:
            errors.append("El método 'secante' requiere dos aproximaciones iniciales x0 y x1.")
    elif problem.method == "puntofijo":
        if problem.x0 is None:
            errors.append("El método 'puntofijo' requiere una aproximación inicial x0.")
        if not problem.gx:
            errors.append(
                "El método 'puntofijo' requiere la función de iteración g(x) "
                "(campo 'gx') tal que x = g(x); no se deriva automáticamente de f(x)."
            )
        elif fx_expr is not None:
            try:
                gx_expr = safe_sympify(problem.gx, [x])
            except UnsafeExpressionError as e:
                errors.append(f"g(x) inválida/insegura: {e}")
            except sp.SympifyError as e:
                errors.append(f"g(x) no es una expresión matemática válida: {e}")

    if problem.rn is not None:
        if not all(isinstance(n, int) and n >= 0 for n in problem.rn):
            errors.append("RN debe ser una lista de enteros no negativos.")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_pvi_problem(problem: PVIProblem) -> ValidationResult:
    errors = []

    if problem.method not in PVI1_METHODS:
        errors.append(f"Método '{problem.method}' no reconocido. Válidos: {PVI1_METHODS}")

    t, y = sp.symbols("t y")
    try:
        safe_sympify(problem.ode_rhs, [t, y])
    except UnsafeExpressionError as e:
        errors.append(f"f(t,y) inválida/insegura: {e}")
    except sp.SympifyError as e:
        errors.append(f"f(t,y) no es una expresión matemática válida: {e}")

    if problem.h is None or problem.h <= 0:
        errors.append("El tamaño de paso h debe ser un número positivo.")

    if problem.t0 is None or problem.t_end is None:
        errors.append("Se requieren t0 y t_end (intervalo de solución).")
    elif problem.t_end <= problem.t0:
        errors.append("t_end debe ser mayor que t0.")

    if problem.y0 is None:
        errors.append("Se requiere la condición inicial y0.")

    if problem.analytical_solution:
        try:
            safe_sympify(problem.analytical_solution, [t])
        except (UnsafeExpressionError, sp.SympifyError) as e:
            errors.append(f"La solución analítica proporcionada no es válida: {e}")

    if problem.rn is not None:
        if not all(isinstance(n, int) and n >= 0 for n in problem.rn):
            errors.append("RN debe ser una lista de enteros no negativos.")

    if problem.evento_y is not None:
        if not isinstance(problem.evento_y, (int, float)):
            errors.append("evento_y (parada por evento) debe ser un número.")
        elif problem.y0 is not None and problem.evento_y == problem.y0:
            errors.append(
                "evento_y no puede ser igual a y0: el evento ya estaría "
                "cumplido en t0, no hay nada que integrar."
            )

    return ValidationResult(ok=(len(errors) == 0), errors=errors)
