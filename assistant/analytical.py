"""
assistant/analytical.py
========================
Motor simbólico compartido (SymPy). Centraliza el patrón que hoy está
duplicado en deffunciones.py, deffuncionespvi.py, deffuncionespvi2.py
y deffuncionespvf.py (`sp.sympify(...)` + `sp.lambdify(...)`), y agrega
la lista blanca de operaciones permitidas que pide la Fase 22 de la
especificación (nunca usar `eval()` sobre texto proveniente del LLM).

Este módulo NO modifica ni reemplaza los `deffunciones*.py` existentes:
esos siguen funcionando exactamente igual. `analytical.py` es para el
código NUEVO (validator.py, executor.py, verifier.py) que necesita
compilar expresiones que puedan venir de un LLM, con validación previa.
"""

from __future__ import annotations

import sympy as sp

# Funciones matemáticas permitidas explícitamente (Fase 22 de la
# especificación). Cualquier nombre de función en la expresión que no
# esté en esta lista blanca se rechaza antes de evaluar nada.
ALLOWED_FUNCTION_NAMES = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "exp", "log", "ln", "sqrt", "Abs", "abs",
    "Min", "Max", "sign", "floor", "ceiling",
}

# Constantes/símbolos permitidos además de las variables del problema.
ALLOWED_CONSTANT_NAMES = {"pi", "E", "I"}


class UnsafeExpressionError(ValueError):
    """Se lanza cuando una expresión usa nombres fuera de la lista blanca."""


def _collect_function_names(expr: sp.Expr) -> set:
    names = set()
    for f in expr.atoms(sp.Function):
        names.add(f.func.__name__)
    for f in expr.atoms(sp.core.function.AppliedUndef):
        names.add(f.func.__name__)
    return names


def safe_sympify(expr_str: str, allowed_symbols):
    """
    Convierte un string en una expresión SymPy, SIN usar eval(), y
    valida que:
      - Solo aparecen símbolos en `allowed_symbols` (las variables
        legítimas del problema, p.ej. {'x'} o {'t', 'y'}).
      - Solo aparecen funciones matemáticas en ALLOWED_FUNCTION_NAMES.

    Lanza UnsafeExpressionError si la expresión usa algo no permitido.
    Lanza sympy.SympifyError (heredado del comportamiento de SymPy) si
    el string no es una expresión matemática válida.
    """
    allowed_symbol_names = {str(s) for s in allowed_symbols}

    expr = sp.sympify(expr_str, convert_xor=True, evaluate=True)

    free_symbol_names = {str(s) for s in expr.free_symbols}
    desconocidos = free_symbol_names - allowed_symbol_names - ALLOWED_CONSTANT_NAMES
    if desconocidos:
        raise UnsafeExpressionError(
            f"La expresión usa símbolos no permitidos/no declarados: {sorted(desconocidos)}"
        )

    nombres_funciones = _collect_function_names(expr)
    no_permitidas = nombres_funciones - ALLOWED_FUNCTION_NAMES
    if no_permitidas:
        raise UnsafeExpressionError(
            f"La expresión usa funciones no permitidas: {sorted(no_permitidas)}"
        )

    return expr


def compile_expr(expr_str: str, symbols):
    """
    Valida (safe_sympify) y compila una expresión a una función numérica
    numpy-vectorizada, lista para evaluar. `symbols` es una tupla de
    símbolos SymPy en el orden en que se llamará a la función resultante,
    p.ej. (t, y) para f(t, y).

    Retorna (expr_sympy, funcion_numerica).
    """
    if not isinstance(symbols, (tuple, list)):
        symbols = (symbols,)
    expr = safe_sympify(expr_str, symbols)
    fn = sp.lambdify(symbols, expr, "numpy")
    return expr, fn


def pretty(expr_or_str, symbols=None) -> str:
    """
    Representación tipo "LaTeX de consola" (arte ASCII/Unicode) de una
    expresión, usando sympy.pretty (lo mismo que usa internamente
    sp.pprint / sp.init_printing, pero devuelto como string en vez de
    imprimirse directamente), para insertar dentro del reporte de
    consola (assistant/reporter.py).
    """
    if isinstance(expr_or_str, str):
        expr = safe_sympify(expr_or_str, symbols or [])
    else:
        expr = expr_or_str
    return sp.pretty(expr, use_unicode=True)


def derivative(expr_str: str, symbol_str: str, symbols):
    """Deriva una expresión (validada) respecto a `symbol_str`."""
    expr = safe_sympify(expr_str, symbols)
    x = sp.symbols(symbol_str)
    return sp.diff(expr, x)


def try_dsolve(ode_lhs_rhs_str: str, dependent: str, independent: str):
    """
    Intenta resolver simbólicamente una EDO de primer orden con SymPy.
    `ode_lhs_rhs_str` debe ser el lado derecho de y' = f(t, y), YA
    validado (proviene de compile_expr / safe_sympify antes de llegar
    aquí en el flujo real).
    Retorna la solución simbólica (sp.Eq) o None si SymPy no puede
    resolverla. Nunca lanza la solución como si fuera verdad absoluta:
    quien llama debe pasar el resultado por verifier.py.
    """
    t = sp.symbols(independent)
    y = sp.Function(dependent)
    rhs = safe_sympify(ode_lhs_rhs_str, [t, sp.Symbol(dependent)])
    rhs = rhs.subs(sp.Symbol(dependent), y(t))
    ode = sp.Eq(sp.diff(y(t), t), rhs)
    try:
        return sp.dsolve(ode, y(t))
    except Exception:
        return None
