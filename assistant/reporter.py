"""
assistant/reporter.py
=======================
Presentación por consola. Es la ÚNICA capa que decide qué se imprime y
con qué precisión (Fase 3 y 4 de la especificación):
    - El cálculo numérico NUNCA se redondea (eso vive en Met_busqraices.py
      / met_solpvi_1ord.py / executor.py, sin tocar).
    - RN controla exclusivamente qué iteraciones de la traza COMPLETA
      se muestran aquí; no controla qué se calcula.
    - La "fórmula/algoritmo" se renderiza con sympy.pretty (arte
      Unicode tipo LaTeX de consola), pedido explícitamente para un
      entorno de terminal en vez de sympy.init_printing() (que altera
      el modo de impresión global del intérprete) o LaTeX real (que no
      se ve en consola).

Columna "ε_a (%)": error aproximado porcentual entre la iteración
actual y la anterior (|r_n - r_(n-1)| / |r_n| * 100), calculado por
Met_busqraices.py y guardado en cada registro del trace bajo la clave
'error_aprox'. Es puramente informativo: NO es el criterio que decide
cuándo detener la iteración (eso sigue siendo el residuo |f(r)|); se
muestra porque varios enunciados piden reportarlo o usarlo como
criterio de parada por su cuenta.
"""

from __future__ import annotations

from typing import List

from .analytical import pretty as sympy_pretty
from .schema import RootFindingResult, PVIResult, ComparisonResult
import sympy as sp

BAR = "=" * 70


def format_number(value, precision: int = 4) -> str:
    """Redondeo EXCLUSIVO de presentación. Nunca se usa el resultado
    de esta función para cálculos posteriores."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Búsqueda de raíces
# ---------------------------------------------------------------------------

def _formula_root(method: str) -> str:
    r_n, r_sig, r_ant, a_n, b_n = sp.symbols("r_n r_sig r_ant a_n b_n")
    f = sp.Function("f")
    g = sp.Function("g")
    fp = sp.Symbol("fprima_rn")  # f'(r_n); nombre sin apostrofe para que sympy.pretty lo imprima limpio

    if method in ("biseccion", "falsapos"):
        if method == "biseccion":
            expr = sp.Eq(r_n, (a_n + b_n) / 2)
        else:
            expr = sp.Eq(
                r_n,
                (a_n * f(b_n) - b_n * f(a_n)) / (f(b_n) - f(a_n)),
            )
    elif method == "newton":
        expr = sp.Eq(r_sig, r_n - f(r_n) / fp)
    elif method == "secante":
        expr = sp.Eq(
            r_sig,
            r_n - (f(r_n) * (r_ant - r_n)) / (f(r_ant) - f(r_n)),
        )
    elif method == "puntofijo":
        expr = sp.Eq(r_sig, g(r_n))
    else:
        raise ValueError(method)
    return sympy_pretty(expr)


_ROOT_METHOD_INFO = {
    "biseccion": {
        "nombre": "Método de Bisección",
        "tipo": "Método cerrado",
        "explicacion": [
            "1. Partir de un intervalo [a_n, b_n] donde f cambia de signo: f(a_n)*f(b_n) < 0.",
            "2. Calcular el punto medio r_n = (a_n + b_n) / 2.",
            "3. Evaluar el signo de f(r_n) para decidir en cuál mitad queda la raíz.",
            "4. Reemplazar a_n o b_n por r_n, reduciendo el intervalo, y repetir hasta la tolerancia.",
        ],
    },
    "falsapos": {
        "nombre": "Método de Falsa Posición (Regula Falsi)",
        "tipo": "Método cerrado",
        "explicacion": [
            "1. Partir de un intervalo [a_n, b_n] donde f cambia de signo.",
            "2. Trazar la recta secante entre (a_n, f(a_n)) y (b_n, f(b_n)).",
            "3. r_n es el punto donde esa recta corta al eje x.",
            "4. Evaluar el signo de f(r_n) y actualizar el intervalo conservando el cambio de signo.",
        ],
    },
    "newton": {
        "nombre": "Método de Newton-Raphson",
        "tipo": "Método abierto",
        "explicacion": [
            "1. Partir de una aproximación inicial r_n.",
            "2. Calcular la pendiente de la tangente: f'(r_n).",
            "3. r_(n+1) es el punto donde esa tangente corta al eje x.",
            "4. Repetir hasta alcanzar la tolerancia. Requiere f'(x) != 0.",
        ],
    },
    "secante": {
        "nombre": "Método de la Secante",
        "tipo": "Método abierto",
        "explicacion": [
            "1. Partir de dos aproximaciones iniciales r_ant y r_n.",
            "2. Trazar la recta secante entre (r_ant, f(r_ant)) y (r_n, f(r_n)).",
            "3. r_(n+1) es el punto donde esa recta corta al eje x.",
            "4. La aproximación más antigua se descarta y se repite el proceso.",
        ],
    },
    "puntofijo": {
        "nombre": "Método de Punto Fijo",
        "tipo": "Método abierto",
        "explicacion": [
            "1. Reescribir f(x) = 0 como una equivalencia x = g(x).",
            "2. Partir de una aproximación inicial r_n.",
            "3. Calcular r_(n+1) = g(r_n).",
            "4. Repetir hasta alcanzar la tolerancia. Converge solo si |g'(x)| < 1 cerca de la raíz.",
        ],
    },
}


def render_root_finding_report(result: RootFindingResult, comparison: ComparisonResult = None) -> str:
    problem = result.problem
    info = _ROOT_METHOD_INFO[problem.method]
    precision = problem.display_precision
    rn = set(problem.rn or [])

    lines = [BAR, f"MÉTODO: {info['nombre']}  ({info['tipo']})", BAR]

    lines.append("FÓRMULA / ALGORITMO:")
    lines.append(_formula_root(problem.method))
    if problem.method == "newton":
        lines.append("(fprima_rn = f'(r_n), la derivada evaluada en r_n)")
    if problem.method == "puntofijo":
        lines.append(f"(g(x) = {problem.gx})")
    lines.append("")

    lines.append("EXPLICACIÓN PASO A PASO:")
    lines.extend(f"  {paso}" for paso in info["explicacion"])
    lines.append("")

    lines.append(f"f(x) = {problem.fx}")
    lines.append("")

    filas = [entry for entry in result.trace if entry["n"] in rn]
    if filas:
        lines.append(f"ITERACIONES SELECCIONADAS (RN = {sorted(rn)}):")
        if problem.method in ("biseccion", "falsapos"):
            lines.append(f"{'n':>4} {'a_n':>14} {'b_n':>14} {'r_n':>14} {'f(r_n)':>14} {'ε_a (%)':>12}")
            for e in filas:
                lines.append(
                    f"{e['n']:>4} {format_number(e['a'], precision):>14} "
                    f"{format_number(e['b'], precision):>14} {format_number(e['r'], precision):>14} "
                    f"{format_number(e['f_r'], precision):>14} {format_number(e.get('error_aprox'), precision):>12}"
                )
        elif problem.method == "newton":
            fp_label = "f'(r_n)"
            lines.append(f"{'n':>4} {'r_n':>14} {'f(r_n)':>14} {fp_label:>14} {'ε_a (%)':>12}")
            for e in filas:
                lines.append(
                    f"{e['n']:>4} {format_number(e['r'], precision):>14} "
                    f"{format_number(e['f_r'], precision):>14} {format_number(e['fp_r'], precision):>14} "
                    f"{format_number(e.get('error_aprox'), precision):>12}"
                )
        elif problem.method == "secante":
            lines.append(f"{'n':>4} {'r_ant':>14} {'r_n':>14} {'r_(n+1)':>14} {'f(r_(n+1))':>14} {'ε_a (%)':>12}")
            for e in filas:
                lines.append(
                    f"{e['n']:>4} {format_number(e['r_prev'], precision):>14} "
                    f"{format_number(e['r_curr'], precision):>14} {format_number(e['r_next'], precision):>14} "
                    f"{format_number(e['f_r_next'], precision):>14} {format_number(e.get('error_aprox'), precision):>12}"
                )
        elif problem.method == "puntofijo":
            lines.append(f"{'n':>4} {'r_n':>14} {'g(r_n)':>14} {'f(r_n)':>14} {'ε_a (%)':>12}")
            for e in filas:
                lines.append(
                    f"{e['n']:>4} {format_number(e['r'], precision):>14} "
                    f"{format_number(e['g_r'], precision):>14} {format_number(e['f_r'], precision):>14} "
                    f"{format_number(e.get('error_aprox'), precision):>12}"
                )
    else:
        lines.append(f"(RN = {sorted(rn)} no coincide con ninguna iteración calculada: 0..{result.iterac})")
    lines.append("")

    lines.append("RESULTADO FINAL:")
    lines.append(f"  r ≈ {format_number(result.r, precision)}   (iteraciones: {result.iterac},  "
                 f"residuo |f(r)| = {format_number(result.error_residual, max(precision, 6))})")

    if comparison is not None:
        lines.append("")
        lines.append("COMPARACIÓN / ERROR:")
        if comparison.reference_value is not None:
            lines.append(f"  Referencia ({comparison.reference_source}): {format_number(comparison.reference_value, precision)}")
            lines.append(f"  Error absoluto: {format_number(comparison.absolute_error, precision)}")
            if comparison.relative_error is not None:
                lines.append(f"  Error relativo: {format_number(comparison.relative_error * 100, precision)}%")
        else:
            lines.append(f"  Sin referencia externa. Residuo |f(r)| = {format_number(comparison.absolute_error, max(precision,6))}")

    lines.append(BAR)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PVI de primer orden
# ---------------------------------------------------------------------------

_PVI1_METHOD_INFO = {
    "euler": {
        "nombre": "Método de Euler",
        "explicacion": [
            "1. Partir de (t_0, y_0).",
            "2. Calcular la pendiente m_n = f(t_n, y_n).",
            "3. Avanzar un paso: y_(n+1) = y_n + h * m_n ;  t_(n+1) = t_n + h.",
        ],
        "formula": lambda: sp.Eq(sp.Symbol("y_sig"), sp.Symbol("y_n") + sp.Symbol("h") * sp.Function("f")(sp.Symbol("t_n"), sp.Symbol("y_n"))),
    },
    "heun": {
        "nombre": "Método de Heun (Euler Mejorado)",
        "explicacion": [
            "1. Predecir con Euler: y*_(n+1) = y_n + h * f(t_n, y_n).",
            "2. Corregir promediando pendientes: m̄ = (f(t_n, y_n) + f(t_(n+1), y*_(n+1))) / 2.",
            "3. y_(n+1) = y_n + h * m̄.",
        ],
        "formula": lambda: sp.Eq(sp.Symbol("y_sig"), sp.Symbol("y_n") + sp.Symbol("h") * sp.Symbol("m_prom")),
    },
    "ptomed": {
        "nombre": "Método del Punto Medio",
        "explicacion": [
            "1. Avanzar medio paso: y_mitad = y_n + (h/2) * f(t_n, y_n).",
            "2. Evaluar la pendiente en el punto medio: f(t_n + h/2, y_mitad).",
            "3. Usar esa pendiente para avanzar el paso completo: y_(n+1) = y_n + h * f(t_n+h/2, y_mitad).",
        ],
        "formula": lambda: sp.Eq(sp.Symbol("y_sig"), sp.Symbol("y_n") + sp.Symbol("h") * sp.Function("f")(sp.Symbol("t_medio"), sp.Symbol("y_medio"))),
    },
    "rk4": {
        "nombre": "Runge-Kutta de 4to orden (RK4)",
        "explicacion": [
            "1. Calcular 4 pendientes: k1=f(t_n,y_n); k2=f(t_n+h/2, y_n+h*k1/2); k3=f(t_n+h/2, y_n+h*k2/2); k4=f(t_n+h, y_n+h*k3).",
            "2. Promediar con pesos: m = (k1 + 2k2 + 2k3 + k4) / 6.",
            "3. y_(n+1) = y_n + h * m.",
        ],
        "formula": lambda: sp.Eq(
            sp.Symbol("y_sig"),
            sp.Symbol("y_n") + sp.Symbol("h") * (sp.Symbol("k1") + 2*sp.Symbol("k2") + 2*sp.Symbol("k3") + sp.Symbol("k4")) / 6,
        ),
    },
}


def render_pvi1_report(result: PVIResult, comparison: ComparisonResult = None) -> str:
    problem = result.problem
    info = _PVI1_METHOD_INFO[problem.method]
    precision = problem.display_precision
    rn = set(problem.rn or [])

    lines = [BAR, f"MÉTODO: {info['nombre']}", BAR]

    lines.append("FÓRMULA / ALGORITMO:")
    lines.append(sympy_pretty(info["formula"]()))
    lines.append("")

    lines.append("EXPLICACIÓN PASO A PASO:")
    lines.extend(f"  {paso}" for paso in info["explicacion"])
    lines.append("")

    lines.append(f"y' = f(t, y) = {problem.ode_rhs}")
    lines.append(f"Condición inicial: y({format_number(problem.t0, precision)}) = {format_number(problem.y0, precision)}")
    lines.append(f"Intervalo: [{format_number(problem.t0, precision)}, {format_number(problem.t_end, precision)}]   h = {format_number(problem.h, precision)}")
    lines.append("")

    indices = sorted(i for i in rn if 0 <= i < len(result.t))
    if indices:
        lines.append(f"ITERACIONES SELECCIONADAS (RN = {sorted(rn)}; n=0 = condición inicial):")
        lines.append(f"{'n':>4} {'t_n':>14} {'y_n':>14}")
        for i in indices:
            etiqueta = "(condición inicial)" if i == 0 else ""
            lines.append(f"{i:>4} {format_number(result.t[i], precision):>14} {format_number(result.y[i], precision):>14}  {etiqueta}")
    else:
        lines.append(f"(RN = {sorted(rn)} no coincide con ningún índice calculado: 0..{len(result.t)-1})")
    lines.append("")

    lines.append("RESULTADO FINAL:")
    lines.append(f"  y({format_number(result.t[-1], precision)}) ≈ {format_number(result.y[-1], precision)}")

    if getattr(result, "evento_alcanzado", None):
        lines.append("")
        lines.append("EVENTO ALCANZADO:")
        if getattr(result, "evento_aproximado", False):
            metodo_cruce = f"extrapolación lineal desde el paso {result.paso_evento}"
        else:
            metodo_cruce = f"interpolación lineal entre los pasos {result.paso_evento - 1} y {result.paso_evento}"
        lines.append(
            f"  La variable llegó al valor objetivo (y = {format_number(problem.evento_y, precision)}) "
            f"en t ≈ {format_number(result.t_evento, precision)} ({metodo_cruce})."
        )
        lines.append(
            "  La integración se detuvo ahí (antes de t_end); el resultado final de arriba "
            "corresponde a este instante, no a t_end."
        )
        if getattr(result, "evento_aproximado", False):
            lines.append(
                "  (Nota: t_evento es una APROXIMACIÓN por extrapolación lineal: el último "
                "paso completo salió del dominio de f(t,y) — ej. sqrt(y) con y<0 por "
                "sobrepaso numérico muy cerca del objetivo, típico al vaciar un tanque.)"
            )

    if comparison is not None:
        lines.append("")
        lines.append("COMPARACIÓN / ERROR:")
        if comparison.reference_value is not None:
            lines.append(f"  Referencia ({comparison.reference_source}): {format_number(comparison.reference_value, precision)}")
            lines.append(f"  Error absoluto: {format_number(comparison.absolute_error, precision)}")
            if comparison.relative_error is not None:
                lines.append(f"  Error relativo: {format_number(comparison.relative_error * 100, precision)}%")
        else:
            lines.append("  Sin solución analítica ni de referencia disponible para comparar.")

    lines.append(BAR)
    return "\n".join(lines)
