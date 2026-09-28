"""
assistant/plotter.py
======================
Generación OPCIONAL de una gráfica PNG del resultado (raíces o PVI),
usando matplotlib (ya es una dependencia del proyecto — la usan los
scripts originales Main_*.py — pero no estaba integrada al pipeline
de assistant/). Esta capa es puramente de PRESENTACIÓN, igual que
reporter.py: nunca recalcula nada, solo dibuja valores que ya vienen
en el RootFindingResult / PVIResult.

Se activa con el flag --plot de amn_cli.py; si no se pide, matplotlib
ni siquiera se importa (mantiene amn_cli.py ligero cuando no se
necesita graficar).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import sympy as sp

from .analytical import safe_sympify, UnsafeExpressionError
from .schema import RootFindingResult, PVIResult


def _rango_desde_trace(trace: list) -> tuple:
    """Extrae un rango [min, max] razonable de valores de x a partir
    de las claves numericas que puedan aparecer en el trace de
    cualquiera de los 5 metodos de raices (a, b, r, r_prev, r_curr,
    r_next), con un margen de contexto alrededor."""
    valores = []
    for entry in trace:
        for clave in ("a", "b", "r", "r_prev", "r_curr", "r_next"):
            v = entry.get(clave)
            if v is not None:
                valores.append(float(v))
    if not valores:
        return (-10.0, 10.0)
    lo, hi = min(valores), max(valores)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    margen = (hi - lo) * 0.2
    return (lo - margen, hi + margen)


def plot_root_finding(result: RootFindingResult, out_path: str) -> str:
    """Grafica f(x) junto con las aproximaciones r_n calculadas
    (todas las del trace, no solo las de RN) y la raiz final.
    Guarda un PNG en out_path y retorna la ruta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    problem = result.problem
    x = sp.symbols("x")
    expr = safe_sympify(problem.fx, [x])
    f = sp.lambdify(x, expr, "numpy")

    lo, hi = _rango_desde_trace(result.trace)
    xs = np.linspace(lo, hi, 400)
    ys = f(xs)

    puntos_r = []
    for entry in result.trace:
        candidato = entry.get("r") if "r" in entry else entry.get("r_next")
        if candidato is not None:
            puntos_r.append((entry["n"], float(candidato)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.plot(xs, ys, label=f"f(x) = {problem.fx}")
    if puntos_r:
        rx = [p[1] for p in puntos_r]
        ry = [float(f(v)) for v in rx]
        ax.scatter(rx, ry, color="tab:orange", s=25, zorder=3, label="Aproximaciones r_n")
    if result.r is not None:
        ax.scatter([result.r], [float(f(result.r))], color="tab:red", s=70,
                   zorder=4, marker="*", label=f"Raíz final r ≈ {result.r:.6f}")
    ax.set_title(f"Búsqueda de raíces — {problem.method}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pvi1(result: PVIResult, out_path: str,
              analytical_solution: Optional[str] = None,
              reference_sol=None) -> str:
    """Grafica la solucion numerica (t_n, y_n) del PVI, mas la
    solucion analitica (si se dio) o la referencia de solve_ivp (si
    se calculo con executor.run_pvi_reference / funcsolref) como una
    curva de comparacion. Guarda un PNG en out_path y retorna la
    ruta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    problem = result.problem
    t_arr = np.asarray(result.t, dtype=float)
    y_arr = np.asarray(result.y, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_arr, y_arr, "o-", color="tab:blue", label=f"Numérico ({problem.method})")

    if analytical_solution:
        try:
            t_sym = sp.symbols("t")
            expr = safe_sympify(analytical_solution, [t_sym])
            f_analitica = sp.lambdify(t_sym, expr, "numpy")
            t_denso = np.linspace(t_arr[0], t_arr[-1], 200)
            ax.plot(t_denso, f_analitica(t_denso), "--", color="tab:green",
                    label="Solución analítica")
        except (UnsafeExpressionError, sp.SympifyError):
            pass
    elif reference_sol is not None:
        ax.plot(reference_sol.t, reference_sol.y[0], "--", color="tab:green",
                label="Referencia (solve_ivp)")

    ax.set_title(f"PVI de primer orden — {problem.method}")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
