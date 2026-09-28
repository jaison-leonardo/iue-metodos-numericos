"""
assistant/executor.py
=======================
Adaptador hacia los métodos numéricos EXISTENTES (Met_busqraices.py,
PVI_1Ord/met_solpvi_1ord.py). No reimplementa ningún algoritmo: solo
sabe cómo invocarlos correctamente con los parámetros de un problema
estructurado.

Detalle importante (documentado en la auditoría, sección 5/11):
- Los argumentos NUMÉRICOS de las funciones (ei, ed, tol, nmi, t0, y0,
  n, h, ...) tienen valores por defecto que quedaron "congelados" en
  el momento de importar el módulo. Por eso aquí SIEMPRE se pasan como
  argumentos explícitos (nunca se depende de los defaults).
- La expresión SIMBÓLICA (f(x) o f(t,y)) sí se relee en cada llamada
  desde el módulo de parámetros correspondiente (params.FX /
  params_pvi1.FUNCED), así que para poder ejecutar un ejercicio
  distinto al que trae el repo por defecto, este módulo hace un
  monkeypatch temporal de esa única variable, y la restaura siempre
  en un `finally` (nunca deja el estado del módulo alterado si algo
  falla a mitad de camino).

Limitación conocida y documentada: como params.py / params_pvi1.py son
módulos globales, dos ejecuciones concurrentes en el mismo proceso que
compartan el símbolo FX/FUNCED se pisarían entre sí. Para el uso actual
(una sesión, un ejercicio a la vez) esto es aceptable; si se necesita
paralelismo real, cada ejecución debería correr en su propio proceso.
"""

from __future__ import annotations

import os
import sys
import contextlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PVI1_DIR = os.path.join(REPO_ROOT, "PVI_1Ord")

for _p in (REPO_ROOT, PVI1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import Met_busqraices as metb          # noqa: E402  (import tardío intencional)
import params as root_params            # noqa: E402
import met_solpvi_1ord as pvim          # noqa: E402
import params_pvi1 as pvi1_params       # noqa: E402
import deffuncionespvi as pvi_deff      # noqa: E402  (para run_pvi_reference)

from .schema import RootFindingProblem, PVIProblem, RootFindingResult, PVIResult

_ROOT_METHOD_FUNCS = {
    "biseccion": metb.biseccion,
    "falsapos": metb.falsapos,
    "newton": metb.newton,
    "secante": metb.secante,
    "puntofijo": metb.puntofijo,
}

_PVI1_METHOD_FUNCS = {
    "euler": pvim.euler_meth,
    "heun": pvim.heun_meth,
    "ptomed": pvim.ptomed_meth,
    "rk4": pvim.rk4_meth,
}


@contextlib.contextmanager
def _patched_attr(module, name, value):
    """Cambia temporalmente module.<name> = value y lo restaura siempre."""
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def run_root_finding(problem: RootFindingProblem) -> RootFindingResult:
    func = _ROOT_METHOD_FUNCS[problem.method]
    trace: list = []

    with _patched_attr(root_params, "FX", problem.fx):
        if problem.method in ("biseccion", "falsapos"):
            r, iterac, error, _iterac_list = func(
                ei=problem.a0, ed=problem.b0, tol=problem.tol, nmi=problem.maxit, trace=trace
            )
        elif problem.method == "newton":
            r, iterac, error, _iterac_list = func(
                r=problem.x0, tol=problem.tol, nmi=problem.maxit, trace=trace
            )
        elif problem.method == "secante":
            r, iterac, error, _iterac_list = func(
                r_anterior=problem.x0, r_actual=problem.x1,
                tol=problem.tol, nmi=problem.maxit, trace=trace
            )
        elif problem.method == "puntofijo":
            # puntofijo necesita ADEMAS g(x) (params.GX), no solo f(x)
            # (params.FX se usa igual que en los otros metodos para el
            # residuo |f(r)| que decide cuando detenerse).
            with _patched_attr(root_params, "GX", problem.gx):
                r, iterac, error, _iterac_list = func(
                    x0=problem.x0, tol=problem.tol, nmi=problem.maxit, trace=trace
                )
        else:
            raise ValueError(f"Método de búsqueda de raíces no soportado: {problem.method}")

    return RootFindingResult(problem=problem, r=r, iterac=iterac, error_residual=error, trace=trace)


def run_pvi_first_order(problem: PVIProblem) -> PVIResult:
    func = _PVI1_METHOD_FUNCS[problem.method]
    n = int(round((problem.t_end - problem.t0) / problem.h))

    # evento_y es opcional (parada por evento, ej. "tiempo hasta vaciar
    # el tanque"): cuando el problema no lo trae, evento_info queda en
    # None y los metodos de met_solpvi_1ord.py se comportan EXACTAMENTE
    # como antes (integran los n pasos completos hasta t_end).
    evento_info = {} if getattr(problem, "evento_y", None) is not None else None

    with _patched_attr(pvi1_params, "FUNCED", problem.ode_rhs):
        t, y = func(
            t0=problem.t0, y0=problem.y0, n=n, h=problem.h,
            evento_y=problem.evento_y, evento_info=evento_info,
        )

    evento_info = evento_info or {}
    return PVIResult(
        problem=problem, t=t.tolist(), y=y.tolist(),
        evento_alcanzado=evento_info.get("alcanzado"),
        t_evento=evento_info.get("t_evento"),
        paso_evento=evento_info.get("paso_evento"),
        evento_aproximado=evento_info.get("aproximado"),
    )


def run_pvi_reference_full(problem: PVIProblem):
    """
    Solución de referencia COMPLETA (objeto de scipy.integrate.solve_ivp,
    con .t / .y[0] de la trayectoria) para cuando el ejercicio NO da una
    solución analítica (sección 17/19 de la especificación). Reutiliza
    PVI_1Ord/deffuncionespvi.py::funcsolref, que ya existía en el repo
    original (usado por los scripts Main_*.py para graficar la
    referencia) pero no estaba conectado al pipeline de assistant/.
    """
    with _patched_attr(pvi1_params, "FUNCED", problem.ode_rhs):
        return pvi_deff.funcsolref(problem.t0, problem.t_end, problem.y0)


def run_pvi_reference(problem: PVIProblem) -> float:
    """Igual que run_pvi_reference_full, pero retorna solo el valor
    final y(t_end) (lo que necesita comparator.compare_pvi)."""
    sol = run_pvi_reference_full(problem)
    return float(sol.y[0][-1])
