"""
Ejemplo end-to-end (modo determinista, sin LLM) usando el ejercicio de
la sección 5 de la especificación: decaimiento de un contaminante
radiactivo, dc/dt = -k*c, resuelto con el método de Euler.

Ejecutar desde la raíz del repositorio:
    python assistant/examples/demo_reactor.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from assistant.schema import PVIProblem
from assistant.validator import validate_pvi_problem
from assistant.executor import run_pvi_first_order
from assistant.verifier import verify_ode_solution
from assistant.comparator import compare_pvi
from assistant.reporter import render_pvi1_report

import sympy as sp


def main():
    problem = PVIProblem(
        ode_rhs="-0.175*y",
        method="euler",
        t0=0.0, y0=100.0, t_end=1.0, h=0.1,
        analytical_solution="100*exp(-0.175*t)",
        rn=[0, 1, 2, 5, 10],
        display_precision=4,
    )

    validacion = validate_pvi_problem(problem)
    validacion.raise_if_invalid()

    verificacion = verify_ode_solution(
        problem.ode_rhs, problem.analytical_solution, problem.t0, problem.y0
    )
    if not verificacion["verified"]:
        print("ADVERTENCIA: la solución analítica proporcionada no pasó la verificación SymPy:")
        print(" ", verificacion["details"])

    resultado = run_pvi_first_order(problem)

    t = sp.symbols("t")
    valor_analitico_final = float(sp.sympify(problem.analytical_solution).subs(t, problem.t_end))
    comparacion = compare_pvi(resultado.y[-1], analytical_final=valor_analitico_final)

    print(render_pvi1_report(resultado, comparison=comparacion))


if __name__ == "__main__":
    main()
