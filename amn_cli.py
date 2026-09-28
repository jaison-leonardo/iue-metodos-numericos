"""
amn_cli.py
===========
Punto de entrada UNICO del asistente AMN. Encadena, en un solo comando
en vez de scripts de ejemplo sueltos:

    assistant/parser.py (si hay LLM configurado)
        -> assistant/validator.py
        -> assistant/executor.py
        -> assistant/verifier.py / assistant/comparator.py
        -> assistant/reporter.py

Modos de uso (ejecutar desde la raiz del repo):

    python amn_cli.py
        Menu interactivo: pegar un enunciado en lenguaje natural (usa
        el LLM configurado en .env) o ingresar el problema a mano
        (sirve tambien si no hay ningun proveedor LLM disponible).

    python amn_cli.py --text "enunciado en lenguaje natural..."
        Una sola pasada sin menu (util para probar varios ejercicios
        rapido, p.ej. desde otro script o copiando/pegando).

    python amn_cli.py --text ejercicio.txt
        Igual que arriba, pero si el valor de --text es la ruta a un
        archivo existente (.txt, .md, etc.), se lee su contenido como
        el enunciado en lugar de tratarlo como texto literal. Util
        para guardar varios ejercicios en archivos y probarlos uno
        por uno sin tener que pegarlos en la terminal.

    python amn_cli.py --json ejercicio.json
        Problema YA estructurado (sin pasar por el LLM). El JSON debe
        tener "problem_type": "root_finding" o "pvi_first_order" mas
        los campos de RootFindingProblem / PVIProblem
        (ver assistant/schema.py).

Flag adicional, combinable con cualquiera de los modos anteriores:

    --plot [--plot-out archivo.png]
        Ademas del reporte de consola, genera una grafica PNG:
        f(x) con las aproximaciones r_n para raices, o y(t) numerico
        vs. la solucion analitica/de referencia para un PVI. Si no se
        da --plot-out, se guarda como amn_plot_<metodo>.png en el
        directorio actual.

Para un PVI, se puede pedir "parada por evento" (ej. "tiempo hasta
vaciar el tanque"): si el problema estructurado trae "evento_y", la
integracion se detiene apenas y cruza ese valor (en vez de siempre
llegar hasta t_end), reportando el instante interpolado del cruce. Es
opcional: si no se da, el comportamiento es igual al de siempre.

Este script NO reimplementa ningun calculo: solo encadena los modulos
de assistant/ que ya existen y que ya tienen sus propias pruebas
(tests/test_assistant_pipeline.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp

from assistant.schema import RootFindingProblem, PVIProblem, ROOT_METHODS, PVI1_METHODS
from assistant.validator import validate_root_finding_problem, validate_pvi_problem
from assistant.executor import (
    run_root_finding, run_pvi_first_order, run_pvi_reference, run_pvi_reference_full,
)
from assistant.verifier import verify_root, verify_ode_solution
from assistant.comparator import compare_root, compare_pvi
from assistant.reporter import render_root_finding_report, render_pvi1_report
from assistant.parser import parse_exercise
from assistant.llm.router import LLMRouter
from assistant.analytical import safe_sympify, UnsafeExpressionError
from assistant.plotter import plot_root_finding, plot_pvi1


# ---------------------------------------------------------------------------
# Entrada de datos (modo manual / interactivo)
# ---------------------------------------------------------------------------

def _leer_multilinea(mensaje: str) -> str:
    print(mensaje)
    print("(termina con una linea vacia)")
    lineas = []
    while True:
        try:
            linea = input()
        except EOFError:
            break
        if linea.strip() == "":
            break
        lineas.append(linea)
    return "\n".join(lineas)


def _pedir_float(mensaje: str, default=None):
    sufijo = f" [{default}]: " if default is not None else ": "
    texto = input(f"{mensaje}{sufijo}").strip()
    if not texto:
        return default
    return float(texto)


def _pedir_int_lista(mensaje: str, default):
    texto = input(f"{mensaje} [{','.join(map(str, default))}]: ").strip()
    if not texto:
        return list(default)
    return [int(x.strip()) for x in texto.split(",") if x.strip() != ""]


def construir_root_finding_manual() -> RootFindingProblem:
    print("\n--- Busqueda de raices (manual) ---")
    fx = input("f(x) = ").strip()
    print(f"Metodos disponibles: {ROOT_METHODS}")
    method = input("Metodo: ").strip().lower()

    problem = RootFindingProblem(fx=fx, method=method)
    if method in ("biseccion", "falsapos"):
        problem.a0 = _pedir_float("a0 (extremo izquierdo)")
        problem.b0 = _pedir_float("b0 (extremo derecho)")
    elif method == "newton":
        problem.x0 = _pedir_float("x0 (aproximacion inicial)")
    elif method == "secante":
        problem.x0 = _pedir_float("x0 (primera aproximacion)")
        problem.x1 = _pedir_float("x1 (segunda aproximacion)")
    elif method == "puntofijo":
        problem.x0 = _pedir_float("x0 (aproximacion inicial)")
        problem.gx = input("g(x) tal que x = g(x): ").strip()

    problem.tol = _pedir_float("Tolerancia", 1e-6)
    problem.maxit = int(_pedir_float("Maximo de iteraciones", 50))
    problem.rn = _pedir_int_lista("RN (iteraciones a mostrar, separadas por coma)", [0])
    return problem


def construir_pvi_manual() -> PVIProblem:
    print("\n--- PVI de primer orden (manual) ---")
    ode_rhs = input("f(t, y) = ").strip()
    print(f"Metodos disponibles: {PVI1_METHODS}")
    method = input("Metodo: ").strip().lower()

    t0 = _pedir_float("t0")
    y0 = _pedir_float("y0")
    t_end = _pedir_float("t_end")
    h = _pedir_float("h (tamano de paso)")
    analitica = input("Solucion analitica y(t) (opcional, Enter para omitir): ").strip() or None

    evento_txt = input(
        "Detener antes de t_end si y llega a un valor objetivo "
        "(parada por evento, ej. 0 para \"tiempo hasta vaciar el tanque\"; "
        "Enter para omitir): "
    ).strip()
    evento_y = float(evento_txt) if evento_txt else None

    problem = PVIProblem(
        ode_rhs=ode_rhs, method=method, t0=t0, y0=y0, t_end=t_end, h=h,
        analytical_solution=analitica, evento_y=evento_y,
    )
    problem.rn = _pedir_int_lista("RN (indices a mostrar, separados por coma)", [0])
    return problem


# ---------------------------------------------------------------------------
# Pipeline: validator -> executor -> verifier/comparator -> reporter
# ---------------------------------------------------------------------------

def _evaluar_analitica_en(expr_str: str, t_valor: float) -> float:
    t = sp.symbols("t")
    expr = safe_sympify(expr_str, [t])
    return float(expr.subs(t, t_valor))


def procesar_root_finding(problem: RootFindingProblem, generar_grafica: bool = False, plot_out: str = None) -> None:
    validacion = validate_root_finding_problem(problem)
    if not validacion.ok:
        print("\nPROBLEMA INVALIDO:")
        for err in validacion.errors:
            print(f"  - {err}")
        return

    resultado = run_root_finding(problem)
    verificacion = verify_root(problem.fx, resultado.r, problem.tol)
    comparacion = compare_root(resultado.r, problem.fx, problem.tol)

    print()
    print(render_root_finding_report(resultado, comparacion))
    print()
    print(f"VERIFICACION INDEPENDIENTE (verifier.py): {verificacion['details']}")

    if generar_grafica:
        ruta = plot_out or f"amn_plot_{problem.method}.png"
        try:
            ruta_final = plot_root_finding(resultado, ruta)
            print(f"\nGrafica guardada en: {ruta_final}")
        except Exception as e:
            print(f"\n(No se pudo generar la grafica: {e})")


def procesar_pvi(problem: PVIProblem, generar_grafica: bool = False, plot_out: str = None) -> None:
    validacion = validate_pvi_problem(problem)
    if not validacion.ok:
        print("\nPROBLEMA INVALIDO:")
        for err in validacion.errors:
            print(f"  - {err}")
        return

    resultado = run_pvi_first_order(problem)
    comparacion = None
    referencia_completa = None  # objeto solve_ivp (.t/.y), solo para la grafica

    if resultado.evento_alcanzado:
        # La integracion se detuvo en el evento (antes de t_end), asi
        # que comparar contra una solucion analitica/solve_ivp evaluada
        # en t_end ya no es correcto (serian instantes distintos). El
        # reporte ya muestra la seccion EVENTO con el detalle.
        print(
            "\n(Se alcanzo la parada por evento antes de t_end: no se "
            "compara contra solucion analitica/solve_ivp evaluada en "
            "t_end, ver seccion EVENTO en el reporte.)"
        )
    elif problem.analytical_solution:
        verificacion = verify_ode_solution(
            problem.ode_rhs, problem.analytical_solution, problem.t0, problem.y0
        )
        print(f"\nVERIFICACION de la solucion analitica (verifier.py): {verificacion['details']}")
        if verificacion["verified"]:
            try:
                analitica_final = _evaluar_analitica_en(problem.analytical_solution, problem.t_end)
                comparacion = compare_pvi(resultado.y[-1], analytical_final=analitica_final)
            except (UnsafeExpressionError, sp.SympifyError, Exception) as e:
                print(f"  (no se pudo evaluar la solucion analitica en t_end: {e})")
        else:
            print("  La solucion analitica proporcionada NO paso la verificacion: no se usara como referencia.")
    else:
        # Sin solucion analitica: calcular una referencia de alta
        # precision con scipy.integrate.solve_ivp (mismo criterio que
        # el ejecutor de raices usa verifier/comparator: nunca se
        # reporta "convergido" sin una referencia real).
        try:
            referencia_completa = run_pvi_reference_full(problem)
            comparacion = compare_pvi(resultado.y[-1], reference_final=float(referencia_completa.y[0][-1]))
            print(
                "\n(Sin solucion analitica: se calculo una referencia de alta "
                "precision con scipy.integrate.solve_ivp para comparar.)"
            )
        except Exception as e:
            print(f"\n(No se pudo calcular una referencia con solve_ivp: {e})")

    print()
    print(render_pvi1_report(resultado, comparacion))

    if generar_grafica:
        ruta = plot_out or f"amn_plot_{problem.method}.png"
        try:
            ruta_final = plot_pvi1(
                resultado, ruta,
                analytical_solution=problem.analytical_solution,
                reference_sol=referencia_completa,
            )
            print(f"\nGrafica guardada en: {ruta_final}")
        except Exception as e:
            print(f"\n(No se pudo generar la grafica: {e})")


def procesar_desde_texto(texto: str, router: LLMRouter, generar_grafica: bool = False, plot_out: str = None) -> None:
    tipo, obj = parse_exercise(texto, router=router)

    if tipo == "no_llm":
        print(
            "\nNingun proveedor LLM esta disponible (revisa tu .env). "
            "Puedes ingresar el problema manualmente (opcion 2 del menu)."
        )
        return
    if tipo == "needs_clarification":
        print(f"\nEl asistente necesita mas informacion: {obj.reason}")
        if obj.missing_information:
            print("Falta:")
            for item in obj.missing_information:
                print(f"  - {item}")
        return
    if tipo == "root_finding":
        procesar_root_finding(obj, generar_grafica=generar_grafica, plot_out=plot_out)
    elif tipo == "pvi_first_order":
        procesar_pvi(obj, generar_grafica=generar_grafica, plot_out=plot_out)


# ---------------------------------------------------------------------------
# Menu interactivo / CLI
# ---------------------------------------------------------------------------

def menu_interactivo() -> None:
    router = LLMRouter()
    while True:
        print("\n" + "=" * 70)
        print("AMN - Asistente Automatizado de Metodos Numericos")
        print("=" * 70)
        print("1) Pegar enunciado en lenguaje natural (usa el LLM configurado)")
        print("2) Ingresar el problema manualmente (busqueda de raices o PVI)")
        print("3) Salir")
        opcion = input("Opcion: ").strip()

        if opcion == "1":
            texto = _leer_multilinea("\nPega el enunciado del ejercicio:")
            if texto.strip():
                grafica = input("¿Generar tambien una grafica PNG? (s/N): ").strip().lower() == "s"
                procesar_desde_texto(texto, router, generar_grafica=grafica)
        elif opcion == "2":
            print("\na) Busqueda de raices")
            print("b) PVI de primer orden")
            sub = input("Opcion: ").strip().lower()
            if sub == "a":
                problema = construir_root_finding_manual()
                grafica = input("¿Generar tambien una grafica PNG? (s/N): ").strip().lower() == "s"
                procesar_root_finding(problema, generar_grafica=grafica)
            elif sub == "b":
                problema = construir_pvi_manual()
                grafica = input("¿Generar tambien una grafica PNG? (s/N): ").strip().lower() == "s"
                procesar_pvi(problema, generar_grafica=grafica)
            else:
                print("Opcion no reconocida.")
        elif opcion == "3":
            break
        else:
            print("Opcion no reconocida.")


def main() -> None:
    cli = argparse.ArgumentParser(description="AMN - punto de entrada unico del asistente.")
    cli.add_argument("--text", help="Enunciado en lenguaje natural (una sola pasada, sin menu). "
                                     "Si es la ruta a un archivo existente, se lee su contenido.")
    cli.add_argument("--json", help="Ruta a un JSON con el problema ya estructurado (sin LLM).")
    cli.add_argument("--plot", action="store_true",
                      help="Ademas del reporte de consola, guarda una grafica PNG del resultado.")
    cli.add_argument("--plot-out", help="Ruta del PNG a generar (por defecto: amn_plot_<metodo>.png).")
    args = cli.parse_args()

    if args.text:
        texto = args.text
        if os.path.isfile(texto):
            with open(texto, "r", encoding="utf-8") as f:
                texto = f.read()
        procesar_desde_texto(texto, LLMRouter(), generar_grafica=args.plot, plot_out=args.plot_out)
        return

    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        tipo = data.pop("problem_type", None)
        if tipo == "root_finding":
            procesar_root_finding(RootFindingProblem(**data), generar_grafica=args.plot, plot_out=args.plot_out)
        elif tipo == "pvi_first_order":
            procesar_pvi(PVIProblem(**data), generar_grafica=args.plot, plot_out=args.plot_out)
        else:
            print('El JSON debe tener "problem_type": "root_finding" o "pvi_first_order".')
        return

    menu_interactivo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
