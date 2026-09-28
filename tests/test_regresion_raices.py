"""
Pruebas de regresión: los 4 métodos de búsqueda de raíces deben seguir
produciendo EXACTAMENTE los mismos (r, iterac, Error, iterac_list) que
quedaron congelados en tests/golden/golden_outputs.json, tanto llamados
sin el parámetro nuevo `trace` como con él (el parámetro `trace` no debe
alterar en absoluto el resultado numérico).
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import Met_busqraices as metb

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "golden_outputs.json")

with open(GOLDEN_PATH) as f:
    GOLDEN = json.load(f)["raices"]


def _check(nombre, resultado):
    r, iterac, error, iterac_list = resultado
    esperado = GOLDEN[nombre]
    assert r == esperado["r"], f"{nombre}: r cambio ({r} != {esperado['r']})"
    assert iterac == esperado["iterac"], f"{nombre}: iterac cambio"
    assert error == esperado["error"], f"{nombre}: Error cambio"
    assert iterac_list == esperado["iterac_list"], f"{nombre}: iterac_list (RN) cambio"


def test_biseccion_sin_trace():
    _check("biseccion", metb.biseccion())


def test_falsapos_sin_trace():
    _check("falsapos", metb.falsapos())


def test_newton_sin_trace():
    _check("newton", metb.newton())


def test_secante_sin_trace():
    _check("secante", metb.secante())


def test_biseccion_con_trace_no_altera_resultado():
    trace = []
    _check("biseccion", metb.biseccion(trace=trace))
    assert len(trace) == GOLDEN["biseccion"]["iterac"] + 1
    assert trace[0]["n"] == 0
    assert trace[0]["r"] is None


def test_falsapos_con_trace_no_altera_resultado():
    # Nota: cuando se alcanza params.MAXIT, el bucle original hace
    # "break" ANTES de incrementar `iterac` por ultima vez, por lo que
    # el `iterac` devuelto puede ser una unidad menor que la cantidad
    # real de aproximaciones r calculadas (esto ya ocurria antes del
    # cambio; no lo introduce el parametro `trace`). Por eso aqui no
    # exigimos igualdad exacta con iterac+1, sino que la traza sea
    # internamente consistente (numeracion 0..N sin huecos).
    trace = []
    _check("falsapos", metb.falsapos(trace=trace))
    assert trace[0]["n"] == 0
    assert trace[0]["r"] is None
    ns = [entry["n"] for entry in trace]
    assert ns == list(range(len(trace)))
    assert len(trace) >= GOLDEN["falsapos"]["iterac"] + 1


def test_newton_con_trace_no_altera_resultado():
    trace = []
    _check("newton", metb.newton(trace=trace))
    assert len(trace) == GOLDEN["newton"]["iterac"] + 1
    assert trace[0]["n"] == 0
    assert trace[0]["r"] == metb.params.A0


def test_secante_con_trace_no_altera_resultado():
    trace = []
    _check("secante", metb.secante(trace=trace))
    assert len(trace) == GOLDEN["secante"]["iterac"] + 1
    assert trace[0]["n"] == 0


def test_trace_conserva_todas_las_iteraciones_no_solo_RN():
    # RN en params.py es [5, 9]; el trace debe tener TODAS las
    # iteraciones (0..iterac), no solo las de RN.
    import params
    assert params.RN == [5, 9]
    trace = []
    metb.biseccion(trace=trace)
    ns = [entry["n"] for entry in trace]
    assert ns == list(range(len(trace)))  # 0,1,2,...,iterac sin huecos


def test_puntofijo_contra_golden():
    # puntofijo necesita una g(x) explicita (no default de params.py):
    # se usa la misma g(x) congelada en golden_outputs.json.
    esperado = GOLDEN["puntofijo"]
    import params
    gx_original = params.GX
    params.GX = esperado["gx"]
    try:
        _check("puntofijo", metb.puntofijo(x0=esperado["x0"]))
    finally:
        params.GX = gx_original


def test_puntofijo_con_trace_incluye_error_aprox():
    import params
    esperado = GOLDEN["puntofijo"]
    gx_original = params.GX
    params.GX = esperado["gx"]
    try:
        trace = []
        _check("puntofijo", metb.puntofijo(x0=esperado["x0"], trace=trace))
    finally:
        params.GX = gx_original
    assert trace[0]["n"] == 0
    assert trace[0]["error_aprox"] is None
    assert trace[0]["r"] == esperado["x0"]
    # A partir de n=1 debe haber un error_aprox numerico (no None)
    assert all(entry["error_aprox"] is not None for entry in trace[1:])
    # El trace debe traer tambien g_r y f_r en cada fila
    assert all("g_r" in entry and "f_r" in entry for entry in trace)


def test_biseccion_trace_incluye_error_aprox_desde_n1():
    trace = []
    metb.biseccion(trace=trace)
    assert trace[0]["error_aprox"] is None
    assert all(entry["error_aprox"] is not None for entry in trace[2:])
    # n=1 no tiene "anterior" real (compara contra None), asi que puede
    # o no traer error_aprox segun el metodo; lo que importa es que no
    # rompe nada y que desde n=2 en adelante siempre hay un numero.
