"""
Pruebas de la capa `assistant/` (nueva), de punta a punta en modo
determinista (sin LLM): construir el problema estructurado a mano,
validar, ejecutar, verificar/comparar y generar el reporte, usando los
métodos numéricos EXISTENTES sin modificarlos.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

from assistant.schema import RootFindingProblem, PVIProblem
from assistant.validator import validate_root_finding_problem, validate_pvi_problem
from assistant.executor import (
    run_root_finding, run_pvi_first_order, run_pvi_reference, run_pvi_reference_full,
)
from assistant.analytical import UnsafeExpressionError, safe_sympify
from assistant.verifier import verify_ode_solution, verify_root
from assistant.comparator import compare_root, compare_pvi
from assistant.reporter import render_root_finding_report, render_pvi1_report
from assistant.config import get_config
from assistant.llm.router import LLMRouter
from assistant.llm.none_provider import NoneProvider
from assistant.parser import parse_exercise, _extract_json
from assistant.llm.gemini_provider import GeminiProvider
from assistant.llm.base import LLMProviderError
import math as _math


# --- validator.py -----------------------------------------------------

def test_validator_rechaza_intervalo_sin_cambio_de_signo():
    p = RootFindingProblem(fx="x**2 + 1", method="biseccion", a0=0, b0=1)
    v = validate_root_finding_problem(p)
    assert not v.ok
    assert any("cambio de signo" in e for e in v.errors)


def test_validator_rechaza_metodo_desconocido():
    p = RootFindingProblem(fx="x-1", method="metodo_inventado", a0=0, b0=2)
    v = validate_root_finding_problem(p)
    assert not v.ok


def test_validator_exige_x0_para_newton():
    p = RootFindingProblem(fx="x-1", method="newton")
    v = validate_root_finding_problem(p)
    assert not v.ok
    assert any("x0" in e for e in v.errors)


def test_validator_pvi_exige_h_positivo():
    p = PVIProblem(ode_rhs="-y", method="euler", t0=0, y0=1, t_end=1, h=0)
    v = validate_pvi_problem(p)
    assert not v.ok


def test_validator_puntofijo_exige_gx():
    p = RootFindingProblem(fx="x**2 - 18", method="puntofijo", x0=4)
    v = validate_root_finding_problem(p)
    assert not v.ok
    assert any("g(x)" in e for e in v.errors)


def test_validator_puntofijo_ok_con_gx_valida():
    p = RootFindingProblem(fx="x**2 - 18", method="puntofijo", x0=4, gx="(x + 18/x)/2")
    v = validate_root_finding_problem(p)
    assert v.ok


# --- analytical.py: seguridad (Fase 22, nunca eval()) ------------------

def test_safe_sympify_rechaza_simbolo_no_declarado():
    import sympy as sp
    x = sp.symbols("x")
    with pytest.raises(UnsafeExpressionError):
        safe_sympify("x + y", [x])  # 'y' no está en la lista de símbolos permitidos


def test_safe_sympify_rechaza_funcion_no_permitida():
    import sympy as sp
    x = sp.symbols("x")
    with pytest.raises(UnsafeExpressionError):
        safe_sympify("gamma(x)", [x])  # no está en ALLOWED_FUNCTION_NAMES


def test_safe_sympify_acepta_funciones_permitidas():
    import sympy as sp
    x = sp.symbols("x")
    expr = safe_sympify("sin(x) + sqrt(x) + exp(x)", [x])
    assert expr is not None


# --- executor.py: no debe dejar el estado global alterado --------------

def test_executor_no_deja_params_alterado():
    import params
    fx_original = params.FX
    p = RootFindingProblem(fx="x**2 - 4", method="biseccion", a0=0, b0=3, rn=[0])
    run_root_finding(p)
    assert params.FX == fx_original


def test_executor_root_finding_produce_traza_y_resultado_correcto():
    p = RootFindingProblem(fx="x**2 - 4", method="biseccion", a0=0, b0=3, tol=1e-6, maxit=50, rn=[0, 1])
    resultado = run_root_finding(p)
    assert abs(resultado.r - 2.0) < 1e-5
    assert resultado.trace[0]["n"] == 0
    assert resultado.trace[-1]["n"] == resultado.iterac


def test_executor_pvi_produce_arreglo_con_condicion_inicial_en_indice_0():
    p = PVIProblem(ode_rhs="-y", method="euler", t0=0, y0=1, t_end=1, h=0.5)
    resultado = run_pvi_first_order(p)
    assert resultado.t[0] == 0
    assert resultado.y[0] == 1


def test_executor_puntofijo_converge_y_reporta_error_aprox():
    p = RootFindingProblem(
        fx="x**2 - 18", method="puntofijo", x0=4, gx="(x + 18/x)/2",
        tol=1e-6, maxit=50, rn=[0, 1],
    )
    resultado = run_root_finding(p)
    assert abs(resultado.r - 18 ** 0.5) < 1e-5
    assert resultado.trace[0]["error_aprox"] is None
    assert resultado.trace[-1]["error_aprox"] is not None


def test_executor_no_deja_params_gx_alterado():
    import params
    gx_original = params.GX
    p = RootFindingProblem(fx="x**2 - 18", method="puntofijo", x0=4, gx="(x + 18/x)/2")
    run_root_finding(p)
    assert params.GX == gx_original


def test_run_pvi_reference_cercano_a_analitica_conocida():
    # y' = -0.175*y, y(0)=100 -> y(1) = 100*exp(-0.175) (analitica),
    # solve_ivp deberia acercarse mucho a ese valor de alta precision.
    import math
    p = PVIProblem(ode_rhs="-0.175*y", method="rk4", t0=0, y0=100, t_end=1, h=0.5)
    referencia = run_pvi_reference(p)
    esperado = 100 * math.exp(-0.175)
    assert abs(referencia - esperado) < 1e-4


def test_run_pvi_reference_full_expone_trayectoria():
    p = PVIProblem(ode_rhs="-0.175*y", method="rk4", t0=0, y0=100, t_end=1, h=0.5)
    sol = run_pvi_reference_full(p)
    assert sol.t[0] == 0
    assert abs(sol.y[0][0] - 100) < 1e-9
    assert len(sol.t) == len(sol.y[0])


# --- verifier.py ---------------------------------------------------------

def test_verify_ode_solution_detecta_solucion_correcta():
    v = verify_ode_solution("-0.175*y", "100*exp(-0.175*t)", t0=0, y0=100)
    assert v["verified"]


def test_verify_ode_solution_detecta_solucion_incorrecta():
    v = verify_ode_solution("-0.175*y", "100*exp(-0.2*t)", t0=0, y0=100)
    assert not v["verified"]


def test_verify_root():
    v = verify_root("x**2 - 4", 2.0, tol=1e-6)
    assert v["verified"]
    v2 = verify_root("x**2 - 4", 3.0, tol=1e-6)
    assert not v2["verified"]


# --- comparator.py ---------------------------------------------------------

def test_compare_root_con_referencia():
    c = compare_root(2.00001, "x**2-4", tol=1e-4, reference_root=2.0)
    assert c.converged
    assert c.absolute_error < 1e-4


def test_compare_pvi_prioriza_analitica_sobre_referencia():
    c = compare_pvi(numeric_final=10.0, analytical_final=10.5, reference_final=999)
    assert c.reference_source == "analytical"
    assert c.reference_value == 10.5


# --- reporter.py: RN controla que se muestra, no que se calcula --------

def test_reporter_respeta_rn_sin_alterar_calculo():
    p = RootFindingProblem(fx="x**2 - 4", method="biseccion", a0=0, b0=3, tol=1e-6, maxit=50, rn=[0])
    resultado = run_root_finding(p)
    texto = render_root_finding_report(resultado)
    # Solo debe aparecer una fila de datos (n=0), aunque se calcularon muchas mas
    assert resultado.iterac > 1
    assert "ITERACIONES SELECCIONADAS (RN = [0])" in texto


# --- config.py / llm: modo sin LLM siempre disponible -------------------

def test_none_provider_siempre_disponible():
    assert NoneProvider().is_available()


def test_router_cae_a_none_sin_api_keys(monkeypatch):
    # Sin ninguna cadena de proveedores configurada (ni API keys
    # presentes), el router debe caer siempre al modo determinista.
    monkeypatch.delenv("AMN_PRIMARY_PROVIDER", raising=False)
    monkeypatch.delenv("AMN_FALLBACK_PROVIDERS", raising=False)
    for var in (
        "API_KEY_GEMINI_JAISON", "API_KEY_GEMINI_JAIME",
        "GROQ_API_KEY_JAISON", "OPENROUTER_API_KEY_JAISON",
    ):
        monkeypatch.delenv(var, raising=False)
    config = get_config()
    assert config.chain_tokens == []
    router = LLMRouter(config)
    assert router.active_provider_name() == "none"


def test_router_usa_cadena_configurada_en_env(monkeypatch):
    # Con la cadena real del .env, el primer eslabon disponible debe
    # ser gemini con la cuenta de Jaime (segun lo pedido).
    config = get_config()
    if not config.chain_tokens:
        pytest.skip("No hay AMN_PRIMARY_PROVIDER configurado en este entorno")
    router = LLMRouter(config)
    nombres = [p.label for p in router.chain]
    assert nombres[0].startswith("gemini")
    assert "JAIME" in nombres[0] or len(nombres) == 1



# --- plotter.py (recomendacion 4: grafica opcional, --plot) --------------

def test_plot_root_finding_genera_png(tmp_path):
    from assistant.plotter import plot_root_finding
    p = RootFindingProblem(fx="x**2 - 4", method="biseccion", a0=0, b0=3, tol=1e-6, maxit=50, rn=[0, 1])
    resultado = run_root_finding(p)
    out = tmp_path / "raiz.png"
    ruta = plot_root_finding(resultado, str(out))
    assert os.path.isfile(ruta)
    assert os.path.getsize(ruta) > 0


def test_plot_pvi1_genera_png_con_analitica(tmp_path):
    from assistant.plotter import plot_pvi1
    p = PVIProblem(
        ode_rhs="-0.175*y", method="rk4", t0=0, y0=100, t_end=1, h=0.5,
        analytical_solution="100*exp(-0.175*t)",
    )
    resultado = run_pvi_first_order(p)
    out = tmp_path / "pvi.png"
    ruta = plot_pvi1(resultado, str(out), analytical_solution=p.analytical_solution)
    assert os.path.isfile(ruta)
    assert os.path.getsize(ruta) > 0


def test_plot_pvi1_genera_png_con_referencia_solve_ivp(tmp_path):
    from assistant.plotter import plot_pvi1
    p = PVIProblem(ode_rhs="t**2 - 2*y", method="rk4", t0=0, y0=1, t_end=3, h=0.5)
    resultado = run_pvi_first_order(p)
    referencia = run_pvi_reference_full(p)
    out = tmp_path / "pvi_ref.png"
    ruta = plot_pvi1(resultado, str(out), reference_sol=referencia)
    assert os.path.isfile(ruta)
    assert os.path.getsize(ruta) > 0


# ---------------------------------------------------------------------------
# Parada por evento en PVI (sesion 5, recomendacion 5): ej. "tiempo hasta
# vaciar el tanque". evento_y es opcional (default None); estas pruebas
# cubren el wiring completo validator -> executor -> reporter.
# ---------------------------------------------------------------------------

def test_validator_pvi_rechaza_evento_igual_a_y0():
    p = PVIProblem(ode_rhs="-y", method="rk4", t0=0, y0=100, t_end=10, h=0.1, evento_y=100)
    resultado = validate_pvi_problem(p)
    assert not resultado.ok
    assert any("evento_y" in e for e in resultado.errors)


def test_validator_pvi_acepta_evento_y_valido():
    p = PVIProblem(ode_rhs="-y", method="rk4", t0=0, y0=100, t_end=10, h=0.1, evento_y=10)
    resultado = validate_pvi_problem(p)
    assert resultado.ok


def test_executor_pvi_con_evento_trunca_antes_de_t_end():
    # y' = -y, y0=100, evento_y=10 => se vacia (llega a 10) mucho antes
    # de t_end=50 (que aqui es solo una cota superior amplia).
    p = PVIProblem(ode_rhs="-y", method="rk4", t0=0, y0=100, t_end=50, h=0.5, evento_y=10)
    resultado = run_pvi_first_order(p)
    assert resultado.evento_alcanzado is True
    assert resultado.t_evento is not None
    assert resultado.t[-1] < p.t_end  # se detuvo antes de la cota superior
    import math
    assert abs(resultado.t_evento - math.log(10)) < 0.1


def test_executor_pvi_sin_evento_no_se_ve_afectado():
    # Mismo problema pero sin evento_y: debe integrar hasta t_end como
    # siempre (comportamiento identico al que ya estaba probado antes
    # de esta sesion).
    p = PVIProblem(ode_rhs="-y", method="rk4", t0=0, y0=100, t_end=5, h=0.5)
    resultado = run_pvi_first_order(p)
    assert resultado.evento_alcanzado is None
    assert resultado.t_evento is None
    assert abs(resultado.t[-1] - 5.0) < 1e-9


def test_executor_pvi_evento_no_alcanzado_dentro_de_t_end():
    # evento_y fuera de rango alcanzable en el intervalo dado: debe
    # completar la integracion normalmente y reportar alcanzado=False.
    p = PVIProblem(ode_rhs="-y", method="rk4", t0=0, y0=100, t_end=2, h=0.5, evento_y=1e-6)
    resultado = run_pvi_first_order(p)
    assert resultado.evento_alcanzado is False
    assert resultado.t_evento is None
    assert abs(resultado.t[-1] - 2.0) < 1e-9


def test_reporter_pvi_muestra_seccion_evento_cuando_se_alcanza():
    p = PVIProblem(ode_rhs="-y", method="rk4", t0=0, y0=100, t_end=50, h=0.5, evento_y=10)
    resultado = run_pvi_first_order(p)
    reporte = render_pvi1_report(resultado, comparison=None)
    assert "EVENTO ALCANZADO" in reporte
    assert "t_end" in reporte  # menciona explicitamente que no llego a t_end


def test_reporter_pvi_sin_evento_no_muestra_seccion_evento():
    p = PVIProblem(ode_rhs="-y", method="rk4", t0=0, y0=100, t_end=5, h=0.5)
    resultado = run_pvi_first_order(p)
    reporte = render_pvi1_report(resultado, comparison=None)
    assert "EVENTO ALCANZADO" not in reporte


# ---------------------------------------------------------------------------
# _extract_json / parse_exercise: robustez ante respuestas de LLM que no
# son JSON limpio (sesion 6, bug real reportado por Jaison: un enunciado
# con formulas en LaTeX -- ej. "e^{-c*t/m}" del problema del
# paracaidista -- puede hacer que un LLM (sobre todo uno "de
# razonamiento" gratuito) devuelva llaves de LaTeX antes del JSON, o a
# veces ni siquiera devuelva JSON. Antes esto crasheaba amn_cli.py con
# un traceback sin manejar; ahora debe reportarse como
# needs_clarification, nunca como una excepcion sin capturar.
# ---------------------------------------------------------------------------

class _RouterFalso:
    """Stub de LLMRouter: devuelve siempre el mismo texto, sin red."""
    def __init__(self, texto, provider="stub"):
        self._texto = texto
        self._provider = provider

    def complete(self, prompt, max_tokens=1024):
        return {"provider": self._provider, "text": self._texto}


def test_extract_json_ignora_llaves_de_latex_antes_del_json():
    # Caso real: el modelo repite la formula del enunciado (con llaves
    # de LaTeX, ej. e^{-c*t/m}) antes de escribir el JSON pedido.
    texto = (
        "Planteo f(m) = (g*m/c)*(1-e^{-c*t/m}) - 36, con g=9.81, c=15, t=10.\n\n"
        '```json\n'
        '{"problem_type": "root_finding", "fx": "(9.81*x/15)*(1-exp(-15*10/x))-36", '
        '"method": "falsapos", "a0": 50, "b0": 100, "tol": 0.00001, "maxit": 50, "rn": [0]}\n'
        '```'
    )
    data = _extract_json(texto)
    assert data["problem_type"] == "root_finding"
    assert data["method"] == "falsapos"


def test_extract_json_sin_ninguna_llave_da_error_claro():
    with pytest.raises(ValueError, match="JSON reconocible"):
        _extract_json("Lo siento, no puedo ayudar con eso en este formato.")


def test_parse_exercise_con_respuesta_no_json_da_needs_clarification_no_crash():
    # Antes de este fix, esto lanzaba una excepcion sin manejar hasta
    # amn_cli.py. Ahora debe degradarse a needs_clarification.
    router = _RouterFalso("Disculpa, no puedo generar una respuesta estructurada ahora mismo.")
    tipo, obj = parse_exercise("cualquier enunciado de prueba", router=router)
    assert tipo == "needs_clarification"
    assert "stub" in obj.reason
    assert obj.missing_information  # sugiere reintentar o entrada manual


def test_parse_exercise_con_latex_en_la_respuesta_funciona_bien():
    router = _RouterFalso(
        "f(m) = (g*m/c)*(1-e^{-c*t/m}) - 36\n\n"
        '{"problem_type": "root_finding", "fx": "(9.81*x/15)*(1-exp(-15*10/x))-36", '
        '"method": "falsapos", "a0": 50, "b0": 100, "tol": 0.00001, "maxit": 50, "rn": [0]}'
    )
    tipo, obj = parse_exercise("problema del paracaidista", router=router)
    assert tipo == "root_finding"
    assert obj.method == "falsapos"


# ---------------------------------------------------------------------------
# gemini_provider.py: deteccion de respuesta truncada por MAX_TOKENS
# (sesion 7, bug real reportado por Jaison: un PVI con un LLM de
# razonamiento gratuito devolvio un JSON cortado a mitad de camino,
# ej. '...{"ode_rhs": "-0.08 * 4 * pi' sin cerrar, porque el modelo gasto
# el presupuesto de tokens pensando internamente antes de terminar de
# escribir la respuesta). Antes esto se colaba como texto truncado hasta
# _extract_json, que fallaba con un mensaje generico sin indicar la
# causa real. Ahora gemini_provider.py debe detectarlo el mismo y
# levantar LLMProviderError, para que LLMRouter pase al siguiente
# proveedor de la cadena en vez de devolver texto incompleto.
# ---------------------------------------------------------------------------

def _respuesta_gemini_falsa(finish_reason, texto):
    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"candidates": [{
                "finishReason": finish_reason,
                "content": {"parts": [{"text": texto}]},
            }]}
    return _Resp()


def test_gemini_provider_detecta_max_tokens_truncado(monkeypatch):
    import requests as requests_real
    provider = GeminiProvider(api_key="fake-key")
    respuesta_truncada = (
        '```json\n{\n  "problem_type": "pvi_first_order",\n'
        '  "ode_rhs": "-0.08 * 4 * pi'
    )
    monkeypatch.setattr(
        requests_real, "post",
        lambda *a, **k: _respuesta_gemini_falsa("MAX_TOKENS", respuesta_truncada),
    )
    with pytest.raises(LLMProviderError, match="MAX_TOKENS"):
        provider.complete("prompt de prueba")


def test_gemini_provider_respuesta_completa_normal(monkeypatch):
    import requests as requests_real
    provider = GeminiProvider(api_key="fake-key")
    respuesta_ok = '{"problem_type": "root_finding", "fx": "x**2-4"}'
    monkeypatch.setattr(
        requests_real, "post",
        lambda *a, **k: _respuesta_gemini_falsa("STOP", respuesta_ok),
    )
    assert provider.complete("prompt de prueba") == respuesta_ok


# ---------------------------------------------------------------------------
# parser.py + ode_derivation.py: cuando el LLM da la tasa de cambio en
# terminos de variables auxiliares (ej. area/volumen/radio) en vez de
# derivarla el mismo (sesion 8: el ejercicio real de la gota que se
# evapora hacia que el LLM gastara tokens de razonamiento en el algebra
# y a veces se quedara sin terminar -- ver el bug de MAX_TOKENS de
# arriba). Ahora esa eliminacion la hace sympy de forma deterministica.
# ---------------------------------------------------------------------------

def test_parse_exercise_deriva_ode_rhs_desde_relaciones_auxiliares():
    r0 = 2.5
    V0 = (4/3) * _math.pi * r0**3
    router = _RouterFalso(
        '```json\n{'
        '"problem_type": "pvi_first_order", "ode_rhs": null, '
        '"rate_expr": "-k*A", '
        '"auxiliary_relations": ["A = 4*pi*r**2", "V = (4/3)*pi*r**3"], '
        '"state_var": "V", "parameters": {"k": 0.08}, '
        f'"method": "heun", "t0": 0, "y0": {V0}, "t_end": 10, "h": 0.25, '
        '"analytical_solution": null, "rn": [0]}\n```'
    )
    tipo, problem = parse_exercise("problema de la gota", router=router)
    assert tipo == "pvi_first_order"
    assert "r" not in [str(s) for s in __import__("sympy").sympify(problem.ode_rhs).free_symbols]
    assert "A" not in [str(s) for s in __import__("sympy").sympify(problem.ode_rhs).free_symbols]

    resultado = run_pvi_first_order(problem)
    r_final = (3 * resultado.y[-1] / (4 * _math.pi)) ** (1/3)
    assert abs(r_final - 1.7) < 0.01  # dr/dt=-k exacto: r_final=2.5-0.08*10=1.7


def test_parse_exercise_relaciones_auxiliares_insuficientes_da_needs_clarification():
    # Falta la relacion V=(4/3)*pi*r**3: no se puede eliminar "r".
    router = _RouterFalso(
        '{"problem_type": "pvi_first_order", "ode_rhs": null, '
        '"rate_expr": "-k*A", '
        '"auxiliary_relations": ["A = 4*pi*r**2"], '
        '"state_var": "V", "parameters": {"k": 0.08}, '
        '"method": "heun", "t0": 0, "y0": 10, "t_end": 10, "h": 0.25, '
        '"analytical_solution": null, "rn": [0]}'
    )
    tipo, obj = parse_exercise("problema incompleto", router=router)
    assert tipo == "needs_clarification"
    assert "variables auxiliares" in obj.reason


def test_parse_exercise_sin_ode_rhs_ni_relaciones_da_needs_clarification():
    router = _RouterFalso(
        '{"problem_type": "pvi_first_order", "ode_rhs": null, '
        '"method": "heun", "t0": 0, "y0": 10, "t_end": 10, "h": 0.25, '
        '"analytical_solution": null, "rn": [0]}'
    )
    tipo, obj = parse_exercise("problema sin nada util", router=router)
    assert tipo == "needs_clarification"
