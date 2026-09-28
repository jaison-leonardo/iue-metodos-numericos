"""
Pruebas de regresión para el PVI de primer orden: euler_meth, heun_meth,
ptomed_meth y rk4_meth deben seguir produciendo exactamente los mismos
arreglos t,y que quedaron congelados en tests/golden/golden_outputs.json.

Estos métodos no se modificaron (no lo necesitan: ya retornan el
arreglo completo, con t[0]=t0 como "iteración 0" / condición inicial),
así que esta prueba es puramente preventiva ante cambios futuros.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "PVI_1Ord"))

import numpy as np
import met_solpvi_1ord as pvim

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "golden_outputs.json")

with open(GOLDEN_PATH) as f:
    GOLDEN = json.load(f)["pvi1"]


def _check(nombre, t, y):
    esperado = GOLDEN[nombre]
    assert np.allclose(t, esperado["t"]), f"{nombre}: t cambio"
    assert np.allclose(y, esperado["y"]), f"{nombre}: y cambio"
    assert t[0] == GOLDEN["params"]["T0"], f"{nombre}: t[0] ya no es la condicion inicial (n=0)"
    assert y[0] == GOLDEN["params"]["Y0"], f"{nombre}: y[0] ya no es la condicion inicial (n=0)"


def test_euler():
    t, y = pvim.euler_meth()
    _check("euler", t, y)


def test_heun():
    t, y = pvim.heun_meth()
    _check("heun", t, y)


def test_ptomed():
    t, y = pvim.ptomed_meth()
    _check("ptomed", t, y)


def test_rk4():
    t, y = pvim.rk4_meth()
    _check("rk4", t, y)


# ---------------------------------------------------------------------------
# Parada por evento (sesion 5, recomendacion 5): evento_y / evento_info son
# parametros OPCIONALES (default None) en los 4 metodos; las pruebas de
# arriba (sin pasar evento_y) ya demuestran que el comportamiento por
# defecto queda identico. Estas pruebas cubren el caso NUEVO.
# ---------------------------------------------------------------------------

def _con_evento(func, evento_y):
    """
    y' = -y, y0 = 100 (decaimiento exponencial puro: y(t) = 100*exp(-t)).
    n=1000, h=0.1 son solo una cota superior generosa; el evento debe
    disparar mucho antes.
    """
    import params_pvi1 as p
    original_funced = p.FUNCED
    p.FUNCED = "-y"
    try:
        info = {}
        t, y = func(t0=0, y0=100, n=1000, h=0.1, evento_y=evento_y, evento_info=info)
        return t, y, info
    finally:
        p.FUNCED = original_funced


def test_euler_evento_detiene_antes_de_n_completo():
    t, y, info = _con_evento(pvim.euler_meth, evento_y=10)
    assert info["alcanzado"] is True
    assert len(t) < 1001  # se detuvo antes de agotar los 1000 pasos
    assert len(t) == len(y)
    # y(t) = 100*exp(-t) = 10  =>  t = ln(10) ~= 2.302585
    assert abs(info["t_evento"] - np.log(10)) < 0.15
    # el ultimo punto calculado debe estar cerca del valor objetivo
    assert abs(y[-1] - 10) < 1.0


def test_rk4_evento_mas_preciso_que_euler():
    _, _, info_euler = _con_evento(pvim.euler_meth, evento_y=10)
    _, _, info_rk4 = _con_evento(pvim.rk4_meth, evento_y=10)
    exacto = np.log(10)
    # RK4 (4to orden) debe aproximar t_evento mejor que Euler (1er orden).
    assert abs(info_rk4["t_evento"] - exacto) < abs(info_euler["t_evento"] - exacto)


def test_heun_y_ptomed_tambien_soportan_evento():
    for func in (pvim.heun_meth, pvim.ptomed_meth):
        t, y, info = _con_evento(func, evento_y=10)
        assert info["alcanzado"] is True
        assert len(t) == len(y)


def test_evento_no_alcanzado_si_n_es_insuficiente():
    """Si evento_y esta fuera de rango en los n pasos dados, se agota el
    bucle normalmente (igual que sin evento), y evento_info lo refleja."""
    import params_pvi1 as p
    original_funced = p.FUNCED
    p.FUNCED = "-y"
    try:
        info = {}
        t, y = pvim.euler_meth(t0=0, y0=100, n=5, h=0.1, evento_y=1e-9, evento_info=info)
        assert info["alcanzado"] is False
        assert info["t_evento"] is None
        assert len(t) == 6  # n+1, igual que sin evento
    finally:
        p.FUNCED = original_funced


def test_evento_none_es_identico_a_no_pasar_el_parametro():
    """evento_y=None (el default) debe dar EXACTAMENTE el mismo resultado
    que no pasar evento_y/evento_info en absoluto (verificado tambien por
    las pruebas de golden de arriba, que no pasan estos parametros)."""
    t1, y1 = pvim.rk4_meth()
    t2, y2 = pvim.rk4_meth(evento_y=None, evento_info=None)
    assert np.array_equal(t1, t2)
    assert np.array_equal(y1, y2)


def test_evento_con_sqrt_cerca_de_cero_no_produce_nan():
    """
    Caso real: vaciado de tanque tipo Torricelli, dy/dt = -k*sqrt(y).
    Con RK4/Euler "ingenuos", el ultimo paso completo puede sobrepasar
    y=0 y evaluar sqrt(negativo) -> NaN. Se espera que el fallback de
    extrapolacion lineal (aproximado=True) evite el NaN y de un
    resultado final finito, con y[-1] == evento_y exactamente.
    """
    import params_pvi1 as p
    original_funced = p.FUNCED
    p.FUNCED = "-0.35*sqrt(y)"
    try:
        info = {}
        t, y = pvim.rk4_meth(t0=0, y0=4, n=200, h=0.1, evento_y=0, evento_info=info)
        assert info["alcanzado"] is True
        assert info["aproximado"] is True
        assert np.isfinite(info["t_evento"])
        assert not np.isnan(y).any()
        assert y[-1] == 0
        # Solucion analitica de Torricelli: y(t)=(sqrt(y0)-k*t/2)^2 hasta
        # que se anula; t_vaciado = 2*sqrt(y0)/k = 2*2/0.35 ~= 11.4286.
        assert abs(info["t_evento"] - (2*2/0.35)) < 0.2
    finally:
        p.FUNCED = original_funced
