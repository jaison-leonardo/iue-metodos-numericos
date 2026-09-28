"""
Script para (re)generar tests/golden/golden_outputs.json.

Congela los resultados numéricos actuales de:
- Búsqueda de raíces (biseccion, falsapos, newton, secante)
- PVI de primer orden (euler, heun, ptomed, rk4)

usando los parámetros que estén definidos en params.py / params_pvi1.py
en el momento de ejecutarlo.

USO:
    Solo se debe re-ejecutar este script de forma DELIBERADA, cuando se
    decida (y se documente por qué) que el comportamiento numérico
    actual debe convertirse en el nuevo "golden output" de referencia.
    NO ejecutarlo automáticamente como parte de los tests: los tests
    (test_regresion_raices.py, test_regresion_pvi1.py) están hechos
    para comparar contra el archivo ya congelado, no para regenerarlo.

Ejecutar desde la raíz del repositorio:
    python tests/golden/generate_golden.py
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "PVI_1Ord"))

import Met_busqraices as metb
import params as p
import met_solpvi_1ord as pvim
import params_pvi1 as p1


def build_golden():
    results = {}

    solb, nitb, erb, iter_rb = metb.biseccion()
    solfp, nitfp, erfp, iter_rfp = metb.falsapos()
    soln, nitn, ern, iter_rn = metb.newton()
    solsec, nitsec, ersec, iter_rsec = metb.secante()

    # puntofijo necesita una g(x) elegida a mano (no se deriva de FX
    # automaticamente). Para f(x)=x**3+3*x-32=0, x=(32-3x)^(1/3) es una
    # g(x) que converge cerca de la raiz (|g'(x)| < 1 ahi); se fija aqui
    # SOLO para congelar un golden output estable, no cambia el default
    # de params.GX que usan los demas escenarios/tests.
    gx_convergente = "(32 - 3*x)**(1/3)"
    p.GX = gx_convergente
    solpf, nitpf, erpf, iter_rpf = metb.puntofijo(x0=3.0)

    results["raices"] = {
        "params": {
            "FX": p.FX, "A0": p.A0, "B0": p.B0, "R0": p.R0,
            "TOL": p.TOL, "MAXIT": p.MAXIT, "RN": p.RN,
        },
        "biseccion": {"r": solb, "iterac": nitb, "error": erb, "iterac_list": iter_rb},
        "falsapos": {"r": solfp, "iterac": nitfp, "error": erfp, "iterac_list": iter_rfp},
        "newton": {"r": soln, "iterac": nitn, "error": ern, "iterac_list": iter_rn},
        "secante": {"r": solsec, "iterac": nitsec, "error": ersec, "iterac_list": iter_rsec},
        "puntofijo": {
            "gx": gx_convergente, "x0": 3.0,
            "r": solpf, "iterac": nitpf, "error": erpf, "iterac_list": iter_rpf,
        },
    }

    teu, yeu = pvim.euler_meth()
    the, yhe = pvim.heun_meth()
    tptom, yptom = pvim.ptomed_meth()
    trk4, yrk4 = pvim.rk4_meth()

    results["pvi1"] = {
        "params": {
            "FUNCED": p1.FUNCED, "FUNCSOLEX": p1.FUNCSOLEX,
            "T0": p1.T0, "Y0": p1.Y0, "T_END": p1.T_END,
            "HP": p1.HP, "NIT": p1.NIT,
        },
        "euler": {"t": teu.tolist(), "y": yeu.tolist()},
        "heun": {"t": the.tolist(), "y": yhe.tolist()},
        "ptomed": {"t": tptom.tolist(), "y": yptom.tolist()},
        "rk4": {"t": trk4.tolist(), "y": yrk4.tolist()},
    }
    return results


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_outputs.json")
    data = build_golden()
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Golden outputs escritos en: {out_path}")
