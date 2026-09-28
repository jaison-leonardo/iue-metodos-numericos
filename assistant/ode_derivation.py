"""
assistant/ode_derivation.py
=============================
Elimina algebraicamente variables auxiliares (ej. área, volumen, radio
relacionados entre sí) para reducir la tasa de cambio dada por un
enunciado a una única ecuación diferencial ode_rhs(t, y) en términos
de la variable de estado.

Esta eliminación es PURAMENTE SIMBÓLICA (sympy.solve + sustitución),
determinista y verificable. NUNCA la hace el LLM: el LLM solo
transcribe las relaciones tal cual el enunciado las da (misma filosofía
ya establecida en parser.py: "el LLM interpreta y clasifica, nunca
calcula"). Antes de este módulo, se le pedía al LLM que derivara él
mismo la sustitución algebraica (ej. "V = (4/3)*pi*r**3" -> despejar r
-> sustituir en A=4*pi*r**2 -> sustituir en dV/dt=-k*A), lo cual lo
hacía gastar tokens de "razonamiento" en álgebra (con el riesgo real de
quedarse sin tokens a mitad de camino, o de cometer un error algebraico
silencioso) en vez de simplemente transcribir lo que el enunciado dice.
"""

from __future__ import annotations

import re

import sympy as sp

from .analytical import ALLOWED_FUNCTION_NAMES, ALLOWED_CONSTANT_NAMES, _collect_function_names


class OdeDerivationError(ValueError):
    """No fue posible eliminar las variables auxiliares de forma única."""


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _nombres_en(texto: str) -> set:
    """Identificadores crudos en un string (regex, antes de sympificar).
    Se usa solo para DESCUBRIR qué símbolos hay que declarar antes de
    sympificar de verdad con las asunciones correctas (positive=True)."""
    return set(_IDENTIFIER_RE.findall(texto))


def derive_ode_rhs(rate_expr: str, auxiliary_relations, state_var: str, parameters: dict = None) -> str:
    """
    Reduce `rate_expr` (la tasa de cambio, ej. "-k*A") a una expresión
    únicamente en términos de `state_var` (ej. "V") y "t", usando
    `auxiliary_relations` (cada una en forma "lhs = rhs", ej.
    ["A = 4*pi*r**2", "V = (4/3)*pi*r**3"]) para eliminar cualquier
    variable intermedia (ej. "r", "A").

    `parameters` son valores numéricos ya conocidos (ej. {"k": 0.08}),
    sustituidos antes de resolver.

    Devuelve el string de ode_rhs listo para PVIProblem.ode_rhs.

    Lanza OdeDerivationError si no se puede resolver de forma única
    (relaciones insuficientes/inconsistentes, más de una solución
    distinta, o quedan variables sin eliminar).
    """
    parameters = parameters or {}
    if not auxiliary_relations:
        raise OdeDerivationError("No se dieron relaciones auxiliares para eliminar variables.")

    nombres_crudos = set()
    for texto in list(auxiliary_relations) + [rate_expr]:
        nombres_crudos |= _nombres_en(texto)
    nombres_crudos -= ALLOWED_FUNCTION_NAMES
    nombres_crudos -= ALLOWED_CONSTANT_NAMES
    nombres_crudos -= set(parameters.keys())
    nombres_crudos.discard("t")

    # Todas las variables físicas (todo excepto t) se asumen positivas:
    # una simplificación razonable para los ejercicios típicos de este
    # curso (radios, áreas, volúmenes, concentraciones... siempre > 0),
    # que además es justo lo que permite a sympy.solve descartar ramas
    # complejas/negativas espurias (ej. al despejar un radio de un
    # volumen mediante una raíz cúbica).
    t_symbol = sp.symbols("t", real=True)
    locals_dict = {"t": t_symbol}
    for nombre in nombres_crudos:
        locals_dict[nombre] = sp.symbols(nombre, positive=True)
    for nombre, valor in parameters.items():
        locals_dict[nombre] = valor

    def _sympify_seguro(texto, contexto):
        try:
            expr = sp.sympify(texto, locals=locals_dict, convert_xor=True, evaluate=True)
        except (sp.SympifyError, TypeError, SyntaxError) as e:
            raise OdeDerivationError(f"No se pudo interpretar {contexto} '{texto}': {e}") from e
        no_permitidas = _collect_function_names(expr) - ALLOWED_FUNCTION_NAMES
        if no_permitidas:
            raise OdeDerivationError(
                f"{contexto} '{texto}' usa funciones no permitidas: {sorted(no_permitidas)}"
            )
        return expr

    rate = _sympify_seguro(rate_expr, "la tasa de cambio")

    ecuaciones = []
    for rel in auxiliary_relations:
        if "=" not in rel:
            raise OdeDerivationError(f"La relación auxiliar '{rel}' no tiene la forma 'lhs = rhs'.")
        lhs_str, rhs_str = rel.split("=", 1)
        lhs = _sympify_seguro(lhs_str.strip(), "la relación auxiliar (lado izquierdo)")
        rhs = _sympify_seguro(rhs_str.strip(), "la relación auxiliar (lado derecho)")
        ecuaciones.append(sp.Eq(lhs, rhs))

    y_symbol = locals_dict.get(state_var)
    if y_symbol is None:
        raise OdeDerivationError(
            f"La variable de estado '{state_var}' no aparece en ninguna relación auxiliar."
        )

    incognitas = sorted(
        (
            s for s in rate.free_symbols.union(*(eq.free_symbols for eq in ecuaciones))
            if s != y_symbol and s != t_symbol
        ),
        key=str,
    )

    y_generico = sp.symbols("y", real=True)

    if not incognitas:
        # rate_expr ya estaba en terminos de state_var/t solamente.
        # Se renombra state_var -> "y" (convencion del resto del
        # pipeline: PVIProblem.ode_rhs siempre se valida/ejecuta en
        # terminos de "t","y", sin importar como se llame la variable
        # de estado en el enunciado original, ej. "V").
        return str(sp.simplify(rate).subs(y_symbol, y_generico))

    try:
        soluciones = sp.solve(ecuaciones, incognitas, dict=True)
    except NotImplementedError as e:
        raise OdeDerivationError(
            f"Sympy no pudo resolver el sistema de relaciones auxiliares: {e}"
        ) from e

    if not soluciones:
        raise OdeDerivationError(
            "El sistema de relaciones auxiliares no tiene solución (revisa que las "
            "relaciones sean consistentes entre sí)."
        )
    if len(soluciones) > 1:
        raise OdeDerivationError(
            f"El sistema de relaciones auxiliares tiene {len(soluciones)} soluciones "
            "distintas (ambiguo): revisa el enunciado o agrega más relaciones para "
            "descartar ramas espurias."
        )

    rate_final = sp.simplify(rate.subs(soluciones[0]))

    restantes = rate_final.free_symbols - {y_symbol, t_symbol}
    if restantes:
        raise OdeDerivationError(
            "No fue posible eliminar por completo las variables auxiliares "
            f"{sorted(str(s) for s in restantes)}; falta alguna relación."
        )

    # Renombrar state_var -> "y" (ver comentario arriba).
    return str(rate_final.subs(y_symbol, y_generico))
