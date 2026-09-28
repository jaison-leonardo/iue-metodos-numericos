"""
assistant/parser.py
=====================
Convierte un enunciado en lenguaje natural a un objeto estructurado
(RootFindingProblem / PVIProblem / NeedsClarification), usando el
proveedor LLM configurado (Fase 8.1/8.2 de la especificación: el LLM
interpreta y clasifica, nunca calcula).

El JSON que devuelve el LLM se parsea con `json.loads` (nunca con
`eval`), y luego SIEMPRE pasa por validator.py antes de poder
ejecutarse (Fase 22: "No se debe ejecutar una interpretación
matemática dudosa").

Si no hay ningún proveedor LLM disponible (modo sin LLM), este módulo
no intenta interpretar texto libre: exige que el problema estructurado
se construya directamente (a mano, o desde otra fuente) y lo dice
explícitamente, en vez de fallar en silencio o inventar valores.
"""

from __future__ import annotations

import json
import os

from .llm.router import LLMRouter
from .llm.base import LLMProviderError
from .schema import RootFindingProblem, PVIProblem, NeedsClarification
from .ode_derivation import derive_ode_rhs, OdeDerivationError

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _extract_json(text: str) -> dict:
    """
    Extrae el bloque JSON que el LLM devolvió, dentro de una respuesta
    que puede traer texto adicional alrededor (razonamiento, o fórmulas
    en LaTeX que TAMBIÉN usan llaves, ej. e^{-c*t/m} — muy común en
    enunciados de caída libre/paracaidismo). Un simple
    text.find("{")/text.rfind("}") puede confundir esas llaves de LaTeX
    con el JSON real (toma la primera/última llave de TODO el texto, no
    necesariamente las del bloque JSON), o fallar por completo si el
    modelo no envolvió el JSON como se esperaba.

    Estrategia: se buscan todos los bloques de llaves balanceadas de
    nivel superior en el texto (en el orden en que aparecen — esto ya
    separa correctamente un "{...}" suelto de LaTeX del bloque JSON,
    porque cada uno se cierra en su propio nivel), se intenta parsear
    cada uno como JSON, y se toma el primero que (a) parsea
    correctamente Y (b) tiene una de las claves que esperamos
    ("problem_type" o "status"). Si ninguno cumple, se levanta un
    ValueError con un fragmento de la respuesta cruda del LLM, para que
    quien llama (parse_exercise) pueda mostrárselo al usuario en vez de
    fallar con una excepción sin manejar.
    """
    texto = text.strip()

    candidatos = []
    depth = 0
    inicio = None
    for idx, ch in enumerate(texto):
        if ch == "{":
            if depth == 0:
                inicio = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and inicio is not None:
                    candidatos.append(texto[inicio:idx + 1])
                    inicio = None

    for candidato in candidatos:
        try:
            data = json.loads(candidato)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and ("problem_type" in data or "status" in data):
            return data

    preview = texto[:400] + ("..." if len(texto) > 400 else "")
    raise ValueError(
        "la respuesta no contiene un JSON reconocible con 'problem_type' o "
        f"'status'. Respuesta cruda recibida: {preview!r}"
    )


def parse_exercise(exercise_text: str, router: LLMRouter = None):
    """
    Retorna una tupla (tipo, objeto):
      ("root_finding", RootFindingProblem)
      ("pvi_first_order", PVIProblem)
      ("needs_clarification", NeedsClarification)
      ("no_llm", None)   -> no hay proveedor LLM disponible; el llamador
                            debe construir el problema estructurado a mano.
    """
    router = router or LLMRouter()

    plantilla = _load_prompt("parse_problem.txt")
    prompt = plantilla.replace("{exercise_text}", exercise_text)

    try:
        # 1024 se quedaba corto con modelos "thinking" gratuitos, que
        # gastan parte del presupuesto en razonamiento interno antes de
        # escribir el JSON final (ver gemini_provider.py / http_chat_base.py
        # para el manejo de ese caso cuando aun asi se agota el presupuesto).
        respuesta = router.complete(prompt, max_tokens=4096)
    except LLMProviderError:
        return "no_llm", None

    try:
        data = _extract_json(respuesta["text"])
    except ValueError as e:
        # El proveedor SI respondio, pero su respuesta no se pudo
        # interpretar como el JSON esperado (ej. antepuso razonamiento,
        # se quedo sin tokens a mitad de camino, o el modelo gratuito
        # simplemente no siguio el formato pedido). Se reporta con
        # needs_clarification (nunca se adivina, y nunca se deja
        # propagar una excepcion sin manejar hasta el usuario).
        proveedor = respuesta.get("provider", "desconocido")
        return "needs_clarification", NeedsClarification(
            reason=f"El proveedor {proveedor} respondio, pero {e}",
            missing_information=[
                "Intenta de nuevo (a veces el modelo gratuito responde distinto), "
                "o ingresa el problema manualmente (opcion 2 del menu).",
            ],
        )

    if data.get("status") == "needs_clarification":
        return "needs_clarification", NeedsClarification(
            reason=data.get("reason", ""),
            missing_information=data.get("missing_information", []),
        )

    tipo = data.get("problem_type")
    if tipo == "root_finding":
        problem = RootFindingProblem(
            fx=data["fx"],
            method=data.get("method") or "biseccion",
            a0=data.get("a0"), b0=data.get("b0"),
            x0=data.get("x0"), x1=data.get("x1"),
            gx=data.get("gx"),
            tol=data.get("tol") or 1e-6,
            maxit=data.get("maxit") or 50,
            rn=data.get("rn") or [0],
            exercise_text=exercise_text,
        )
        return "root_finding", problem

    if tipo == "pvi_first_order":
        ode_rhs = data.get("ode_rhs")
        if not ode_rhs and data.get("auxiliary_relations"):
            # El enunciado dio la tasa de cambio en terminos de
            # variables auxiliares (ej. area/volumen/radio) en vez de
            # una expresion directa en t,y. El LLM NO debe resolver esa
            # eliminacion algebraica el mismo (fuente de errores y de
            # respuestas truncadas al gastar tokens "pensando"): se
            # hace aqui, de forma determinista, con sympy.
            try:
                ode_rhs = derive_ode_rhs(
                    rate_expr=data.get("rate_expr"),
                    auxiliary_relations=data.get("auxiliary_relations"),
                    state_var=data.get("state_var"),
                    parameters=data.get("parameters") or {},
                )
            except OdeDerivationError as e:
                return "needs_clarification", NeedsClarification(
                    reason=(
                        "El enunciado da la tasa de cambio en terminos de "
                        "variables auxiliares, pero no se pudieron eliminar "
                        f"de forma automatica: {e}"
                    ),
                    missing_information=[
                        "Revisa que 'rate_expr' y 'auxiliary_relations' den "
                        "toda la informacion necesaria, o ingresa "
                        "ode_rhs directamente (opcion 2 del menu).",
                    ],
                )

        if not ode_rhs:
            return "needs_clarification", NeedsClarification(
                reason=(
                    "El LLM no devolvio ni 'ode_rhs' ni relaciones "
                    "auxiliares suficientes para derivarlo."
                ),
                missing_information=["ode_rhs"],
            )

        problem = PVIProblem(
            ode_rhs=ode_rhs,
            method=data.get("method") or "euler",
            t0=data["t0"], y0=data["y0"], t_end=data["t_end"], h=data["h"],
            analytical_solution=data.get("analytical_solution"),
            evento_y=data.get("evento_y"),
            parameters=data.get("parameters") or {},
            rn=data.get("rn") or [0],
            exercise_text=exercise_text,
        )
        return "pvi_first_order", problem

    return "needs_clarification", NeedsClarification(
        reason="El LLM no devolvió un problem_type reconocido.",
        missing_information=["problem_type"],
    )
