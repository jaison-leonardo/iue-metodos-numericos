"""
Diagnostico de conectividad LLM real.

Este script SI necesita ejecutarse en una maquina con salida a
internet real hacia Gemini/Groq/OpenRouter (tu computador, no un
entorno de nube en sandbox). Prueba, en orden, cada proveedor/cuenta
configurado en .env y reporta cual respondio.

Ejecutar desde la raiz del repositorio:
    python assistant/examples/test_llm_connection.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from assistant.llm.router import LLMRouter


def main():
    router = LLMRouter()

    print("Cadena de proveedores configurada (.env):")
    for d in router.describe_chain():
        estado = "OK (key + paquete presentes)" if d["available"] else "no disponible"
        print(f"  - {d['label']:40s} {estado}")
    print()

    prompt = (
        "Responde EXCLUSIVAMENTE con este JSON, sin texto adicional: "
        '{"status": "ok", "quien_respondio": "<tu nombre de modelo>"}'
    )

    # max_tokens generoso: algunos modelos gratuitos auto-enrutados (p.ej. via
    # OpenRouter) son modelos de "razonamiento" que gastan varios tokens
    # pensando antes de escribir la respuesta final; con un limite muy bajo
    # (100) el contenido final puede llegar vacio.
    try:
        resultado = router.complete(prompt, max_tokens=800)
        print("EXITO")
        print("Proveedor que respondio:", resultado["provider"])
        print("Respuesta cruda:")
        print(resultado["text"])
    except Exception as e:
        print("FALLO: ningun proveedor respondio.")
        print(e)


if __name__ == "__main__":
    main()
