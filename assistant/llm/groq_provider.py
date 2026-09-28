"""
assistant/llm/groq_provider.py
=================================
Adaptador para Groq, via HTTP directo (chat/completions compatible con
OpenAI) contra https://api.groq.com/openai/v1. Proveedor de RESPALDO.

Modelo por defecto configurable con AMN_MODEL_GROQ. "openai/gpt-oss-120b"
es un modelo de pesos abiertos servido por Groq con buen desempeño en
seguimiento de instrucciones/extraccion estructurada; si se prioriza
velocidad/costo, cambiar a "openai/gpt-oss-20b" via esa variable.
"""

from __future__ import annotations

import os

from .http_chat_base import HTTPChatCompletionsProvider


class GroqProvider(HTTPChatCompletionsProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
    default_model = os.environ.get("AMN_MODEL_GROQ", "openai/gpt-oss-120b")
