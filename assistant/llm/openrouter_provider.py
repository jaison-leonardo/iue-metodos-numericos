"""
assistant/llm/openrouter_provider.py
=======================================
Adaptador para OpenRouter (https://openrouter.ai/api/v1), via HTTP
directo. Proveedor de RESPALDO.

Modelo por defecto: "openrouter/free". Jaison lo probo en vivo con
`curl` y funciono: OpenRouter enruta automaticamente ese alias hacia
CUALQUIER modelo gratuito disponible en ese momento (en la prueba,
resolvio a "inclusionai/ling-3.0-flash-fin:free"). Se prefirio este
alias sobre fijar un modelo ":free" especifico porque el catalogo de
modelos gratuitos de OpenRouter cambia con frecuencia (se agregan y
retiran); "openrouter/free" evita que el proyecto deje de funcionar
cuando eso pase. Si se prefiere fijar un modelo puntual, cambiar
AMN_MODEL_OPENROUTER (ver catalogo en https://openrouter.ai/models).
"""

from __future__ import annotations

import os

from .http_chat_base import HTTPChatCompletionsProvider


class OpenRouterProvider(HTTPChatCompletionsProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    default_model = os.environ.get("AMN_MODEL_OPENROUTER", "openrouter/free")

    def __init__(self, api_key, model=None):
        super().__init__(
            api_key, model,
            extra_headers={
                "HTTP-Referer": "https://github.com/iue-metodos-numericos",
                "X-Title": "AMN - Asistente de Metodos Numericos",
            },
        )
