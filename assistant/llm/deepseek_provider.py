"""
assistant/llm/deepseek_provider.py
=====================================
Adaptador para DeepSeek (https://api.deepseek.com), via HTTP directo,
tambien compatible con el formato de chat completions. Se deja
implementado (Fase 6 de la especificacion) aunque hoy no haya una API
key de DeepSeek configurada; no agrega dependencias nuevas.
"""

from __future__ import annotations

import os

from .http_chat_base import HTTPChatCompletionsProvider


class DeepSeekProvider(HTTPChatCompletionsProvider):
    name = "deepseek"
    base_url = "https://api.deepseek.com"
    default_model = os.environ.get("AMN_MODEL_DEEPSEEK", "deepseek-chat")
