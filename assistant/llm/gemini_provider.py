"""
assistant/llm/gemini_provider.py
===================================
Adaptador para Google Gemini usando su API REST NATIVA
(https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent),
via HTTP directo con `requests` (sin SDK).

Se eligio la API nativa (en vez de la capa de compatibilidad con
OpenAI) porque Jaison la probo en vivo con `curl` y funciono
correctamente con su API key real, devolviendo `modelVersion:
"gemini-3.8-flash"`.

Modelo por defecto: "gemini-flash-latest" (un alias que Google
mantiene apuntando siempre al ultimo modelo "flash" estable). Se
prefirio el alias "-latest" sobre fijar una version puntual
(p.ej. "gemini-2.0-flash") precisamente para no tener que actualizar
el codigo/.env cada vez que Google libera una version nueva. Si se
prefiere fijar una version exacta por reproducibilidad, se puede
cambiar via AMN_MODEL_GEMINI sin tocar codigo.
"""

from __future__ import annotations

import os

from .base import LLMProvider, LLMProviderError

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = os.environ.get("AMN_MODEL_GEMINI", "gemini-flash-latest")

    def __init__(self, api_key: str | None, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.default_model

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import requests  # noqa: F401
        except ImportError:
            return False
        return True

    def complete(self, prompt: str, *, system: str = None, max_tokens: int = 1024) -> str:
        if not self.api_key:
            raise LLMProviderError("gemini: falta la API key configurada.")
        try:
            import requests
        except ImportError as e:
            raise LLMProviderError("El paquete 'requests' no está instalado.") from e

        contenido = prompt if not system else f"{system}\n\n{prompt}"
        payload = {
            "contents": [{"parts": [{"text": contenido}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }

        try:
            resp = requests.post(
                f"{_BASE_URL}/{self.model}:generateContent",
                headers={"Content-Type": "application/json", "X-goog-api-key": self.api_key},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            candidato = data["candidates"][0]
            finish_reason = candidato.get("finishReason")
            partes = candidato.get("content", {}).get("parts", [])
            texto = "".join(p.get("text", "") for p in partes)

            # Los modelos "flash" recientes de Gemini son modelos "thinking"
            # por defecto: gastan una parte de maxOutputTokens en
            # razonamiento interno ANTES de escribir la respuesta final.
            # Si el presupuesto se agota a mitad de camino, la API corta
            # la respuesta (finishReason="MAX_TOKENS") y el texto que
            # llega puede quedar truncado (ej. un JSON sin cerrar). En vez
            # de devolver ese texto incompleto en silencio (que luego
            # falla de forma confusa en _extract_json), se levanta
            # LLMProviderError aqui mismo: esto hace que el router pase
            # automaticamente al siguiente proveedor de la cadena, igual
            # que ya se hace para Groq/OpenRouter/DeepSeek en
            # http_chat_base.py ante el mismo problema.
            if not texto or finish_reason == "MAX_TOKENS":
                raise LLMProviderError(
                    f"gemini (modelo {self.model}) no completo la respuesta "
                    f"(finish_reason={finish_reason!r}, {len(texto)} caracteres "
                    "recibidos antes del corte). Probablemente se quedo sin "
                    "tokens pensando internamente antes de terminar."
                )
            return texto
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Error llamando a gemini (modelo {self.model}): {e}") from e
