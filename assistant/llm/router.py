"""
assistant/llm/router.py
=========================
Orquesta la cadena proveedor-primario -> fallbacks -> "none", con
manejo de errores (Fase 6). Este es el único punto donde se decide
"qué proveedor usar ahora"; todo lo demás en el código habla con la
interfaz LLMProvider, nunca con un proveedor concreto.

Soporta varias cuentas del mismo proveedor (p.ej. dos cuentas de
Gemini): cada eslabón de la cadena lleva también la etiqueta de la
variable de entorno de la que salió su API key (p.ej.
"gemini (API_KEY_GEMINI_JAIME)"), para que los mensajes de error sean
comprensibles cuando hay más de una cuenta del mismo proveedor.
"""

from __future__ import annotations

from ..config import AssistantConfig, get_config
from .base import LLMProvider, LLMProviderError
from .none_provider import NoneProvider
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .openrouter_provider import OpenRouterProvider
from .deepseek_provider import DeepSeekProvider

_PROVIDER_CLASSES = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "deepseek": DeepSeekProvider,
}


def _build_provider(provider_name: str, env_var_name, api_key) -> LLMProvider:
    cls = _PROVIDER_CLASSES.get(provider_name)
    if cls is None:
        provider = NoneProvider()
    else:
        provider = cls(api_key)
    provider.label = f"{provider.name}" + (f" ({env_var_name})" if env_var_name else "")
    return provider


class LLMRouter:
    """Intenta cada proveedor/cuenta de la cadena en orden; si uno falla
    (sin key, SDK ausente, error de red, error de la API), pasa al
    siguiente. Nunca deja al sistema sin respuesta: "none" siempre es
    el último eslabón."""

    def __init__(self, config: AssistantConfig = None):
        self.config = config or get_config()
        self.chain = [
            _build_provider(name, env_var, key)
            for name, env_var, key in self.config.provider_chain()
        ]

    def complete(self, prompt: str, *, system: str = None, max_tokens: int = 1024):
        errores = []
        for provider in self.chain:
            if not provider.is_available():
                errores.append(f"{provider.label}: no disponible (sin key o SDK no instalado)")
                continue
            try:
                texto = provider.complete(prompt, system=system, max_tokens=max_tokens)
                return {"provider": provider.label, "text": texto}
            except LLMProviderError as e:
                errores.append(f"{provider.label}: {e}")
                continue
        raise LLMProviderError(
            "Ningún proveedor LLM pudo responder. Detalle: " + " | ".join(errores)
        )

    def active_provider_name(self) -> str:
        for provider in self.chain:
            if provider.is_available():
                return provider.label
        return "none"

    def describe_chain(self):
        """Lista legible de la cadena configurada, para diagnóstico."""
        return [
            {"label": p.label, "available": p.is_available()}
            for p in self.chain
        ]
