"""
assistant/llm/base.py
=======================
Interfaz común que deben implementar todos los proveedores LLM.
`executor.py`/`parser.py` nunca hablan directamente con un SDK
concreto: siempre pasan por esta interfaz, para que cambiar de
proveedor sea un cambio de configuración, no de código
(sección 23 de la especificación: "El cliente debe estar desacoplado
de la lógica matemática").
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(RuntimeError):
    """Error al invocar un proveedor (key ausente, timeout, rate limit, etc.)."""


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """True si el proveedor tiene lo necesario para intentar una llamada
        (p.ej. una API key configurada). No garantiza que la llamada vaya
        a tener éxito (la red puede fallar igual)."""

    @abstractmethod
    def complete(self, prompt: str, *, system: str = None, max_tokens: int = 1024) -> str:
        """Envía `prompt` (y opcionalmente `system`) al proveedor y
        retorna el texto de respuesta. Debe lanzar LLMProviderError en
        cualquier fallo (nunca debe devolver silenciosamente una
        respuesta vacía o inventada)."""
