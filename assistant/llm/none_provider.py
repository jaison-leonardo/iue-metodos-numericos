"""
assistant/llm/none_provider.py
================================
"Modo sin LLM" (Fase 6): siempre disponible, nunca hace red. Se usa
cuando no hay ninguna API key configurada, o como último recurso de la
cadena de fallback. No interpreta lenguaje natural: exige que el
problema ya venga estructurado (schema.py), y lo dice explícitamente
si se le pide interpretar texto libre.
"""

from __future__ import annotations

from .base import LLMProvider, LLMProviderError


class NoneProvider(LLMProvider):
    name = "none"

    def is_available(self) -> bool:
        return True

    def complete(self, prompt: str, *, system: str = None, max_tokens: int = 1024) -> str:
        raise LLMProviderError(
            "Modo sin LLM activo: no se puede interpretar lenguaje natural. "
            "Proporcione el ejercicio ya estructurado (RootFindingProblem / PVIProblem)."
        )
