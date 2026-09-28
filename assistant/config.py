"""
assistant/config.py
=====================
Configuración del proveedor LLM (Fase 6 de la especificación). Lee
exclusivamente de variables de entorno o de un archivo local `.env`
NO versionado; nunca hay una API key escrita en el código fuente.

Soporta VARIAS CUENTAS para un mismo proveedor (por ejemplo, dos
cuentas distintas de Gemini) mediante el formato:

    AMN_PRIMARY_PROVIDER=<proveedor>:<NOMBRE_VARIABLE_DE_ENTORNO>
    AMN_FALLBACK_PROVIDERS=<proveedor>:<VAR>,<proveedor>:<VAR>,...

Ejemplo (el que usa este proyecto en `.env`):
    AMN_PRIMARY_PROVIDER=gemini:API_KEY_GEMINI_JAIME
    AMN_FALLBACK_PROVIDERS=openrouter:OPENROUTER_API_KEY_JAISON,groq:GROQ_API_KEY_JAISON,gemini:API_KEY_GEMINI_JAISON

Cada token "<proveedor>:<VAR>" se resuelve así: `<proveedor>` decide
qué adaptador de assistant/llm/ se instancia, y el VALOR de la
variable de entorno `<VAR>` es la API key que se le pasa. Esto permite
tener, por ejemplo, "gemini:API_KEY_GEMINI_JAIME" y
"gemini:API_KEY_GEMINI_JAISON" como dos eslabones distintos de la
cadena, cada uno con su propia cuenta.

Proveedores soportados: gemini, groq, openrouter, deepseek, none.
Claude y OpenAI (las APIs pagas de Anthropic/OpenAI propiamente
dichas) se descartaron deliberadamente de este proyecto por costo;
todos los proveedores soportados hoy son gratuitos o de muy bajo
costo, y todos hablan el mismo protocolo compatible con OpenAI (ver
assistant/llm/openai_compatible_base.py), por lo que basta una sola
dependencia externa (`openai`) para los cuatro.

El sistema SIEMPRE puede funcionar sin ningún proveedor disponible:
la cadena termina implícitamente en "none" (modo determinista).
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOTENV_PATH = os.path.join(REPO_ROOT, ".env")


def _load_dotenv_once():
    if not os.path.isfile(_DOTENV_PATH):
        return
    with open(_DOTENV_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Las variables de entorno YA presentes tienen prioridad:
            # no se sobreescriben si ya existen.
            os.environ.setdefault(key, value)


_load_dotenv_once()


def _parse_chain_token(token: str):
    """'gemini:API_KEY_GEMINI_JAIME' -> ('gemini', 'API_KEY_GEMINI_JAIME', <valor o None>)
    'none' (sin ':')                -> ('none', None, None)
    """
    token = token.strip()
    if not token:
        return None
    if ":" in token:
        provider_name, env_var_name = token.split(":", 1)
        provider_name = provider_name.strip().lower()
        env_var_name = env_var_name.strip()
        return provider_name, env_var_name, os.environ.get(env_var_name)
    return token.lower(), None, None


class AssistantConfig:
    def __init__(self):
        primary = os.environ.get("AMN_PRIMARY_PROVIDER", "").strip()
        fallbacks = os.environ.get("AMN_FALLBACK_PROVIDERS", "").strip()
        raw_chain = ",".join(part for part in (primary, fallbacks) if part)
        self.chain_tokens = [t for t in raw_chain.split(",") if t.strip()]

    def provider_chain(self):
        """
        Retorna una lista de tuplas (provider_name, env_var_name, api_key),
        en el orden configurado, siempre terminando en ("none", None, None)
        si "none" no aparece ya explícitamente en la cadena.
        """
        chain = []
        for token in self.chain_tokens:
            parsed = _parse_chain_token(token)
            if parsed:
                chain.append(parsed)
        if not any(p == "none" for p, _, _ in chain):
            chain.append(("none", None, None))
        return chain


def get_config() -> AssistantConfig:
    return AssistantConfig()
