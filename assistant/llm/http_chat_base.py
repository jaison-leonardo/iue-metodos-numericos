"""
assistant/llm/http_chat_base.py
==================================
Clase base compartida para proveedores que exponen el formato de
"chat completions" estilo OpenAI (Groq, OpenRouter, DeepSeek lo hacen)
usando HTTP DIRECTO (`requests`), sin el SDK `openai`.

Se cambio de SDK a HTTP directo tras la prueba real de Jaison: el SDK
`openai` (via su dependencia `httpx`) fallo en su Python global
(C:\\Python312, compartido con otros proyectos como el de EEG) con
"Client.__init__() got an unexpected keyword argument 'proxies'" —un
choque de versiones openai<->httpx conocido. `requests` es una
libreria mucho mas estable/antigua, con una API que practicamente no
cambia entre versiones, y evita ese problema de raiz. Tambien reduce
el numero de dependencias nuevas del proyecto.
"""

from __future__ import annotations

from .base import LLMProvider, LLMProviderError


class HTTPChatCompletionsProvider(LLMProvider):
    base_url: str = None          # p.ej. "https://api.groq.com/openai/v1"
    default_model: str = None

    def __init__(self, api_key: str | None, model: str | None = None, extra_headers: dict | None = None):
        self.api_key = api_key
        self.model = model or self.default_model
        self.extra_headers = extra_headers or {}

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
            raise LLMProviderError(f"{self.name}: falta la API key configurada.")
        try:
            import requests
        except ImportError as e:
            raise LLMProviderError(
                "El paquete 'requests' no está instalado."
            ) from e

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": messages, "max_tokens": max_tokens},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise LLMProviderError(f"Error llamando a {self.name} (modelo {self.model}): {e}") from e

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as e:
            raise LLMProviderError(
                f"{self.name} (modelo {self.model}): respuesta con formato inesperado: {data}"
            ) from e

        content = message.get("content")
        if content:
            return content

        # Algunos modelos "gratis" auto-enrutados (p.ej. via OpenRouter) son
        # modelos de razonamiento: si max_tokens se agota mientras el modelo
        # todavia esta "pensando", 'content' llega vacio/None y el texto util
        # (si lo hay) queda en 'reasoning' o 'reasoning_content' en su lugar.
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            return reasoning

        finish_reason = choice.get("finish_reason")
        raise LLMProviderError(
            f"{self.name} (modelo {self.model}) devolvió una respuesta vacía "
            f"(finish_reason={finish_reason!r}). El modelo gratuito auto-enrutado "
            "puede haber agotado max_tokens en razonamiento interno antes de "
            "escribir el contenido final; sube max_tokens o intenta con otro "
            "proveedor de la cadena."
        )
