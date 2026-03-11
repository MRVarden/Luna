"""Local provider — Ollama, llama.cpp, vLLM, LM Studio via local server.

Strategy:
  1. If the ``openai`` package is installed, use AsyncOpenAI (fast path).
  2. Otherwise, fall back to raw HTTP via httpx (no extra SDK needed).
  3. For Ollama native API (``/api/chat``), auto-detect from base_url.

Supported servers (all work out of the box):
  - Ollama          http://localhost:11434      (native or /v1)
  - llama.cpp       http://localhost:8080/v1
  - vLLM            http://localhost:8000/v1
  - LM Studio       http://localhost:1234/v1
  - text-gen-webui  http://localhost:5000/v1
"""

from __future__ import annotations

import logging

from luna.llm_bridge.bridge import LLMBridge, LLMBridgeError, LLMResponse

log = logging.getLogger(__name__)

# Timeout for local inference — large models can be slow.
_DEFAULT_TIMEOUT = 300.0


def _is_ollama_native(base_url: str) -> bool:
    """Detect if base_url points to Ollama's native ``/api/chat`` endpoint.

    Heuristic:
      - URL contains ``/api/`` → native
      - URL ends with ``:11434`` (no ``/v1``) → native
      - Everything else → OpenAI-compatible
    """
    url = base_url.rstrip("/")
    if "/api/" in url or url.endswith("/api"):
        return True
    if ":11434" in url and "/v1" not in url:
        return True
    return False


class LocalProvider(LLMBridge):
    """Local LLM provider — zero-config for any local server.

    Prefers the ``openai`` SDK if installed (avoids reinventing the wheel).
    Falls back to raw ``httpx`` so users with only a local server don't
    need to ``pip install openai``.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._ollama_native = _is_ollama_native(base_url)

        # Try openai SDK first (existing installs keep working).
        try:
            import openai

            self._backend = "openai_sdk"
            # For Ollama native, force the SDK to use /v1 compat endpoint.
            sdk_url = base_url if not self._ollama_native else self._base_url + "/v1"
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=sdk_url,
            )
            self._http = None
            log.info("Local provider: openai SDK backend, model=%s", model)
        except ImportError:
            import httpx

            self._backend = "httpx"
            self._client = None
            self._http = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
            log.info(
                "Local provider: httpx backend (%s), model=%s",
                "ollama-native" if self._ollama_native else "openai-compat",
                model,
            )

    # ------------------------------------------------------------------
    #  LLMBridge interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if self._backend == "openai_sdk":
            return await self._complete_openai_sdk(
                messages, system_prompt, max_tokens, temperature,
            )
        if self._ollama_native:
            return await self._complete_ollama_native(
                messages, system_prompt, max_tokens, temperature,
            )
        return await self._complete_openai_compat(
            messages, system_prompt, max_tokens, temperature,
        )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    # ------------------------------------------------------------------
    #  Backend: openai SDK (preferred)
    # ------------------------------------------------------------------

    async def _complete_openai_sdk(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        assert self._client is not None  # noqa: S101
        full_messages = self._prepend_system(messages, system_prompt)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise LLMBridgeError(
                f"Local LLM error: {exc}",
                provider="local",
                original=exc,
            ) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or self._model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    # ------------------------------------------------------------------
    #  Backend: httpx — OpenAI-compatible (llama.cpp, vLLM, LM Studio…)
    # ------------------------------------------------------------------

    async def _complete_openai_compat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        assert self._http is not None  # noqa: S101
        full_messages = self._prepend_system(messages, system_prompt)

        url = self._base_url
        if "/v1" not in url:
            url += "/v1"
        url += "/chat/completions"

        payload = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = self._build_headers()

        try:
            resp = await self._http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        except Exception as exc:
            raise LLMBridgeError(
                f"Local LLM error: {exc}",
                provider="local",
                original=exc,
            ) from exc

        data = resp.json()
        usage = data.get("usage") or {}
        return LLMResponse(
            content=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    # ------------------------------------------------------------------
    #  Backend: httpx — Ollama native (/api/chat)
    # ------------------------------------------------------------------

    async def _complete_ollama_native(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        assert self._http is not None  # noqa: S101
        full_messages = self._prepend_system(messages, system_prompt)

        url = self._base_url
        if not url.endswith("/api/chat"):
            url = url.rstrip("/") + "/api/chat"

        payload: dict = {
            "model": self._model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = await self._http.post(url, json=payload)
            resp.raise_for_status()
        except Exception as exc:
            raise LLMBridgeError(
                f"Ollama error: {exc}",
                provider="local",
                original=exc,
            ) from exc

        data = resp.json()
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", self._model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepend_system(
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> list[dict[str, str]]:
        if not system_prompt:
            return messages
        return [{"role": "system", "content": system_prompt}] + list(messages)

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "ollama":
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
