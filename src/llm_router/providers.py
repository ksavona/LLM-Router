from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .settings import ProviderEndpoint, provider_api_key, provider_requires_key


@dataclass(slots=True)
class ProviderResponse:
    provider_id: str
    model: str
    content: str
    raw: dict[str, Any]
    usage: dict[str, Any] | None = None


class ProviderRequestError(RuntimeError):
    def __init__(self, provider_id: str, reason: str, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.provider_id = provider_id
        self.reason = reason
        self.status_code = status_code


class ProviderClient:
    def __init__(self, endpoint: ProviderEndpoint, timeout_seconds: float = 90.0) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ProviderResponse:
        kind = self.endpoint.kind.strip().lower()
        if kind == "anthropic":
            return await self._anthropic_chat(model, messages, temperature, max_tokens)
        if kind == "google":
            return await self._google_chat(model, messages, temperature, max_tokens)
        if kind == "cohere":
            return await self._cohere_chat(model, messages, temperature, max_tokens)
        if kind == "azure_openai":
            return await self._azure_openai_chat(model, messages, temperature, max_tokens)
        return await self._openai_compat_chat(model, messages, temperature, max_tokens)

    def _auth_headers(self) -> dict[str, str]:
        if not provider_requires_key(self.endpoint):
            return {}
        token = provider_api_key(self.endpoint)
        if not token:
            raise ProviderRequestError(self.endpoint.id, "Missing API key/token for provider")
        return {"Authorization": f"Bearer {token}"}

    async def _openai_compat_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ProviderResponse:
        base = self.endpoint.base_url.rstrip("/")
        url = f"{base}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload, headers=self._auth_headers())
            except httpx.HTTPError as exc:
                raise ProviderRequestError(self.endpoint.id, str(exc)) from exc

        if response.status_code >= 400:
            raise ProviderRequestError(self.endpoint.id, _error_reason(response), response.status_code)

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderRequestError(self.endpoint.id, "Provider returned no choices", response.status_code)

        message = choices[0].get("message") or {}
        content = _message_content_to_text(message.get("content") or "")
        return ProviderResponse(
            provider_id=self.endpoint.id,
            model=str(data.get("model") or model),
            content=str(content),
            raw=data,
            usage=_normalize_usage(data.get("usage")),
        )

    async def _anthropic_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ProviderResponse:
        base = self.endpoint.base_url.rstrip("/")
        url = f"{base}/v1/messages"
        token = provider_api_key(self.endpoint)
        if not token:
            raise ProviderRequestError(self.endpoint.id, "Missing API key/token for provider")

        payload: dict[str, Any] = {
            "model": model,
            "messages": _to_anthropic_messages(messages),
            "max_tokens": max_tokens or 1024,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ProviderRequestError(self.endpoint.id, str(exc)) from exc

        if response.status_code >= 400:
            raise ProviderRequestError(self.endpoint.id, _error_reason(response), response.status_code)

        data = response.json()
        content_blocks = data.get("content") or []
        text_parts = [str(part.get("text", "")) for part in content_blocks if part.get("type") == "text"]
        text = "".join(text_parts)
        usage = {
            "input_tokens": (data.get("usage") or {}).get("input_tokens"),
            "output_tokens": (data.get("usage") or {}).get("output_tokens"),
        }
        return ProviderResponse(
            provider_id=self.endpoint.id,
            model=str(data.get("model") or model),
            content=text,
            raw=data,
            usage=_normalize_usage(usage),
        )

    async def _google_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ProviderResponse:
        base = self.endpoint.base_url.rstrip("/")
        token = provider_api_key(self.endpoint)
        if not token:
            raise ProviderRequestError(self.endpoint.id, "Missing API key/token for provider")

        url = f"{base}/v1beta/models/{model}:generateContent?key={token}"
        payload: dict[str, Any] = {
            "contents": _to_google_contents(messages),
        }
        generation_cfg: dict[str, Any] = {}
        if temperature is not None:
            generation_cfg["temperature"] = temperature
        if max_tokens is not None:
            generation_cfg["maxOutputTokens"] = max_tokens
        if generation_cfg:
            payload["generationConfig"] = generation_cfg

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload)
            except httpx.HTTPError as exc:
                raise ProviderRequestError(self.endpoint.id, str(exc)) from exc

        if response.status_code >= 400:
            raise ProviderRequestError(self.endpoint.id, _error_reason(response), response.status_code)

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderRequestError(self.endpoint.id, "Provider returned no candidates", response.status_code)

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts)
        usage = data.get("usageMetadata")
        return ProviderResponse(
            provider_id=self.endpoint.id,
            model=model,
            content=text,
            raw=data,
            usage=_normalize_usage(usage),
        )

    async def _cohere_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ProviderResponse:
        base = self.endpoint.base_url.rstrip("/")
        url = f"{base}/v2/chat"
        token = provider_api_key(self.endpoint)
        if not token:
            raise ProviderRequestError(self.endpoint.id, "Missing API key/token for provider")

        payload: dict[str, Any] = {
            "model": model,
            "messages": _to_cohere_messages(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ProviderRequestError(self.endpoint.id, str(exc)) from exc

        if response.status_code >= 400:
            raise ProviderRequestError(self.endpoint.id, _error_reason(response), response.status_code)

        data = response.json()
        text = _cohere_text(data)
        return ProviderResponse(
            provider_id=self.endpoint.id,
            model=str(data.get("model") or model),
            content=text,
            raw=data,
            usage=_normalize_usage(data.get("usage")),
        )

    async def _azure_openai_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ProviderResponse:
        base = self.endpoint.base_url.rstrip("/")
        token = provider_api_key(self.endpoint)
        if not token:
            raise ProviderRequestError(self.endpoint.id, "Missing API key/token for provider")

        deployment = str(self.endpoint.metadata.get("deployment") or model).strip()
        api_version = str(self.endpoint.metadata.get("api_version") or "2024-10-21").strip()
        url = f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "api-key": token,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ProviderRequestError(self.endpoint.id, str(exc)) from exc

        if response.status_code >= 400:
            raise ProviderRequestError(self.endpoint.id, _error_reason(response), response.status_code)

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderRequestError(self.endpoint.id, "Provider returned no choices", response.status_code)

        message = choices[0].get("message") or {}
        content = _message_content_to_text(message.get("content") or "")
        return ProviderResponse(
            provider_id=self.endpoint.id,
            model=str(data.get("model") or model),
            content=str(content),
            raw=data,
            usage=_normalize_usage(data.get("usage")),
        )


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
        else:
            text = str(content)

        if role not in {"user", "assistant"}:
            role = "user"
        converted.append({"role": role, "content": [{"type": "text", "text": text}]})
    return converted


def _to_google_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
        else:
            text = str(content)
        google_role = "model" if role == "assistant" else "user"
        converted.append({"role": google_role, "parts": [{"text": text}]})
    return converted


def _to_cohere_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        if role == "system":
            role = "user"
        if role not in {"user", "assistant"}:
            role = "user"
        content = _message_content_to_text(msg.get("content") or "")
        converted.append({"role": role, "content": content})
    return converted


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
        return "\n".join(x for x in parts if x)
    return str(content)


def _cohere_text(data: dict[str, Any]) -> str:
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    texts.append(str(part.get("text", "")))
            joined = "".join(texts).strip()
            if joined:
                return joined

    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text

    generations = data.get("generations")
    if isinstance(generations, list) and generations:
        first = generations[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))

    return ""


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None

    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")

    if prompt is None and usage.get("input_tokens") is not None:
        prompt = usage.get("input_tokens")
    if completion is None and usage.get("output_tokens") is not None:
        completion = usage.get("output_tokens")
    if total is None and isinstance(prompt, int) and isinstance(completion, int):
        total = prompt + completion

    normalized = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    for k, v in usage.items():
        if k not in normalized:
            normalized[k] = v
    return normalized


def _error_reason(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = None

    if isinstance(data, dict):
        if isinstance(data.get("error"), dict):
            msg = data["error"].get("message")
            if msg:
                return str(msg)
        msg = data.get("message")
        if msg:
            return str(msg)

    text = response.text.strip()
    if text:
        return text[:400]
    return f"HTTP {response.status_code}"
