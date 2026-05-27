from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import subprocess
from typing import Any

import yaml


def default_runtime_dir() -> Path:
    return Path.home() / ".llm-router"


def default_settings_path() -> Path:
    return default_runtime_dir() / "config.yaml"


def default_router_config_path() -> Path:
    return Path.home() / ".hermes" / "plugins" / "hermes-smart-router" / "router_config.yaml"


@dataclass(slots=True)
class ProviderEndpoint:
    id: str
    kind: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    auth_mode: str = "api_key"
    auth_url: str | None = None
    default_model: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeSettings:
    host: str = "127.0.0.1"
    port: int = 8142
    request_timeout_seconds: float = 90.0
    require_client_api_key: bool = False
    client_api_key: str | None = None
    cors_allow_origins: list[str] = field(default_factory=lambda: ["*"])
    append_router_stamp: bool = True
    router_config_path: str = str(default_router_config_path())
    route_provider_bindings: dict[str, list[str]] = field(
        default_factory=lambda: {
            "codex": ["openai", "xai", "mistral", "groq", "together", "openrouter", "github_models"],
            "openai-codex": ["openai", "xai", "mistral", "groq", "together", "openrouter", "github_models"],
            "copilot": ["github_models", "openrouter", "openai", "xai", "groq", "together", "mistral"],
            "gemini": ["google", "openrouter", "openai"],
            "google-gemini-cli": ["google", "openrouter", "openai"],
            "anthropic": ["anthropic", "openrouter", "openai", "mistral"],
            "claude": ["anthropic", "openrouter", "openai"],
        }
    )


@dataclass(slots=True)
class LLMRouterSettings:
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    providers: list[ProviderEndpoint] = field(default_factory=list)


DEFAULT_PROVIDER_TEMPLATES: list[ProviderEndpoint] = [
    ProviderEndpoint(
        id="openai_codex",
        kind="openai_compat",
        base_url="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-5-codex",
        auth_mode="oauth_or_api_key",
        auth_url="https://platform.openai.com/",
    ),
    ProviderEndpoint(
        id="openai",
        kind="openai_compat",
        base_url="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
        auth_mode="api_key",
        auth_url="https://platform.openai.com/api-keys",
    ),
    ProviderEndpoint(
        id="anthropic",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-3-5-sonnet-latest",
        auth_mode="api_key",
        auth_url="https://console.anthropic.com/settings/keys",
    ),
    ProviderEndpoint(
        id="google",
        kind="google",
        base_url="https://generativelanguage.googleapis.com",
        api_key_env="GOOGLE_API_KEY",
        default_model="gemini-2.5-pro",
        auth_mode="api_key",
        auth_url="https://aistudio.google.com/apikey",
    ),
    ProviderEndpoint(
        id="openrouter",
        kind="openai_compat",
        base_url="https://openrouter.ai/api",
        api_key_env="OPENROUTER_API_KEY",
        default_model="openai/gpt-4.1-mini",
        auth_mode="api_key",
        auth_url="https://openrouter.ai/keys",
    ),
    ProviderEndpoint(
        id="github_models",
        kind="openai_compat",
        base_url="https://models.inference.ai.azure.com",
        api_key_env="GITHUB_TOKEN",
        default_model="gpt-4.1-mini",
        auth_mode="token",
        auth_url="https://github.com/settings/tokens",
    ),
    ProviderEndpoint(
        id="github_copilot",
        kind="openai_compat",
        base_url="https://models.inference.ai.azure.com",
        api_key_env="GITHUB_TOKEN",
        default_model="gpt-4.1-mini",
        auth_mode="gh_or_token",
        auth_url="https://github.com/settings/tokens",
    ),
    ProviderEndpoint(
        id="xai",
        kind="openai_compat",
        base_url="https://api.x.ai",
        api_key_env="XAI_API_KEY",
        default_model="grok-3-mini",
        auth_mode="api_key",
        auth_url="https://console.x.ai",
    ),
    ProviderEndpoint(
        id="mistral",
        kind="openai_compat",
        base_url="https://api.mistral.ai",
        api_key_env="MISTRAL_API_KEY",
        default_model="mistral-small-latest",
        auth_mode="api_key",
        auth_url="https://console.mistral.ai/api-keys/",
    ),
    ProviderEndpoint(
        id="groq",
        kind="openai_compat",
        base_url="https://api.groq.com/openai",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        auth_mode="api_key",
        auth_url="https://console.groq.com/keys",
    ),
    ProviderEndpoint(
        id="together",
        kind="openai_compat",
        base_url="https://api.together.xyz",
        api_key_env="TOGETHER_API_KEY",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        auth_mode="api_key",
        auth_url="https://api.together.xyz/settings/api-keys",
    ),
    ProviderEndpoint(
        id="cohere",
        kind="cohere",
        base_url="https://api.cohere.com",
        api_key_env="COHERE_API_KEY",
        default_model="command-r-plus",
        auth_mode="api_key",
        auth_url="https://dashboard.cohere.com/api-keys",
    ),
    ProviderEndpoint(
        id="azure_openai",
        kind="azure_openai",
        base_url="https://YOUR-RESOURCE.openai.azure.com",
        api_key_env="AZURE_OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
        auth_mode="api_key",
        metadata={
            "api_version": "2024-10-21",
            "deployment": "gpt-4.1-mini",
        },
    ),
]


def default_settings() -> LLMRouterSettings:
    return LLMRouterSettings(providers=[ProviderEndpoint(**asdict(p)) for p in DEFAULT_PROVIDER_TEMPLATES])


def load_settings(path: Path | None = None) -> LLMRouterSettings:
    target = path or default_settings_path()
    if not target.exists():
        cfg = default_settings()
        save_settings(cfg, target)
        return cfg

    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    runtime_raw = raw.get("runtime") or {}
    providers_raw = raw.get("providers") or []

    runtime = RuntimeSettings(
        host=str(runtime_raw.get("host", "127.0.0.1")),
        port=int(runtime_raw.get("port", 8142)),
        request_timeout_seconds=float(runtime_raw.get("request_timeout_seconds", 90.0)),
        require_client_api_key=bool(runtime_raw.get("require_client_api_key", False)),
        client_api_key=runtime_raw.get("client_api_key"),
        cors_allow_origins=list(runtime_raw.get("cors_allow_origins", ["*"])),
        append_router_stamp=bool(runtime_raw.get("append_router_stamp", True)),
        router_config_path=str(runtime_raw.get("router_config_path", default_router_config_path())),
        route_provider_bindings=dict(runtime_raw.get("route_provider_bindings", RuntimeSettings().route_provider_bindings)),
    )

    providers: list[ProviderEndpoint] = []
    for p in providers_raw:
        if not isinstance(p, dict):
            continue
        providers.append(
            ProviderEndpoint(
                id=str(p.get("id", "")).strip(),
                kind=str(p.get("kind", "openai_compat")).strip(),
                base_url=str(p.get("base_url", "")).strip(),
                api_key=p.get("api_key"),
                api_key_env=p.get("api_key_env"),
                auth_mode=str(p.get("auth_mode", "api_key")),
                auth_url=p.get("auth_url"),
                default_model=p.get("default_model"),
                enabled=bool(p.get("enabled", True)),
                metadata=dict(p.get("metadata", {})),
            )
        )

    if not providers:
        providers = [ProviderEndpoint(**asdict(p)) for p in DEFAULT_PROVIDER_TEMPLATES]

    return LLMRouterSettings(runtime=runtime, providers=providers)


def save_settings(settings: LLMRouterSettings, path: Path | None = None) -> None:
    target = path or default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(_to_serializable(asdict(settings)), sort_keys=False), encoding="utf-8")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def provider_api_key(provider: ProviderEndpoint) -> str | None:
    import os

    if provider.api_key:
        return provider.api_key
    env_name = str(provider.api_key_env or "").strip()
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value

    provider_id = str(provider.id or "").strip().lower()
    auth_mode = str(provider.auth_mode or "").strip().lower()
    if provider_id in {"github_models", "github_copilot"} or "gh" in auth_mode or "copilot" in provider_id:
        token = _gh_auth_token()
        if token:
            return token
    return None


def provider_requires_key(provider: ProviderEndpoint) -> bool:
    auth_mode = str(provider.auth_mode or "api_key").strip().lower()
    return auth_mode not in {"none", "local", "unauthenticated"}


def _gh_auth_token() -> str | None:
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    token = str(proc.stdout or "").strip()
    return token or None
