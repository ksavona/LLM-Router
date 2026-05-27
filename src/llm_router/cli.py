from __future__ import annotations

import asyncio
import argparse
from dataclasses import asdict
import getpass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import webbrowser

import httpx
import uvicorn

from .api import build_app
from .service import RouterExecutionService
from .settings import (
    DEFAULT_PROVIDER_TEMPLATES,
    ProviderEndpoint,
    default_settings_path,
    load_settings,
    provider_api_key,
    provider_requires_key,
    save_settings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-router")
    parser.add_argument("--config", type=Path, default=default_settings_path(), help="Path to llm-router config")

    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Launch interactive setup menu")
    setup.add_argument("--quick", action="store_true", help="Non-interactive defaults")
    sub.add_parser("settup", help="Alias for setup (common typo)")

    serve = sub.add_parser("serve", help="Start OpenAI-compatible provider server")
    serve.add_argument("--host", type=str, default=None, help="Bind host")
    serve.add_argument("--port", type=int, default=None, help="Bind port")

    sub.add_parser("status", help="Show provider and auth status")
    doctor = sub.add_parser("doctor", help="Run diagnostics for provider readiness")
    doctor.add_argument("--probe-network", action="store_true", help="Attempt lightweight HTTP probe for provider endpoints")

    auth = sub.add_parser("auth", help="Open browser authentication pages for providers")
    auth.add_argument("--provider", type=str, default="all", help="Provider id or 'all'")

    test = sub.add_parser("test", help="Send a test prompt and inspect routed provider/model")
    test.add_argument("prompt", nargs="?", default=None, help="Prompt text to test")
    test.add_argument("--model", type=str, default="auto", help="Model hint to pass to router")
    test.add_argument("--temperature", type=float, default=None, help="Optional temperature")
    test.add_argument("--max-tokens", type=int, default=None, help="Optional max tokens")
    return parser


def cmd_setup(config_path: Path, quick: bool = False) -> int:
    settings = load_settings(config_path)
    settings.providers = _merge_provider_templates(settings.providers)

    if quick:
        save_settings(settings, config_path)
        print(f"Saved default setup to {config_path}")
        return 0

    should_save = _interactive_setup_menu(settings)
    if not should_save:
        _clear_screen()
        print("Setup canceled. No changes saved.")
        return 0

    save_settings(settings, config_path)
    _clear_screen()
    print(f"Setup saved to {config_path}")
    print("Run: llm-router serve")
    return 0


ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"
ANSI_INVERT = "\033[7m"


PROVIDER_CATALOG: list[dict[str, str | None]] = [
    {"name": "Nous Portal (Nous Research subscription)", "id": "nous_portal", "kind": "openai_compat", "base_url": "https://portal.nousresearch.com/api", "api_key_env": "NOUS_API_KEY", "auth_mode": "api_key", "auth_url": "https://portal.nousresearch.com", "default_model": "hermes-3"},
    {"name": "OpenRouter (100+ models, pay-per-use)", "id": "openrouter", "kind": "openai_compat", "base_url": "https://openrouter.ai/api", "api_key_env": "OPENROUTER_API_KEY", "auth_mode": "api_key", "auth_url": "https://openrouter.ai/keys", "default_model": "openai/gpt-4.1-mini"},
    {"name": "NovitaAI (AI-native cloud: Model API, Agent Sandbox, GPU Cloud)", "id": "novitaai", "kind": "openai_compat", "base_url": "https://api.novita.ai", "api_key_env": "NOVITA_API_KEY", "auth_mode": "api_key", "auth_url": "https://novita.ai", "default_model": "deepseek-v3"},
    {"name": "LM Studio (local desktop app with built-in model server)", "id": "lm_studio", "kind": "openai_compat", "base_url": "http://127.0.0.1:1234", "api_key_env": None, "auth_mode": "local", "auth_url": "https://lmstudio.ai", "default_model": "local-model"},
    {"name": "Anthropic (Claude models - API key or Claude Code)", "id": "anthropic", "kind": "anthropic", "base_url": "https://api.anthropic.com", "api_key_env": "ANTHROPIC_API_KEY", "auth_mode": "api_key", "auth_url": "https://console.anthropic.com/settings/keys", "default_model": "claude-3-5-sonnet-latest"},
    {"name": "OpenAI Codex", "id": "openai_codex", "kind": "openai_compat", "base_url": "https://api.openai.com", "api_key_env": "OPENAI_API_KEY", "auth_mode": "oauth_or_api_key", "auth_url": "https://platform.openai.com", "default_model": "gpt-5-codex"},
    {"name": "Qwen Cloud / DashScope Coding (Qwen + multi-provider)", "id": "qwen_dashscope", "kind": "openai_compat", "base_url": "https://dashscope.aliyuncs.com/compatible-mode", "api_key_env": "DASHSCOPE_API_KEY", "auth_mode": "api_key", "auth_url": "https://dashscope.console.aliyun.com", "default_model": "qwen-plus"},
    {"name": "xAI Grok OAuth (SuperGrok Subscription)", "id": "xai_grok_oauth", "kind": "openai_compat", "base_url": "https://api.x.ai", "api_key_env": "XAI_API_KEY", "auth_mode": "oauth", "auth_url": "https://console.x.ai", "default_model": "grok-3-mini"},
    {"name": "Xiaomi MiMo (MiMo-V2.5 and V2 models - pro, omni, flash)", "id": "xiaomi_mimo", "kind": "openai_compat", "base_url": "https://api.mimo.mi.com", "api_key_env": "MIMO_API_KEY", "auth_mode": "api_key", "auth_url": "https://mimo.mi.com", "default_model": "mimo-v2.5-pro"},
    {"name": "Tencent TokenHub (Hy3 Preview - direct API via tokenhub.tencentmaas.com)", "id": "tencent_tokenhub", "kind": "openai_compat", "base_url": "https://tokenhub.tencentmaas.com", "api_key_env": "TENCENT_TOKENHUB_API_KEY", "auth_mode": "api_key", "auth_url": "https://tokenhub.tencentmaas.com", "default_model": "hy3-preview"},
    {"name": "NVIDIA NIM (Nemotron models - build.nvidia.com or local NIM)", "id": "nvidia_nim", "kind": "openai_compat", "base_url": "https://integrate.api.nvidia.com", "api_key_env": "NVIDIA_API_KEY", "auth_mode": "api_key", "auth_url": "https://build.nvidia.com", "default_model": "nvidia/llama-3.1-nemotron-70b-instruct"},
    {"name": "GitHub Copilot (uses GITHUB_TOKEN or gh auth token)", "id": "github_copilot", "kind": "openai_compat", "base_url": "https://models.inference.ai.azure.com", "api_key_env": "GITHUB_TOKEN", "auth_mode": "gh_or_token", "auth_url": "https://github.com/settings/tokens", "default_model": "gpt-4.1-mini"},
    {"name": "GitHub Copilot ACP (spawns copilot --acp --stdio)", "id": "github_copilot_acp", "kind": "openai_compat", "base_url": "https://models.inference.ai.azure.com", "api_key_env": "GITHUB_TOKEN", "auth_mode": "gh_or_token", "auth_url": "https://github.com/settings/tokens", "default_model": "gpt-4.1-mini"},
    {"name": "Hugging Face Inference Providers (20+ open models)", "id": "huggingface_inference", "kind": "openai_compat", "base_url": "https://router.huggingface.co", "api_key_env": "HF_TOKEN", "auth_mode": "token", "auth_url": "https://huggingface.co/settings/tokens", "default_model": "meta-llama/Llama-3.1-8B-Instruct"},
    {"name": "Google AI Studio (Gemini models - native Gemini API)", "id": "google", "kind": "google", "base_url": "https://generativelanguage.googleapis.com", "api_key_env": "GOOGLE_API_KEY", "auth_mode": "api_key", "auth_url": "https://aistudio.google.com/apikey", "default_model": "gemini-2.5-pro"},
    {"name": "Google Gemini via OAuth + Code Assist (free tier supported; no API key needed)", "id": "google_gemini_oauth", "kind": "google", "base_url": "https://generativelanguage.googleapis.com", "api_key_env": "GOOGLE_API_KEY", "auth_mode": "oauth", "auth_url": "https://aistudio.google.com", "default_model": "gemini-2.5-pro"},
    {"name": "DeepSeek (DeepSeek-V3, R1, coder - direct API)", "id": "deepseek", "kind": "openai_compat", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "auth_mode": "api_key", "auth_url": "https://platform.deepseek.com", "default_model": "deepseek-chat"},
    {"name": "xAI (Grok models - direct API)", "id": "xai", "kind": "openai_compat", "base_url": "https://api.x.ai", "api_key_env": "XAI_API_KEY", "auth_mode": "api_key", "auth_url": "https://console.x.ai", "default_model": "grok-3-mini"},
    {"name": "Z.AI / GLM (Zhipu AI direct API)", "id": "zai_glm", "kind": "openai_compat", "base_url": "https://open.bigmodel.cn/api/paas", "api_key_env": "ZHIPU_API_KEY", "auth_mode": "api_key", "auth_url": "https://open.bigmodel.cn", "default_model": "glm-4-plus"},
    {"name": "Kimi Coding Plan (api.kimi.com) & Moonshot API", "id": "kimi_coding", "kind": "openai_compat", "base_url": "https://api.kimi.com", "api_key_env": "KIMI_API_KEY", "auth_mode": "api_key", "auth_url": "https://platform.moonshot.ai", "default_model": "kimi-k2"},
    {"name": "Kimi / Moonshot China (Moonshot CN direct API)", "id": "moonshot_cn", "kind": "openai_compat", "base_url": "https://api.moonshot.cn", "api_key_env": "MOONSHOT_API_KEY", "auth_mode": "api_key", "auth_url": "https://platform.moonshot.cn", "default_model": "moonshot-v1-8k"},
    {"name": "StepFun Step Plan (agent/coding models via Step Plan API)", "id": "stepfun", "kind": "openai_compat", "base_url": "https://api.stepfun.com", "api_key_env": "STEPFUN_API_KEY", "auth_mode": "api_key", "auth_url": "https://platform.stepfun.com", "default_model": "step-2-mini"},
    {"name": "MiniMax (global direct API)", "id": "minimax", "kind": "openai_compat", "base_url": "https://api.minimax.io", "api_key_env": "MINIMAX_API_KEY", "auth_mode": "api_key", "auth_url": "https://platform.minimax.io", "default_model": "abab6.5s-chat"},
    {"name": "MiniMax via OAuth browser login (Coding Plan, minimax.io)", "id": "minimax_oauth", "kind": "openai_compat", "base_url": "https://api.minimax.io", "api_key_env": "MINIMAX_API_KEY", "auth_mode": "oauth", "auth_url": "https://platform.minimax.io", "default_model": "abab6.5s-chat"},
    {"name": "MiniMax China (domestic direct API)", "id": "minimax_cn", "kind": "openai_compat", "base_url": "https://api.minimaxi.com", "api_key_env": "MINIMAX_CN_API_KEY", "auth_mode": "api_key", "auth_url": "https://api.minimaxi.com", "default_model": "abab6.5s-chat"},
    {"name": "Ollama Cloud (cloud-hosted open models - ollama.com)", "id": "ollama_cloud", "kind": "openai_compat", "base_url": "https://ollama.com/api", "api_key_env": "OLLAMA_API_KEY", "auth_mode": "api_key", "auth_url": "https://ollama.com", "default_model": "llama3.1"},
    {"name": "Arcee AI (Trinity models - direct API)", "id": "arcee_ai", "kind": "openai_compat", "base_url": "https://api.arcee.ai", "api_key_env": "ARCEE_API_KEY", "auth_mode": "api_key", "auth_url": "https://app.arcee.ai", "default_model": "arcee-trinity"},
    {"name": "GMI Cloud (multi-model direct API)", "id": "gmi_cloud", "kind": "openai_compat", "base_url": "https://api.gmicloud.ai", "api_key_env": "GMI_API_KEY", "auth_mode": "api_key", "auth_url": "https://gmicloud.ai", "default_model": "gmi-auto"},
    {"name": "Kilo Code (Kilo Gateway API)", "id": "kilo_code", "kind": "openai_compat", "base_url": "https://api.kilo-code.com", "api_key_env": "KILO_API_KEY", "auth_mode": "api_key", "auth_url": "https://kilo-code.com", "default_model": "kilo-coder"},
    {"name": "OpenCode Zen (35+ curated models, pay-as-you-go)", "id": "opencode_zen", "kind": "openai_compat", "base_url": "https://api.opencode.ai", "api_key_env": "OPENCODE_API_KEY", "auth_mode": "api_key", "auth_url": "https://opencode.ai", "default_model": "zen-auto"},
    {"name": "OpenCode Go (open models, $10/month subscription)", "id": "opencode_go", "kind": "openai_compat", "base_url": "https://api.opencode.ai", "api_key_env": "OPENCODE_API_KEY", "auth_mode": "subscription", "auth_url": "https://opencode.ai", "default_model": "go-open"},
    {"name": "AWS Bedrock (Claude, Nova, Llama, DeepSeek - IAM or API key)", "id": "aws_bedrock", "kind": "openai_compat", "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com", "api_key_env": "AWS_BEARER_TOKEN_BEDROCK", "auth_mode": "iam_or_api_key", "auth_url": "https://console.aws.amazon.com/bedrock", "default_model": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
    {"name": "Azure Foundry (OpenAI-style or Anthropic-style endpoint - your Azure AI deployment)", "id": "azure_foundry", "kind": "openai_compat", "base_url": "https://YOUR-FOUNDRY-ENDPOINT", "api_key_env": "AZURE_FOUNDRY_API_KEY", "auth_mode": "api_key", "auth_url": "https://ai.azure.com", "default_model": "gpt-4.1-mini"},
    {"name": "Vercel AI Gateway", "id": "vercel_ai_gateway", "kind": "openai_compat", "base_url": "https://ai-gateway.vercel.sh/v1", "api_key_env": "VERCEL_AI_GATEWAY_API_KEY", "auth_mode": "api_key", "auth_url": "https://vercel.com/dashboard", "default_model": "openai/gpt-4.1-mini"},
    {"name": "Qwen OAuth (reuses local Qwen CLI login)", "id": "qwen_oauth", "kind": "openai_compat", "base_url": "https://dashscope.aliyuncs.com/compatible-mode", "api_key_env": "DASHSCOPE_API_KEY", "auth_mode": "oauth", "auth_url": "https://dashscope.console.aliyun.com", "default_model": "qwen-plus"},
    {"name": "Alibaba Cloud Coding Plan - dedicated coding tier", "id": "alibaba_coding_plan", "kind": "openai_compat", "base_url": "https://dashscope.aliyuncs.com/compatible-mode", "api_key_env": "DASHSCOPE_API_KEY", "auth_mode": "api_key", "auth_url": "https://dashscope.console.aliyun.com", "default_model": "qwen-coder-plus"},
    {"name": "custom (direct API)", "id": "custom", "kind": "openai_compat", "base_url": "https://api.example.com", "api_key_env": "CUSTOM_API_KEY", "auth_mode": "api_key", "auth_url": None, "default_model": "custom-model"},
]


def _interactive_setup_menu(settings) -> bool:
    while True:
        runtime_ready = _runtime_ready(settings)
        providers_ready, providers_total = _providers_ready_counts(settings.providers)

        options = [
            f"Runtime Settings      [{_status_tag(runtime_ready)}]",
            f"Provider Settings     [{_color_text(f'{providers_ready}/{providers_total} READY', ANSI_GREEN if providers_ready == providers_total else ANSI_YELLOW)}]",
            "Provider Catalog",
            "Add Custom Provider",
            _color_text("Save and Exit", ANSI_GREEN),
            _color_text("Exit Without Saving", ANSI_RED),
        ]
        choice = _arrow_menu(
            title="LLM ROUTER // SETUP TERMINAL",
            subtitle="Hack mode active - arrow keys navigate, Enter selects",
            options=options,
        )
        if choice is None or choice == 5:
            return False
        if choice == 0:
            _runtime_settings_menu(settings)
        elif choice == 1:
            _provider_settings_menu(settings)
        elif choice == 2:
            _provider_catalog_menu(settings)
        elif choice == 3:
            settings.providers.append(_prompt_custom_provider())
        elif choice == 4:
            return True


def _runtime_settings_menu(settings) -> None:
    while True:
        client_key_set = bool(str(settings.runtime.client_api_key or "").strip())
        require_key = bool(settings.runtime.require_client_api_key)
        key_status = _status_marker(client_key_set)
        key_requirement = _color_text("required", ANSI_YELLOW) if require_key else _color_text("optional", ANSI_CYAN)
        options = [
            f"Host                 {settings.runtime.host} {_status_marker(bool(str(settings.runtime.host).strip()))}",
            f"Port                 {settings.runtime.port} {_status_marker(settings.runtime.port > 0)}",
            f"Require Client Key   {_toggle_text(require_key)}",
            f"Client API Key       {_masked_value(settings.runtime.client_api_key)} {key_status} ({key_requirement})",
            f"Router Config Path   {settings.runtime.router_config_path} {_status_marker(bool(str(settings.runtime.router_config_path).strip()))}",
            f"Append Router Stamp  {_toggle_text(bool(settings.runtime.append_router_stamp))}",
            "Back",
        ]
        choice = _arrow_menu(
            title="RUNTIME SETTINGS",
            subtitle="Configured values glow green. Missing values glow red.",
            options=options,
        )
        if choice is None or choice == 6:
            return
        if choice == 0:
            entered = _prompt_text("Server host", settings.runtime.host)
            if entered is not None:
                settings.runtime.host = entered
        elif choice == 1:
            entered = _prompt_text("Server port", str(settings.runtime.port))
            if entered is not None:
                try:
                    parsed = int(entered)
                    if parsed > 0:
                        settings.runtime.port = parsed
                except ValueError:
                    _pause_message("Invalid port. Press Enter to continue.")
        elif choice == 2:
            settings.runtime.require_client_api_key = not settings.runtime.require_client_api_key
        elif choice == 3:
            entered = _prompt_secret("Inbound client API key", settings.runtime.client_api_key)
            if entered is not None:
                settings.runtime.client_api_key = entered
        elif choice == 4:
            entered = _prompt_text("Router config path", settings.runtime.router_config_path)
            if entered is not None:
                settings.runtime.router_config_path = entered
        elif choice == 5:
            settings.runtime.append_router_stamp = not settings.runtime.append_router_stamp


def _provider_settings_menu(settings) -> None:
    while True:
        providers = list(settings.providers)
        options: list[str] = []
        for provider in providers:
            ready = _provider_ready(provider)
            status = _status_marker(ready)
            enabled = _color_text("ENABLED", ANSI_GREEN) if provider.enabled else _color_text("DISABLED", ANSI_RED)
            options.append(f"{provider.id:<16} {enabled}  {status}")
        options.append("Add from catalog")
        options.append("Back")

        choice = _arrow_menu(
            title="PROVIDER SETTINGS",
            subtitle="Pick a provider to edit. Ready providers are green.",
            options=options,
        )
        if choice is None or choice == len(options) - 1:
            return
        if choice == len(options) - 2:
            _provider_catalog_menu(settings)
            continue
        _edit_provider_menu(providers[choice])


def _edit_provider_menu(provider: ProviderEndpoint) -> None:
    while True:
        env_name = str(provider.api_key_env or "").strip()
        env_present = bool(env_name and os.environ.get(env_name))
        key_resolved = bool(provider_api_key(provider))
        options = [
            f"Enabled              {_toggle_text(provider.enabled)}",
            f"Base URL             {provider.base_url or '-'} {_status_marker(bool(str(provider.base_url).strip()))}",
            f"Default Model        {provider.default_model or '-'} {_status_marker(bool(str(provider.default_model or '').strip()))}",
            f"API Key Env Var      {provider.api_key_env or '-'} {_status_marker(env_present)}",
            f"API Key/Token        {_masked_value(provider.api_key)} {_status_marker(bool(provider.api_key))}",
            f"Resolved Auth        {_color_text('AVAILABLE', ANSI_GREEN) if key_resolved else _color_text('MISSING', ANSI_RED)}",
            "Auth Shortcuts",
            "Back",
        ]
        choice = _arrow_menu(
            title=f"PROVIDER // {provider.id.upper()}",
            subtitle=f"kind={provider.kind}",
            options=options,
        )
        if choice is None or choice == 7:
            return
        if choice == 0:
            provider.enabled = not provider.enabled
        elif choice == 1:
            entered = _prompt_text("Base URL", provider.base_url)
            if entered is not None:
                provider.base_url = entered
        elif choice == 2:
            entered = _prompt_text("Default model", provider.default_model or "")
            if entered is not None:
                provider.default_model = entered or None
        elif choice == 3:
            entered = _prompt_text("API key env var", provider.api_key_env or "")
            if entered is not None:
                provider.api_key_env = entered or None
        elif choice == 4:
            entered = _prompt_secret("API key/token", provider.api_key)
            if entered is not None:
                provider.api_key = entered or None
        elif choice == 5:
            _provider_credentials_wizard(_provider_profile_from_endpoint(provider), provider)
        elif choice == 6:
            _provider_auth_shortcuts_menu(provider)


def _provider_auth_shortcuts_menu(provider: ProviderEndpoint) -> None:
    options = [
        "Open provider auth page",
        "Open OpenAI API keys",
        "Open GitHub token page (Codex/Copilot)",
        "Open Anthropic keys",
        "Open Google AI Studio keys",
        "Open OpenRouter keys",
        "Back",
    ]
    while True:
        choice = _arrow_menu(
            title=f"AUTH SHORTCUTS // {provider.id.upper()}",
            subtitle="Provider auth quick links",
            options=options,
        )
        if choice is None or choice == 6:
            return

        if choice == 0:
            url = provider.auth_url or _default_auth_url(provider.id)
            if url:
                webbrowser.open(url)
                _pause_message(f"Opened {url}. Press Enter to continue.")
            else:
                _pause_message("No auth URL configured. Press Enter to continue.")
            continue

        url_map = {
            1: "https://platform.openai.com/api-keys",
            2: "https://github.com/settings/tokens",
            3: "https://console.anthropic.com/settings/keys",
            4: "https://aistudio.google.com/apikey",
            5: "https://openrouter.ai/keys",
        }
        url = url_map.get(choice)
        if url:
            webbrowser.open(url)
            _pause_message(f"Opened {url}. Press Enter to continue.")


def _provider_catalog_menu(settings) -> None:
    while True:
        current_id = _active_provider_id(settings.providers)
        options: list[str] = []
        for profile in PROVIDER_CATALOG:
            pid = str(profile.get("id") or "")
            marker = "(●)" if pid == current_id else "(○)"
            options.append(f"{marker} {profile.get('name')}")
        options.append("Back")

        choice = _arrow_menu(
            title="INFERENCE PROVIDER",
            subtitle="Choose how to connect to your main chat model.",
            options=options,
        )
        if choice is None or choice == len(options) - 1:
            return

        selected = PROVIDER_CATALOG[choice]
        endpoint = _apply_provider_profile(settings.providers, selected)
        _provider_credentials_wizard(selected, endpoint)


def _active_provider_id(providers: list[ProviderEndpoint]) -> str:
    for provider in providers:
        if provider.enabled:
            return str(provider.id)
    return ""


def _apply_provider_profile(providers: list[ProviderEndpoint], profile: dict[str, str | None]) -> ProviderEndpoint:
    profile_id = str(profile.get("id") or "custom").strip().lower()
    existing = None
    for provider in providers:
        if provider.id == profile_id:
            existing = provider
            break

    if existing is None:
        existing = ProviderEndpoint(id=profile_id, kind="openai_compat", base_url="", enabled=True)
        providers.append(existing)

    existing.kind = str(profile.get("kind") or "openai_compat")
    existing.base_url = str(profile.get("base_url") or existing.base_url)
    existing.api_key_env = str(profile.get("api_key_env")) if profile.get("api_key_env") else None
    existing.auth_mode = str(profile.get("auth_mode") or existing.auth_mode)
    existing.auth_url = str(profile.get("auth_url")) if profile.get("auth_url") else None
    existing.default_model = str(profile.get("default_model")) if profile.get("default_model") else existing.default_model
    existing.enabled = True
    existing.metadata = dict(existing.metadata or {})
    existing.metadata["display_name"] = str(profile.get("name") or existing.id)
    return existing


def _provider_credentials_wizard(profile: dict[str, str | None], endpoint: ProviderEndpoint) -> None:
    profile_name = str(profile.get("name") or endpoint.id)
    provider_id = str(profile.get("id") or endpoint.id).strip().lower()

    _clear_screen()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│             Hermes-style Provider Setup                │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Configure provider credentials. Press Ctrl+C to exit. │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    print(f"Current provider: {profile_name}")

    if provider_id in {"openai_codex", "github_copilot", "github_copilot_acp"}:
        has_creds = bool(provider_api_key(endpoint))
        check = "✓" if has_creds else "✗"
        label = "OpenAI Codex credentials" if provider_id == "openai_codex" else "GitHub Copilot credentials"
        print()
        print(f"{label}: {check}")
        print()
        print("  1. Use existing credentials")
        print("  2. Reauthenticate (new OAuth login)")
        print("  3. Cancel")
        choice = input("\n  Choice [1/2/3]: ").strip()
        if choice == "1":
            return
        if choice == "2":
            if provider_id == "openai_codex":
                _openai_codex_reauth_flow(endpoint)
            else:
                _github_copilot_reauth_flow(endpoint)
            return
        return

    print("\nCredential source:")
    print("  1. API key/token")
    print("  2. OAuth/browser sign-in")
    print("  3. Skip")
    choice = input("\n  Choice [1/2/3]: ").strip()
    if choice == "1":
        entered = getpass.getpass("Paste API key/token (optional, Enter to skip): ").strip()
        if entered:
            endpoint.api_key = entered
    elif choice == "2":
        if endpoint.auth_url:
            webbrowser.open(endpoint.auth_url)
            _pause_message("Browser opened for sign in. Press Enter when done.")


def _provider_profile_from_endpoint(endpoint: ProviderEndpoint) -> dict[str, str | None]:
    display_name = str((endpoint.metadata or {}).get("display_name") or endpoint.id)
    return {
        "name": display_name,
        "id": endpoint.id,
        "kind": endpoint.kind,
        "base_url": endpoint.base_url,
        "api_key_env": endpoint.api_key_env,
        "auth_mode": endpoint.auth_mode,
        "auth_url": endpoint.auth_url,
        "default_model": endpoint.default_model,
    }


def _openai_codex_reauth_flow(endpoint: ProviderEndpoint) -> None:
    _clear_screen()
    print("Starting a fresh OpenAI Codex login...\n")
    print("Signing in to OpenAI Codex...")
    print("(LLM Router creates its own session - won't affect Codex CLI or VS Code)\n")

    if not _command_exists("codex"):
        print("Codex CLI is not installed.\n")
        print("  1. Install Codex CLI now")
        print("  2. Continue with API key fallback")
        print("  3. Cancel")
        choice = input("\n  Choice [1/2/3]: ").strip()
        if choice == "1":
            installed = _install_openai_codex_cli()
            if not installed:
                _pause_message("Codex CLI install failed. Press Enter for fallback options.")
            elif not _command_exists("codex"):
                _pause_message("Codex CLI installed but not yet on PATH in this terminal. Open a new terminal or continue with fallback.")
        elif choice == "3":
            return

    if _command_exists("codex"):
        print("Launching Codex CLI auth flow...\n")
        try:
            result = subprocess.run(["codex", "login", "--device-auth"], check=False)
        except Exception:
            result = None

        if result is not None and result.returncode == 0:
            _pause_message("OpenAI Codex auth detected. Press Enter to continue.")
            return
        if _codex_login_active():
            _pause_message("OpenAI Codex is already logged in. Press Enter to continue.")
            return
        _pause_message("Codex CLI login did not complete successfully. Press Enter for fallback options.")
    else:
        _pause_message("Codex CLI is not installed, so device-code login cannot be generated here. Press Enter for fallback options.")

    print("Fallback options:")
    print("  1. Open OpenAI API keys page")
    print("  2. Paste OPENAI_API_KEY manually")
    print("  3. Cancel")
    choice = input("\n  Choice [1/2/3]: ").strip()
    if choice == "1":
        webbrowser.open("https://platform.openai.com/api-keys")
    if choice not in {"1", "2"}:
        return
    entered = getpass.getpass("\nPaste OPENAI_API_KEY (optional): ").strip()
    if entered:
        endpoint.api_key = entered


def _github_copilot_reauth_flow(endpoint: ProviderEndpoint) -> None:
    _clear_screen()
    print("Starting a fresh GitHub Copilot login...\n")
    print("Signing in to GitHub Copilot...")
    print("(LLM Router creates its own session - won't affect VS Code sign-in)\n")

    if not _command_exists("gh"):
        print("GitHub CLI (gh) is not installed.\n")
        print("  1. Install GitHub CLI now")
        print("  2. Continue with token fallback")
        print("  3. Cancel")
        choice = input("\n  Choice [1/2/3]: ").strip()
        if choice == "1":
            installed = _install_github_cli()
            if not installed:
                _pause_message("GitHub CLI install failed. Press Enter for fallback options.")
            elif not _command_exists("gh"):
                _pause_message("GitHub CLI installed but not yet on PATH in this terminal. Open a new terminal or continue with fallback.")
        elif choice == "3":
            return

    if _command_exists("gh"):
        print("Launching GitHub CLI auth flow in browser...\n")
        try:
            subprocess.run(["gh", "auth", "login", "-w"], check=False)
        except Exception:
            pass
        if provider_api_key(endpoint):
            _pause_message("GitHub auth detected via gh auth token. Press Enter to continue.")
            return

    token_url = "https://github.com/settings/tokens"
    print("Open this URL to create or review your token:")
    print(f"  {token_url}\n")
    webbrowser.open(token_url)
    entered = getpass.getpass("Paste GITHUB_TOKEN (optional): ").strip()
    if entered:
        endpoint.api_key = entered


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def _codex_login_active() -> bool:
    if not _command_exists("codex"):
        return False
    try:
        result = subprocess.run(["codex", "login", "status"], check=False)
    except Exception:
        return False
    return result.returncode == 0


def _install_openai_codex_cli() -> bool:
    if not _command_exists("npm"):
        print("npm is required to install Codex CLI but was not found.")
        return False
    print("Installing Codex CLI with npm...\n")
    try:
        result = subprocess.run(["npm", "install", "-g", "@openai/codex"], check=False)
    except Exception:
        return False
    return result.returncode == 0


def _install_github_cli() -> bool:
    print("Installing GitHub CLI...\n")
    try:
        if os.name == "nt" and _command_exists("winget"):
            result = subprocess.run(
                [
                    "winget",
                    "install",
                    "--id",
                    "GitHub.cli",
                    "-e",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
                check=False,
            )
            return result.returncode == 0

        if sys.platform == "darwin" and _command_exists("brew"):
            result = subprocess.run(["brew", "install", "gh"], check=False)
            return result.returncode == 0

        if sys.platform.startswith("linux") and _command_exists("apt-get"):
            result = subprocess.run(["sudo", "apt-get", "install", "-y", "gh"], check=False)
            return result.returncode == 0
    except Exception:
        return False

    print("Automatic GitHub CLI installation is not supported on this OS/shell.")
    return False


def _runtime_ready(settings) -> bool:
    host_ok = bool(str(settings.runtime.host or "").strip())
    port_ok = int(settings.runtime.port) > 0
    path_ok = bool(str(settings.runtime.router_config_path or "").strip())
    key_ok = True
    if settings.runtime.require_client_api_key:
        key_ok = bool(str(settings.runtime.client_api_key or "").strip())
    return host_ok and port_ok and path_ok and key_ok


def _providers_ready_counts(providers: list[ProviderEndpoint]) -> tuple[int, int]:
    ready = sum(1 for p in providers if _provider_ready(p))
    return ready, len(providers)


def _provider_ready(provider: ProviderEndpoint) -> bool:
    if not provider.enabled:
        return False
    if not str(provider.base_url or "").strip():
        return False
    if not provider_requires_key(provider):
        return True
    return bool(provider_api_key(provider))


def _status_marker(ok: bool) -> str:
    return _color_text("[SET]", ANSI_GREEN) if ok else _color_text("[MISSING]", ANSI_RED)


def _status_tag(ok: bool) -> str:
    return _color_text("READY", ANSI_GREEN) if ok else _color_text("INCOMPLETE", ANSI_RED)


def _toggle_text(value: bool) -> str:
    return _color_text("ON", ANSI_GREEN) if value else _color_text("OFF", ANSI_RED)


def _masked_value(value: str | None) -> str:
    if not value:
        return _color_text("<none>", ANSI_RED)
    visible = min(4, len(value))
    return _color_text("*" * max(len(value) - visible, 0) + value[-visible:], ANSI_GREEN)


def _color_text(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _arrow_menu(title: str, subtitle: str, options: list[str]) -> int | None:
    index = 0
    while True:
        _clear_screen()
        print(_color_text("#############################################", ANSI_MAGENTA))
        print(_color_text("#            LLM ROUTER TERMINAL            #", ANSI_MAGENTA))
        print(_color_text("#############################################", ANSI_MAGENTA))
        print(_color_text(title, ANSI_CYAN))
        print(_color_text(subtitle, ANSI_YELLOW))
        print(_color_text(f"Selection: {index + 1}/{len(options)}", ANSI_YELLOW))
        print()
        for i, option in enumerate(options):
            if i == index:
                selected = f"[*] {option}"
                print(_color_text(f"{ANSI_INVERT}{selected}{ANSI_RESET}", ANSI_CYAN))
            else:
                print(f"[ ] {option}")
        print()
        print(_color_text("UP/DOWN or W/S = navigate | ENTER = select | ESC = back", ANSI_YELLOW))

        key = _read_menu_key()
        if key == "up":
            index = (index - 1) % len(options)
        elif key == "down":
            index = (index + 1) % len(options)
        elif key == "enter":
            return index
        elif key == "escape":
            return None


def _read_menu_key() -> str:
    if os.name == "nt":
        import msvcrt

        while True:
            char = msvcrt.getwch()
            if char in {"\r", "\n"}:
                return "enter"
            if char == "\x1b":
                if msvcrt.kbhit():
                    nxt = msvcrt.getwch()
                    if nxt == "[" and msvcrt.kbhit():
                        final = msvcrt.getwch()
                        if final == "A":
                            return "up"
                        if final == "B":
                            return "down"
                return "escape"
            if char in {"\x00", "\xe0"}:
                code = msvcrt.getwch()
                if code == "H":
                    return "up"
                if code == "P":
                    return "down"
            if char == "w":
                return "up"
            if char == "s":
                return "down"

    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x1b":
            nxt = sys.stdin.read(1)
            if nxt == "[":
                final = sys.stdin.read(1)
                if final == "A":
                    return "up"
                if final == "B":
                    return "down"
            return "escape"
        if char in {"\r", "\n"}:
            return "enter"
        if char == "w":
            return "up"
        if char == "s":
            return "down"
        return "other"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _prompt_text(label: str, current: str) -> str | None:
    _clear_screen()
    print(_color_text(label, ANSI_CYAN))
    print(f"Current: {current}")
    print("Enter new value and press Enter.")
    print("Leave blank to keep existing value. Type '-' to clear.")
    entered = input("> ").strip()
    if not entered:
        return None
    if entered == "-":
        return ""
    return entered


def _prompt_secret(label: str, current: str | None) -> str | None:
    _clear_screen()
    print(_color_text(label, ANSI_CYAN))
    print(f"Current: {_masked_value(current)}")
    print("Enter new secret and press Enter.")
    print("Leave blank to keep existing value. Type '-' to clear.")
    entered = getpass.getpass("> ").strip()
    if not entered:
        return None
    if entered == "-":
        return ""
    return entered


def _pause_message(message: str) -> None:
    print(message)
    input("")


def cmd_test(
    config_path: Path,
    prompt: str | None,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
) -> int:
    settings = load_settings(config_path)
    settings.providers = _merge_provider_templates(settings.providers)

    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        prompt_text = input("Enter prompt to test route: ").strip()
    if not prompt_text:
        print("No prompt provided.")
        return 1

    service = RouterExecutionService(settings=settings)
    messages = [{"role": "user", "content": prompt_text}]

    try:
        result = asyncio.run(
            service.run_completion(
                prompt_text=prompt_text,
                messages=messages,
                client_model_hint=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Test failed: {exc}")
        return 1

    print("LLM Router Test Result")
    print("----------------------")
    print(f"Provider: {result.provider}")
    print(f"Endpoint: {result.endpoint_id}")
    print(f"Model: {result.model}")
    if result.failures:
        print("Fallbacks:")
        for failure in result.failures:
            print(f"- {failure.route_provider}/{failure.model} -> {failure.reason}")
    print("Reply:")
    print(result.content)
    return 0


def cmd_status(config_path: Path) -> int:
    settings = load_settings(config_path)
    settings.providers = _merge_provider_templates(settings.providers)
    print(f"Config: {config_path}")
    print(f"Server: http://{settings.runtime.host}:{settings.runtime.port}")
    print(f"Client key required: {settings.runtime.require_client_api_key}")
    print(f"Router config path: {settings.runtime.router_config_path}")
    print("Providers:")
    for provider in settings.providers:
        key_ok = bool(provider_api_key(provider)) if provider_requires_key(provider) else True
        print(
            f"- {provider.id}: enabled={provider.enabled}, kind={provider.kind}, "
            f"base_url={provider.base_url}, model={provider.default_model or '-'}, "
            f"key={'set' if key_ok else 'missing'}"
        )
    return 0


def cmd_serve(config_path: Path, host: str | None, port: int | None) -> int:
    settings = load_settings(config_path)
    settings.providers = _merge_provider_templates(settings.providers)
    if host:
        settings.runtime.host = host
    if port:
        settings.runtime.port = int(port)

    app = build_app(settings)
    uvicorn.run(app, host=settings.runtime.host, port=settings.runtime.port, log_level="info")
    return 0


def _prompt_custom_provider() -> ProviderEndpoint:
    print("\nCustom Provider")
    provider_id = input("Provider id (e.g. mistral): ").strip().lower()
    kind = input("Kind [openai_compat/anthropic/google]: ").strip().lower() or "openai_compat"
    base_url = input("Base URL: ").strip()
    default_model = input("Default model: ").strip() or None
    api_key_env = input("API key env var (optional): ").strip() or None
    api_key = getpass.getpass("API key/token (hidden, optional): ").strip() or None

    return ProviderEndpoint(
        id=provider_id,
        kind=kind,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        default_model=default_model,
        enabled=True,
    )


def _default_auth_url(provider_id: str) -> str | None:
    table = {
        "github_models": "https://github.com/settings/tokens",
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/settings/keys",
        "google": "https://aistudio.google.com/apikey",
        "openrouter": "https://openrouter.ai/keys",
    }
    return table.get(provider_id)


def cmd_doctor(config_path: Path, probe_network: bool = False) -> int:
    settings = load_settings(config_path)
    settings.providers = _merge_provider_templates(settings.providers)

    issues = 0
    print("LLM Router Doctor")
    print("-----------------")
    print(f"Config: {config_path}")

    router_cfg = Path(settings.runtime.router_config_path)
    if router_cfg.exists():
        print(f"OK   router config exists: {router_cfg}")
    else:
        issues += 1
        print(f"FAIL router config missing: {router_cfg}")

    if settings.runtime.require_client_api_key and not str(settings.runtime.client_api_key or "").strip():
        issues += 1
        print("FAIL inbound API key is required but not set")
    else:
        print("OK   inbound API key policy is valid")

    for provider in settings.providers:
        status = "OK"
        message = "ready"
        if not provider.enabled:
            status = "SKIP"
            message = "disabled"
        elif not provider.base_url:
            status = "FAIL"
            message = "missing base_url"
            issues += 1
        elif provider_requires_key(provider) and not provider_api_key(provider):
            status = "WARN"
            message = "missing API key/token"

        print(f"{status:<4} {provider.id:<14} {message}")

        if probe_network and provider.enabled and provider.base_url:
            try:
                with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                    response = client.get(provider.base_url)
                print(f"     probe HTTP {response.status_code} {provider.base_url}")
            except Exception as exc:  # noqa: BLE001
                print(f"     probe failed: {exc}")

    if issues:
        print(f"\nDoctor found {issues} blocking issue(s).")
        return 1

    print("\nDoctor checks passed.")
    return 0


def cmd_auth(config_path: Path, provider: str) -> int:
    settings = load_settings(config_path)
    settings.providers = _merge_provider_templates(settings.providers)
    provider_key = str(provider or "all").strip().lower()

    opened = 0
    for endpoint in settings.providers:
        if provider_key != "all" and endpoint.id.lower() != provider_key:
            continue

        auth_url = endpoint.auth_url or _default_auth_url(endpoint.id)
        if not auth_url:
            print(f"No auth URL configured for {endpoint.id}")
            continue

        webbrowser.open(auth_url)
        opened += 1
        print(f"Opened auth page for {endpoint.id}: {auth_url}")

    if opened == 0:
        print("No auth pages opened.")
        return 1
    return 0


def _merge_provider_templates(existing: list[ProviderEndpoint]) -> list[ProviderEndpoint]:
    by_id = {p.id: p for p in existing if p.id}
    for template in DEFAULT_PROVIDER_TEMPLATES:
        if template.id not in by_id:
            by_id[template.id] = ProviderEndpoint(**asdict(template))
            continue

        current = by_id[template.id]
        if not current.base_url:
            current.base_url = template.base_url
        if not current.api_key_env:
            current.api_key_env = template.api_key_env
        if not current.default_model:
            current.default_model = template.default_model
        if not current.auth_url:
            current.auth_url = template.auth_url
        if not current.kind:
            current.kind = template.kind
    return [by_id[key] for key in sorted(by_id.keys())]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"setup", "settup"}:
        return cmd_setup(args.config, quick=getattr(args, "quick", False))
    if args.command == "status":
        return cmd_status(args.config)
    if args.command == "serve":
        return cmd_serve(args.config, host=args.host, port=args.port)
    if args.command == "doctor":
        return cmd_doctor(args.config, probe_network=bool(args.probe_network))
    if args.command == "auth":
        return cmd_auth(args.config, provider=args.provider)
    if args.command == "test":
        return cmd_test(
            args.config,
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
