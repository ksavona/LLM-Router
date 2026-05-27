from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
import os
from pathlib import Path
import webbrowser

import httpx
import uvicorn

from .api import build_app
from .settings import (
    DEFAULT_PROVIDER_TEMPLATES,
    ProviderEndpoint,
    default_settings_path,
    load_settings,
    provider_api_key,
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


def _interactive_setup_menu(settings) -> bool:
    while True:
        runtime_ready = _runtime_ready(settings)
        providers_ready, providers_total = _providers_ready_counts(settings.providers)

        options = [
            f"Runtime Settings      [{_status_tag(runtime_ready)}]",
            f"Provider Settings     [{_color_text(f'{providers_ready}/{providers_total} READY', ANSI_GREEN if providers_ready == providers_total else ANSI_YELLOW)}]",
            "Add Custom Provider",
            _color_text("Save and Exit", ANSI_GREEN),
            _color_text("Exit Without Saving", ANSI_RED),
        ]
        choice = _arrow_menu(
            title="LLM ROUTER // SETUP TERMINAL",
            subtitle="Hack mode active - arrow keys navigate, Enter selects",
            options=options,
        )
        if choice is None or choice == 4:
            return False
        if choice == 0:
            _runtime_settings_menu(settings)
        elif choice == 1:
            _provider_settings_menu(settings)
        elif choice == 2:
            settings.providers.append(_prompt_custom_provider())
        elif choice == 3:
            return True


def _runtime_settings_menu(settings) -> None:
    while True:
        client_key_set = bool(str(settings.runtime.client_api_key or "").strip())
        require_key = bool(settings.runtime.require_client_api_key)
        key_ok = (not require_key) or client_key_set
        options = [
            f"Host                 {settings.runtime.host} {_status_marker(bool(str(settings.runtime.host).strip()))}",
            f"Port                 {settings.runtime.port} {_status_marker(settings.runtime.port > 0)}",
            f"Require Client Key   {_toggle_text(require_key)}",
            f"Client API Key       {_masked_value(settings.runtime.client_api_key)} {_status_marker(key_ok)}",
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
        options.append("Back")

        choice = _arrow_menu(
            title="PROVIDER SETTINGS",
            subtitle="Pick a provider to edit. Ready providers are green.",
            options=options,
        )
        if choice is None or choice == len(options) - 1:
            return
        _edit_provider_menu(providers[choice])


def _edit_provider_menu(provider: ProviderEndpoint) -> None:
    while True:
        key_set = bool(provider_api_key(provider))
        options = [
            f"Enabled              {_toggle_text(provider.enabled)}",
            f"Base URL             {provider.base_url or '-'} {_status_marker(bool(str(provider.base_url).strip()))}",
            f"Default Model        {provider.default_model or '-'} {_status_marker(bool(str(provider.default_model or '').strip()))}",
            f"API Key Env Var      {provider.api_key_env or '-'} {_status_marker(bool(str(provider.api_key_env or '').strip()))}",
            f"API Key/Token        {_masked_value(provider.api_key)} {_status_marker(key_set)}",
            "Open Auth Page",
            "Back",
        ]
        choice = _arrow_menu(
            title=f"PROVIDER // {provider.id.upper()}",
            subtitle=f"kind={provider.kind}",
            options=options,
        )
        if choice is None or choice == 6:
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
            auth_url = provider.auth_url or _default_auth_url(provider.id)
            if auth_url:
                webbrowser.open(auth_url)
                _pause_message(f"Opened auth page for {provider.id}. Press Enter to continue.")
            else:
                _pause_message(f"No auth URL configured for {provider.id}. Press Enter to continue.")


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
        print()
        for i, option in enumerate(options):
            if i == index:
                print(_color_text(f">> {option}", ANSI_CYAN))
            else:
                print(f"   {option}")
        print()
        print(_color_text("UP/DOWN = navigate | ENTER = select | ESC = back", ANSI_YELLOW))

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


def cmd_status(config_path: Path) -> int:
    settings = load_settings(config_path)
    settings.providers = _merge_provider_templates(settings.providers)
    print(f"Config: {config_path}")
    print(f"Server: http://{settings.runtime.host}:{settings.runtime.port}")
    print(f"Client key required: {settings.runtime.require_client_api_key}")
    print(f"Router config path: {settings.runtime.router_config_path}")
    print("Providers:")
    for provider in settings.providers:
        key_ok = bool(provider_api_key(provider))
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
        elif not provider_api_key(provider):
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
