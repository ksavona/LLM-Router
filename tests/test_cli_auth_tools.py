from __future__ import annotations

from types import SimpleNamespace

from llm_router import cli


def test_resolve_command_finds_windows_cmd_shim(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_which(name: str) -> str | None:
        calls.append(name)
        if name.lower() == "codex.cmd":
            return r"C:\Users\test\AppData\Roaming\npm\codex.CMD"
        return None

    monkeypatch.setattr(cli.os, "name", "nt", raising=False)
    monkeypatch.setattr(cli.shutil, "which", _fake_which)

    resolved = cli._resolve_command("codex")
    assert resolved == r"C:\Users\test\AppData\Roaming\npm\codex.CMD"
    assert calls[:2] == ["codex", "codex.cmd"]


def test_run_command_uses_resolved_executable(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli, "_resolve_command", lambda command: r"C:\shim\codex.CMD" if command == "codex" else None)
    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    result = cli._run_command(["codex", "login", "status"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert captured["args"] == [r"C:\shim\codex.CMD", "login", "status"]


def test_codex_login_active_true_on_zero_exit(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_command_exists", lambda command: command == "codex")
    monkeypatch.setattr(
        cli,
        "_run_command",
        lambda args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="Logged in using ChatGPT"),
    )

    assert cli._codex_login_active() is True


def test_codex_login_active_true_when_output_says_logged_in(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_command_exists", lambda command: command == "codex")
    monkeypatch.setattr(
        cli,
        "_run_command",
        lambda args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="already logged in"),
    )

    assert cli._codex_login_active() is True


def test_codex_login_active_false_when_not_logged_in(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_command_exists", lambda command: command == "codex")
    monkeypatch.setattr(
        cli,
        "_run_command",
        lambda args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="not authenticated"),
    )

    assert cli._codex_login_active() is False