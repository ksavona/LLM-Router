# LLM Router

LLM Router is an OpenAI-compatible provider gateway that sits between client apps and multiple LLM vendors.

It accepts standard API requests, routes each prompt to the best model for cost/capability, retries across providers when failures occur, and returns responses with transparent routing metadata.

## Key Features

- OpenAI-compatible endpoints for broad app compatibility.
- Multi-provider support (OpenAI, Anthropic, Google, OpenRouter, GitHub Models, xAI, Mistral, Groq, Together, Cohere, Azure OpenAI).
- Automatic routing and fallback with failure reason tracking.
- Streaming support (`stream=true`) for chat and completions endpoints.
- Terminal-first setup and diagnostics.
- Optional Hermes compatibility plugin for existing Hermes deployments.

## Quick Start

```bash
git clone https://github.com/ksavona/LLM-Router.git && cd LLM-Router && python -m pip install -e .
```

Default endpoint:

```text
http://127.0.0.1:8142/v1/chat/completions
```

## CLI Commands

```bash
llm-router setup
llm-router status
llm-router doctor
llm-router auth --provider all
llm-router serve
```

## API Compatibility

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`

Inbound auth headers accepted:

- `Authorization: Bearer <key>`
- `x-api-key: <key>`

## Hermes Compatibility

This repository still includes Hermes compatibility modules and metadata so existing Hermes users can continue running routed workflows during migration.

Compatibility surfaces are intentionally isolated and can be removed later if you decide to ship only provider-mode behavior.

## Development

```bash
python -m pip install -e .[dev]
pytest
ruff check .
```

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Installation](docs/INSTALLATION.md)

## License

LLM Router is source-available.

Personal, educational, research, and non-commercial use is allowed under the PolyForm Noncommercial License 1.0.0.

Commercial use requires a paid commercial license from the author.

Commercial use includes use by companies, consultants, agencies, SaaS providers, managed service providers, or any revenue-generating product or service.

For commercial licensing, contact Kristian Savona Ventura.

See [LICENSE](LICENSE) for details.
