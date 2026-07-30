# Prompt Injection Security

Research and tooling for defending LLM applications against prompt injection.

## [`llm-guard-api/`](llm-guard-api/)

A layered guard that sits between user input and an LLM and decides whether each
prompt should be allowed, sanitized, or blocked:

1. Regex heuristics — bounded pattern matching, sub-millisecond
2. Intent classifier — TF-IDF + logistic regression (CPU) or fine-tuned DeBERTa
3. Decision engine — auditable rules producing ALLOW / SANITIZE / BLOCK
4. Sanitizer — strips meta-instructions and re-wraps with explicit boundaries
5. Gemini client — safety settings on, retries, graceful filtering

Usable as a library, a CLI, or a FastAPI service. Works offline out of the box:
the default classifier trains itself from the bundled corpus in about ten
seconds, with no GPU, model download, or API key required.

```bash
cd llm-guard-api
pip install -r requirements.txt
python app.py --no-llm "Ignore all previous instructions"
```

See [`llm-guard-api/README.md`](llm-guard-api/README.md) for setup, the HTTP API,
configuration, and design notes.
