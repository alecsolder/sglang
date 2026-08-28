# SGLang constraint metrics

An out-of-tree SGLang plugin for measuring how grammar constraints change a
model's token distribution. Metrics are request-scoped and support normal and
streaming responses.

The package uses SGLang's public plugin entry point and does not modify the
installed SGLang package. Installing it alone has no serving-path effect.

## Compatibility

| Sampling path | Required SGLang support |
|---|---|
| Normal decoding | Sampling-observer auxiliary outputs, available on current `main` |
| DFlash | The linear-speculation observer integration, pending upstream |

Other speculative and diffusion sampling paths are rejected during startup.
The plugin requires NVIDIA CUDA and explicitly supports only the `llguidance`
and `xgrammar` grammar backends. XGrammar reuses its packed-mask path; the
plugin does not add XGrammar batching optimizations.
The plugin does not support PD disaggregation, pipeline parallelism, Ray, the
Rust server, multiple tokenizer or detokenizer workers, reasoning parsers, or
strict thinking.

Install a compatible source build of SGLang first, then install this package in
the same environment:

```bash
python -m pip install -e '.[viewer]'
```

The package deliberately does not declare SGLang as a dependency, so installing
it cannot replace the source build being tested.

## Enable the plugin

Select the plugin and the supported grammar backend when launching SGLang:

```bash
SGLANG_PLUGINS=constraint_metrics sglang serve MODEL \
  --grammar-backend llguidance \
  --max-running-requests 32
```

Use `--grammar-backend xgrammar` to select XGrammar.

`SGLANG_PLUGINS=constraint_metrics` is the feature-level switch. Without it,
the installed package remains inactive. With it, startup verifies all required
hooks and rejects incompatible server configurations before model loading.

Individual requests opt in through `sampling_params.constraint_metrics`:

```json
{
  "text": "Return a JSON object",
  "sampling_params": {
    "max_new_tokens": 64,
    "json_schema": "{\"type\":\"object\"}",
    "constraint_metrics": {"mode": "summary"}
  }
}
```

Supported modes are:

- `summary`: aggregate metrics only.
- `triggered`: aggregates plus token rows that cross a trigger.
- `forensic`: aggregates plus every token row and optional top-k tables.

Triggered and forensic requests can set `stream_all_token_metrics: true` when
streaming. Forensic requests can set `top_k` up to 32. The process-wide GPU
workspace cap defaults to 8 and can be changed with
`SGLANG_CONSTRAINT_METRICS_MAX_TOP_K`; requests above that cap are rejected.

The terminal response adds `meta_info.constraint_metrics`:

```json
{
  "meta_info": {
    "constraint_metrics": {
      "status": "complete",
      "mode": "summary",
      "warnings": [],
      "summary": {"output_token_count": 12},
      "input_token_count": 7,
      "output_token_count": 12,
      "tokens": []
    }
  }
}
```

Streaming chunks can additionally contain
`meta_info.constraint_metrics_tokens`, aligned with the chunk's output tokens.
The terminal chunk still carries the complete aggregate response.

Treat an enabled endpoint as trusted: clients allowed to set arbitrary sampling
parameters can request extra GPU computation and forensic response retention.
Enforce request policy at the gateway before exposing it to untrusted callers.

## Terminal viewer

The `viewer` extra installs a terminal UI that can issue a forensic request or
open a saved response:

```bash
sglang-constraint-metrics-view \
  --url http://localhost:30000 \
  --request request.json \
  --top-k 8 \
  --save response.json

sglang-constraint-metrics-view --response response.json
```

The viewer starts on the peak-pressure token. Use arrows or `j`/`k` to move,
`n`/`p` to jump between rows at or above one nat of pressure, `s` for the full
summary, `Enter` or `i` for the full-token view, and `?` for metric definitions.

See [`examples/muse_glimmer`](examples/muse_glimmer) for a complete public-model
example using `meta-models/Muse-Glimmer-30B`. The
[`examples/structured_output`](examples/structured_output) walkthrough includes
three saved requests, full responses, and offline TUI fixtures.

## Development

Run from this package directory in a compatible SGLang environment:

```bash
ruff check .
isort --check-only .
black --check .
python -m pytest -q
python -m build
```

The kernel tests require an NVIDIA GPU. CPU-only environments skip them.
