# Structured-output pressure demo

This three-part Muse-Glimmer demo shows what constraint pressure measures:

1. `01_unprompted` asks an ordinary question without mentioning JSON. The JSON
   schema immediately steers the model.
2. `02_order_mismatch` asks for the schema's fields in reverse order. The
   llguidance grammar compiled from the schema emits its declared first field,
   while the forensic rows expose the field the model preferred.
3. `03_order_match` asks for the same fields in schema order. No preferred token
   is masked, and the residual pressure is low.

The public model is `meta-models/Muse-Glimmer-30B` at revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. Human-readable rendered contexts
are in `prompts/`; the assistant recipient prefix is prefilled so the demo
measures the JSON body rather than Muse's message envelope.

The checked-in capture produced:

| Case | Masked argmax tokens | Max pressure | Key evidence |
| --- | ---: | ---: | --- |
| No JSON instruction | 4 | 10.93 nats | The first token is already constrained |
| Reverse field order | 1 | 3.26 nats | Grammar emits `city`; model preferred `temperature` |
| Matching field order | 0 | 0.157 nats | Every emitted token is the model's argmax |

The runner recursively discovers JSON files under `requests/` instead of
enumerating demo cases. For each single-request payload, it preserves other
sampling parameters, forces non-streaming forensic metrics, and mirrors the
relative path into:

- `submitted_requests/`: the exact bytes posted to `/generate`
- `responses/`: the complete response JSON, newline-normalized for the repository
- `token_labels/`: token text needed by the offline TUI

Optional `*.expect.json` files next to requests describe qualitative summary
bounds and token evidence. Tests check them, while the replay tool ignores them
as input requests. The checked-in examples verify the three claims above
without naming cases in Python.

This layout also accepts a directory of requests dumped from an eval. One bad
request does not prevent the rest from being captured, and the command exits
nonzero after finishing if any request failed. Other sampling parameters are
preserved; only `stream` and `sampling_params.constraint_metrics` are replaced.

Captured requests may contain private prompts. Use output directories outside
this checkout for private eval data, and review files before committing them.

Start a compatible server (the deployment cap must cover the requested
forensic `top_k`):

```bash
SGLANG_PLUGINS=constraint_metrics \
SGLANG_CONSTRAINT_METRICS_MAX_TOP_K=8 \
sglang serve \
  --model-path meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --grammar-backend llguidance \
  --max-running-requests 8
```

Then replay every request:

```bash
python examples/structured_output/run_demo.py \
  --url http://localhost:30000 \
  --model meta-models/Muse-Glimmer-30B
```

For a different eval model, pass its identity with `--model`; the runner checks
`/model_info` before posting tokenizer-specific payloads.

Open a saved case completely offline:

```bash
sglang-constraint-metrics-view \
  --response examples/structured_output/responses/02_order_mismatch.json \
  --token-labels examples/structured_output/token_labels/02_order_mismatch.json
```
