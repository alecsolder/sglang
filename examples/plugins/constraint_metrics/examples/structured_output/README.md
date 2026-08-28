# Structured-output pressure demo

This three-part Muse-Glimmer demo shows what constraint pressure measures:

1. `01_unprompted` asks an ordinary question without mentioning JSON. The JSON
   schema immediately steers the model.
2. `02_order_mismatch` asks for the schema's fields in reverse order. The
   schema emits its alphabetically declared first field while the forensic
   rows expose the field the model preferred.
3. `03_order_match` asks for the same fields in schema order. Model and grammar
   agree, so pressure should be near zero.

The public model is `meta-models/Muse-Glimmer-30B` at revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. Human-readable rendered contexts
are in `prompts/`; the assistant recipient prefix is prefilled so the demo
measures the JSON body rather than Muse's message envelope.

The runner recursively discovers JSON files under `requests/` instead of
enumerating demo cases. For each single-request payload, it preserves the
sampling parameters, forces non-streaming forensic metrics, and mirrors the
relative path into:

- `submitted_requests/`: the exact bytes posted to `/generate`
- `responses/`: the complete HTTP response body
- `token_labels/`: token text needed by the offline TUI

This layout also accepts a directory of requests dumped from an eval. One bad
request does not prevent the rest from being captured, and the command exits
nonzero after finishing if any request failed.

Replay every request against an existing compatible server:

```bash
python examples/structured_output/run_demo.py \
  --url http://localhost:30000 \
  --model meta-models/Muse-Glimmer-30B
```

Omit `--url` to let the script start and stop the pinned public model itself.
For a different eval model, pass its identity with `--model`; the runner checks
`/model_info` before posting tokenizer-specific payloads.

Open a saved case completely offline:

```bash
sglang-constraint-metrics-view \
  --response examples/structured_output/responses/02_order_mismatch.json \
  --token-labels examples/structured_output/token_labels/02_order_mismatch.json
```
