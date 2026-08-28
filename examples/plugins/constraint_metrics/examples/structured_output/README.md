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
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. Each file under `requests/` is the
exact `/generate` payload. The corresponding file under `responses/` is the
complete response body, without reducing its text, token IDs, metadata, or
constraint metrics. `token_labels/` is derived separately so the TUI remains
useful without a running tokenizer. Human-readable rendered contexts are in
`prompts/`; the assistant recipient prefix is prefilled so the demo measures
the JSON body rather than Muse's message envelope.

Replay every request against an existing compatible server:

```bash
python examples/structured_output/run_demo.py --url http://localhost:30000
```

Omit `--url` to let the script start and stop the pinned public model itself.
The runner discovers every JSON file under `requests/`, preserves relative
paths, and posts each payload unchanged, so the same command can inspect a
larger directory of captured eval requests.

Open a saved case completely offline:

```bash
sglang-constraint-metrics-view \
  --response examples/structured_output/responses/02_order_mismatch.json \
  --token-labels examples/structured_output/token_labels/02_order_mismatch.json
```
