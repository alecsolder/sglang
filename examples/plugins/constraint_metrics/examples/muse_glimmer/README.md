# Muse-Glimmer calibration example

This example applies a grammar that matches Muse-Glimmer's normal assistant
turn. A successful run should show near-zero constraint pressure and an
`argmax_masked_count` of zero, demonstrating that measurement does not steer
greedy decoding.

Launch the public model from the repository root after installing the plugin:

```bash
SGLANG_PLUGINS=constraint_metrics sglang serve meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --grammar-backend llguidance \
  --max-running-requests 32
```

From the plugin package directory, run the request and save its full response:

```bash
python examples/muse_glimmer/run_request.py --save response.json
sglang-constraint-metrics-view \
  --response response.json \
  --url http://localhost:30000
```

The runner renders one generic prompt with the model's public chat template,
inlines [`chat_turn.lark`](chat_turn.lark), requests forensic metrics, and fails
if capture is incomplete, emits warnings, masks the model's top choice, or
reports peak pressure at or above `1e-2` nats. The server and runner commands
pin the same public model revision because the grammar contains tokenizer IDs.

Exact token counts and pressure values may change with SGLang, model, and batch
composition. The expected invariant is the shape: pressure remains far below
one nat and the grammar does not mask an argmax.
