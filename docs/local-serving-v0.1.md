# Lumi Local Serving Gate v0.1

## Decision

The sealed Lumi-150M 4-bit MLX artifact now runs behind a loopback-first,
OpenAI-compatible chat-completions API and a responsive local chat UI. A real
black-box run on the 16GB M5 Mac passes model loading, model discovery, the UI
and both static assets, non-streaming generation, SSE streaming, usage counts,
and terminal `[DONE]` handling.

Use this server and UI to validate product plumbing around Lumi. This gate does
not improve model quality. Lumi-150M remains a base next-token model with
150,625,536 parameters trained on 50,003,968 tokens. It has no learned chat
template, instruction SFT, preference optimization, safety post-training, tool
use, or retrieval. Repetition and wrong answers are therefore expected.

## Local interfaces

The standard launch serves all interfaces from one process:

| Interface | Route | v0.1 behavior |
| --- | --- | --- |
| Chat UI | `GET /` | responsive local-only conversation UI |
| Health | `GET /health` | server and lazy/preloaded model state |
| Model discovery | `GET /v1/models` | one Lumi model plus context/quantization metadata |
| Chat completion | `POST /v1/chat/completions` | OpenAI-shaped JSON or SSE response |
| UI assets | `GET /app.js`, `GET /styles.css` | packaged, same-origin static files |

The chat endpoint supports text-only `system`, `developer`, `user`, and
`assistant` messages; `temperature`, `top_p`, `seed`, `max_tokens` or
`max_completion_tokens`; streaming; and streaming usage. It intentionally
rejects unsupported tools, functions, audio, images, logprobs, stop sequences,
multiple choices, and non-text response formats with an OpenAI-shaped error.

Because this checkpoint is a base model, the server transparently serializes
roles as plain `System:`, `User:`, and `Assistant:` text. That is a plumbing
format, not a claim that the model learned a chat protocol.

## Safety and runtime behavior

- The default bind is `127.0.0.1`; a non-loopback bind is refused unless the
  operator explicitly passes `--allow-network`.
- The API accepts `application/json` only and sends no cross-origin permission.
- Static responses use a restrictive content-security policy, deny framing,
  disable camera/geolocation/microphone permissions, and are never cached.
- The model directory's `SHA256SUMS` is verified before MLX loads the artifact.
- Generation is serialized with a lock. Concurrent HTTP connections cannot run
  unsafe overlapping generations against the same MLX model instance.
- Requests are capped at 1 MiB and the prompt plus requested completion must fit
  the artifact's 2,048-token context window.
- UI history stays in browser `localStorage` on the device; the last 40 complete
  non-error messages are retained. New-conversation and stop controls abort an
  active browser request safely.

## Verified M5 result

The canonical gate used the sealed affine 4-bit artifact under
`lumi-data/local-inference-gate-v0.1/lumi-150m-4bit-mlx`.

| Check | Result |
| --- | ---: |
| Model loaded and checksum-verified | PASS |
| Model ID | `lumi-150m-local` |
| Quantization metadata | 4-bit, group size 64, affine |
| Context window | 2,048 tokens |
| HTML, JavaScript, and CSS assets | PASS |
| Non-streaming completion | PASS, 8 generated tokens |
| SSE role/text/finish/usage/`[DONE]` | PASS |
| Warm non-stream request elapsed time | 0.062 seconds |
| Warm stream request elapsed time | 0.044 seconds |

These elapsed times are short fixed-probe integration timings, not a standalone
throughput benchmark. The controlled MLX benchmark remains the authoritative
speed result in `docs/local-inference-gate-v0.1.md`.

The generated report remains outside Git at
`lumi-data/local-serving-gate-v0.1/server-report.json`. Its SHA-256 is
`9a7c1fdf9b171adf5863f836343ae58de5b0fdd5be504a4f07e8e88810b48aa7`.
The earlier pre-final UI report is preserved separately and is not canonical.

## Run locally

Install the Apple runtime and launch the sealed model:

```bash
python -m pip install -e '.[dev,apple]'

python scripts/serve_local.py \
  --model "$LUMI_DATA/local-inference-gate-v0.1/lumi-150m-4bit-mlx" \
  --host 127.0.0.1 \
  --port 8080
```

Open `http://127.0.0.1:8080` on the Mac. An OpenAI-style client can use
`http://127.0.0.1:8080/v1` as its base URL and `lumi-150m-local` as its model.
No API key is required because the default server is reachable only from the
same Mac.

Non-streaming example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lumi-150m-local",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

Streaming example:

```bash
curl --no-buffer http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lumi-150m-local",
    "messages": [{"role": "user", "content": "In Python, a list can"}],
    "temperature": 0,
    "max_tokens": 16,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Re-run the durable black-box gate against a running server with a new output
path; it refuses to overwrite an existing report:

```bash
python scripts/check_local_server.py \
  --base-url http://127.0.0.1:8080 \
  --output "$LUMI_DATA/local-serving-gate-v0.1/server-report-next.json"
```

## Next gate

The 1B architecture, held-out-safe 1.4B-token artifact, and local cloud
preflight are now sealed in [`scaling-pilot-1b-v0.1.md`](scaling-pilot-1b-v0.1.md)
and [`cloud-pilot-1b-v0.1.md`](cloud-pilot-1b-v0.1.md). No paid GPU starts until
an actually available provider target is recorded, the exact bundle is
transferred, and the remote CUDA/BF16/resource/hash preflight passes.
