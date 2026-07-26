# Lumi chat product shell

This is the staging frontend and API proxy for Lumi L1. It is intentionally
useful before the final model exists:

- multiple local conversations with browser-only history;
- OpenAI-compatible server-sent-event parsing;
- an explicit mock stream when no model endpoint is configured;
- a release-readiness view for base, SFT, benchmark, safety, and serving gates;
- a server-only proxy contract for the eventual sealed inference endpoint.

It never claims the staging stream is model output. Public deployment stays
blocked by `configs/lumi-4b-serving-contract-v0.1.json`.

## Local development

```bash
npm ci
npm run dev
```

Copy `.env.example` to an ignored `.env.local` only when an authenticated model
server exists. The proxy stays in staging mode until both the model endpoint and
the exact `LUMI_SERVING_CONTRACT_SHA256` are present. Endpoint, API key, and
contract binding are read only by the server route and must never be exposed to
client JavaScript.

## Verification

```bash
npm run lint
npm test
```

The Sites/vinext deployment skeleton is present, but this branch does not create
or deploy a public site while the final 80B, SFT, benchmark, and safety evidence
is incomplete.
