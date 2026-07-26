import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

async function requestWorker(path = "/", init = {}) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${path}`, init),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      passThroughOnException() {},
      waitUntil() {},
    },
  );
}

test("server-renders the Lumi staging workspace", async () => {
  const response = await requestWorker("/", {
    headers: { accept: "text/html" },
  });
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Lumi — AI workspace<\/title>/i);
  assert.match(html, /Build beside the model/);
  assert.match(html, /This is not the final model/);
  assert.match(html, /Readiness/);
  assert.match(html, /Private staging/);
  assert.doesNotMatch(html, /Your site is taking shape|SkeletonPreview/);
});

test("keeps model credentials server-side and the serving state explicit", async () => {
  const [client, route, status, envExample] = await Promise.all([
    readFile(new URL("app/components/LumiStudio.tsx", root), "utf8"),
    readFile(new URL("app/api/chat/route.ts", root), "utf8"),
    readFile(new URL("app/api/status/route.ts", root), "utf8"),
    readFile(new URL(".env.example", root), "utf8"),
  ]);
  assert.match(client, /localStorage/);
  assert.match(client, /readStream/);
  assert.doesNotMatch(client, /LUMI_MODEL_API_KEY|LUMI_MODEL_BASE_URL/);
  assert.match(route, /process\.env\.LUMI_MODEL_API_KEY/);
  assert.match(route, /text\/event-stream/);
  assert.match(status, /final_80b_checkpoint/);
  assert.match(envExample, /LUMI_MODEL_BASE_URL=/);
  assert.match(envExample, /LUMI_SERVING_CONTRACT_SHA256=/);
  assert.doesNotMatch(envExample, /Bearer\s+\S+|AKIA[A-Z0-9]+/);
});

test("serves explicit staging status and an OpenAI-shaped stream", async () => {
  const statusResponse = await requestWorker("/api/status");
  assert.equal(statusResponse.status, 200);
  assert.deepEqual(await statusResponse.json(), {
    blockers: [
      "final_80b_checkpoint",
      "sft_checkpoint",
      "benchmark_report",
      "safety_report",
    ],
    mode: "staging",
    model: "lumi-4b-instruct",
    ready: false,
  });

  const chatResponse = await requestWorker("/api/chat", {
    body: JSON.stringify({
      messages: [{ content: "Hello Lumi", role: "user" }],
      stream: true,
    }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  assert.equal(chatResponse.status, 200);
  assert.match(chatResponse.headers.get("content-type") ?? "", /text\/event-stream/);
  assert.equal(chatResponse.headers.get("x-lumi-mode"), "staging");
  const body = await chatResponse.text();
  assert.match(body, /lumi-4b-instruct-staging/);
  const streamedText = [...body.matchAll(/^data: (\{.*\})$/gm)]
    .map((match) => JSON.parse(match[1]).choices?.[0]?.delta?.content ?? "")
    .join("");
  assert.match(streamedText, /not final model output/);
  assert.match(body, /data: \[DONE\]/);
});
