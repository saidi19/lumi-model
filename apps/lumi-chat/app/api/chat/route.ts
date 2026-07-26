import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

type ChatMessage = {
  content: string;
  role: "assistant" | "user";
};

const MAX_MESSAGES = 64;
const MAX_MESSAGE_CHARACTERS = 65_536;

function errorResponse(message: string, status = 400) {
  return Response.json(
    {
      error: {
        code: "invalid_request_error",
        message,
        type: "invalid_request_error",
      },
    },
    { status },
  );
}

function parseMessages(value: unknown): ChatMessage[] {
  if (!value || typeof value !== "object") throw new Error("Body must be a JSON object.");
  const messages = (value as { messages?: unknown }).messages;
  if (!Array.isArray(messages) || messages.length < 1 || messages.length > MAX_MESSAGES) {
    throw new Error(`Messages must contain between 1 and ${MAX_MESSAGES} turns.`);
  }
  return messages.map((message, index) => {
    if (!message || typeof message !== "object") {
      throw new Error(`Message ${index} must be an object.`);
    }
    const { content, role } = message as { content?: unknown; role?: unknown };
    if (role !== "assistant" && role !== "user") {
      throw new Error(`Message ${index} has an unsupported role.`);
    }
    if (
      typeof content !== "string" ||
      !content.trim() ||
      content.length > MAX_MESSAGE_CHARACTERS
    ) {
      throw new Error(`Message ${index} has invalid content.`);
    }
    return { content, role };
  });
}

function frame(value: unknown): Uint8Array {
  return new TextEncoder().encode(`data: ${JSON.stringify(value)}\n\n`);
}

function mockStream(messages: ChatMessage[]): Response {
  const last = messages.at(-1)?.content ?? "";
  const bengali = /[\u0980-\u09ff]/u.test(last);
  const text = bengali
    ? "Lumi L1 এখনো 80B base training শেষ করছে। এই staging reply শুধু chat history, streaming UI এবং API flow পরীক্ষা করছে—এটি final model quality নয়।"
    : "Lumi L1 is still completing its 80B base training. This staging reply tests chat history, streaming UI, and the API flow only—it is not final model output.";
  const pieces = text.match(/\S+\s*/g) ?? [text];
  const completionId = `chatcmpl-staging-${crypto.randomUUID()}`;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      controller.enqueue(
        frame({
          choices: [
            {
              delta: { content: "", role: "assistant" },
              finish_reason: null,
              index: 0,
            },
          ],
          id: completionId,
          model: "lumi-4b-instruct-staging",
          object: "chat.completion.chunk",
        }),
      );
      for (const piece of pieces) {
        controller.enqueue(
          frame({
            choices: [
              {
                delta: { content: piece },
                finish_reason: null,
                index: 0,
              },
            ],
            id: completionId,
            model: "lumi-4b-instruct-staging",
            object: "chat.completion.chunk",
          }),
        );
        await new Promise((resolve) => setTimeout(resolve, 24));
      }
      controller.enqueue(
        frame({
          choices: [{ delta: {}, finish_reason: "stop", index: 0 }],
          id: completionId,
          model: "lumi-4b-instruct-staging",
          object: "chat.completion.chunk",
        }),
      );
      controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
      controller.close();
    },
  });
  return new Response(stream, {
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "text/event-stream; charset=utf-8",
      "X-Lumi-Mode": "staging",
    },
  });
}

function completionEndpoint(baseUrl: string): string {
  const normalized = baseUrl.replace(/\/+$/, "");
  const url = new URL(normalized);
  const local =
    url.hostname === "127.0.0.1" || url.hostname === "::1" || url.hostname === "localhost";
  if (url.protocol !== "https:" && !(process.env.NODE_ENV !== "production" && local)) {
    throw new Error("Model base URL must use HTTPS outside local development.");
  }
  return normalized.endsWith("/v1")
    ? `${normalized}/chat/completions`
    : `${normalized}/v1/chat/completions`;
}

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return errorResponse("Body must be valid JSON.");
  }

  let messages: ChatMessage[];
  try {
    messages = parseMessages(body);
  } catch (error) {
    return errorResponse(error instanceof Error ? error.message : "Invalid messages.");
  }

  const baseUrl = process.env.LUMI_MODEL_BASE_URL?.trim();
  const servingContractSha = process.env.LUMI_SERVING_CONTRACT_SHA256?.trim() ?? "";
  if (!baseUrl || !/^[0-9a-f]{64}$/.test(servingContractSha)) {
    return mockStream(messages);
  }

  let endpoint: string;
  try {
    endpoint = completionEndpoint(baseUrl);
  } catch (error) {
    return errorResponse(
      error instanceof Error ? error.message : "Invalid model endpoint.",
      503,
    );
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Lumi-Serving-Contract": servingContractSha,
  };
  const apiKey = process.env.LUMI_MODEL_API_KEY?.trim();
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  let upstream: Response;
  try {
    upstream = await fetch(endpoint, {
      body: JSON.stringify({
        max_completion_tokens: 1024,
        messages,
        model: process.env.LUMI_MODEL_ID?.trim() || "lumi-4b-instruct",
        stream: true,
        stream_options: { include_usage: true },
        temperature: 0.7,
        top_p: 0.95,
      }),
      headers,
      method: "POST",
      signal: request.signal,
    });
  } catch {
    return errorResponse("The model runtime is unavailable.", 503);
  }
  if (!upstream.ok || !upstream.body) {
    return errorResponse(`The model runtime returned HTTP ${upstream.status}.`, 502);
  }
  return new Response(upstream.body, {
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": upstream.headers.get("Content-Type") ?? "text/event-stream",
      "X-Lumi-Mode": "connected",
    },
    status: upstream.status,
  });
}
