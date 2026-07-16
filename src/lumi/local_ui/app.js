"use strict";

const MODEL_ID = "lumi-150m-local";
const STORAGE_KEY = "lumi-local-chat-v0.1";

const state = {
  messages: [],
  running: false,
  controller: null,
  modelId: MODEL_ID,
};

const elements = {
  composer: document.querySelector("#composer"),
  conversation: document.querySelector("#conversation"),
  maxTokens: document.querySelector("#max-tokens"),
  messages: document.querySelector("#messages"),
  mobileNewChat: document.querySelector("#mobile-new-chat"),
  newChat: document.querySelector("#new-chat"),
  prompt: document.querySelector("#prompt"),
  sendButton: document.querySelector("#send-button"),
  statusDot: document.querySelector("#status-dot"),
  runtimeStatus: document.querySelector("#runtime-status"),
  temperature: document.querySelector("#temperature"),
  welcome: document.querySelector("#welcome"),
};

function saveConversation() {
  const serializable = state.messages
    .filter((message) => !message.pending && !message.error)
    .slice(-40);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
}

function loadConversation() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(value)) {
      state.messages = value.filter(
        (message) =>
          message &&
          ["assistant", "user"].includes(message.role) &&
          typeof message.content === "string",
      );
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function messageElement(message, index) {
  const article = document.createElement("article");
  article.className = `message ${message.role}${message.error ? " error" : ""}`;
  article.dataset.index = String(index);

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = message.role === "assistant" ? "L" : "YOU";

  const body = document.createElement("div");
  body.className = "message-body";
  const role = document.createElement("p");
  role.className = "message-role";
  role.textContent = message.role === "assistant" ? "Lumi" : "You";
  const content = document.createElement("p");
  content.className = "message-content";
  content.textContent = message.content;

  body.append(role, content);
  if (message.meta) {
    const meta = document.createElement("p");
    meta.className = "message-meta";
    meta.textContent = message.meta;
    body.append(meta);
  }
  article.append(avatar, body);
  return article;
}

function render() {
  elements.welcome.classList.toggle("hidden", state.messages.length > 0);
  elements.messages.replaceChildren(
    ...state.messages.map((message, index) => messageElement(message, index)),
  );
  elements.sendButton.disabled = !state.running && !elements.prompt.value.trim();
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function resizePrompt() {
  elements.prompt.style.height = "auto";
  elements.prompt.style.height = `${Math.min(elements.prompt.scrollHeight, 150)}px`;
  elements.sendButton.disabled = !state.running && !elements.prompt.value.trim();
}

function setRunning(running) {
  state.running = running;
  elements.prompt.disabled = running;
  elements.temperature.disabled = running;
  elements.maxTokens.disabled = running;
  elements.sendButton.disabled = !running && !elements.prompt.value.trim();
  elements.sendButton.querySelector(".send-label").textContent = running ? "Stop" : "Send";
  elements.sendButton.querySelector(".send-arrow").textContent = running ? "■" : "↑";
}

function appendMessage(role, content, extra = {}) {
  state.messages.push({ role, content, ...extra });
  render();
  return state.messages.length - 1;
}

function completionMessages() {
  return state.messages
    .filter((message) => !message.pending && !message.error)
    .map(({ role, content }) => ({ role, content }));
}

async function readEventStream(response, onChunk) {
  if (!response.body) {
    throw new Error("Streaming response body is unavailable.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        const payload = JSON.parse(data);
        if (payload.error) throw new Error(payload.error.message || "Generation failed.");
        onChunk(payload);
      }
    }
    if (done) return;
  }
}

async function submitPrompt() {
  const prompt = elements.prompt.value.trim();
  if (!prompt || state.running) return;

  appendMessage("user", prompt);
  const requestMessages = completionMessages();
  const assistantIndex = appendMessage("assistant", "", { pending: true });
  elements.prompt.value = "";
  resizePrompt();
  setRunning(true);
  state.controller = new AbortController();

  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.modelId,
        messages: requestMessages,
        max_tokens: Number(elements.maxTokens.value),
        temperature: Number(elements.temperature.value),
        top_p: 1,
        stream: true,
        stream_options: { include_usage: true },
      }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error?.message || `Request failed (${response.status}).`);
    }

    let usage = null;
    let finishReason = null;
    await readEventStream(response, (payload) => {
      if (payload.usage) usage = payload.usage;
      const choice = payload.choices?.[0];
      if (!choice) return;
      const delta = choice.delta?.content;
      const assistant = state.messages[assistantIndex];
      if (!assistant) return;
      if (delta) assistant.content += delta;
      if (choice.finish_reason) finishReason = choice.finish_reason;
      render();
    });
    const assistant = state.messages[assistantIndex];
    if (assistant) {
      assistant.pending = false;
      assistant.meta = usage
        ? `${usage.completion_tokens} generated tokens · ${finishReason || "complete"}`
        : finishReason || "complete";
    }
    saveConversation();
  } catch (error) {
    const assistant = state.messages[assistantIndex];
    if (assistant) {
      assistant.pending = false;
      if (error.name === "AbortError") {
        assistant.meta = "Stopped";
        if (!assistant.content) assistant.content = "Generation stopped.";
      } else {
        assistant.error = true;
        assistant.content = error.message || "Lumi could not generate a response.";
      }
    }
  } finally {
    state.controller = null;
    setRunning(false);
    saveConversation();
    render();
    elements.prompt.focus();
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok || payload.status !== "ok") throw new Error("offline");
    state.modelId = payload.model || MODEL_ID;
    elements.statusDot.classList.add("online");
    elements.runtimeStatus.textContent = payload.model_loaded ? "Model ready" : "Server ready";
  } catch {
    elements.statusDot.classList.remove("online");
    elements.runtimeStatus.textContent = "Runtime unavailable";
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.running && state.controller) {
    state.controller.abort();
    return;
  }
  submitPrompt();
});

elements.prompt.addEventListener("input", resizePrompt);
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

function clearConversation() {
  if (state.controller) state.controller.abort();
  state.messages = [];
  localStorage.removeItem(STORAGE_KEY);
  render();
  elements.prompt.focus();
}

elements.newChat.addEventListener("click", clearConversation);
elements.mobileNewChat.addEventListener("click", clearConversation);

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.prompt.value = button.dataset.prompt || "";
    resizePrompt();
    elements.prompt.focus();
  });
});

loadConversation();
resizePrompt();
render();
checkHealth();
