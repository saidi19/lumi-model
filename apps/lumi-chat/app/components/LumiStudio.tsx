"use client";

import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type Role = "assistant" | "user";
type View = "chat" | "readiness";

type Message = {
  content: string;
  id: string;
  pending?: boolean;
  role: Role;
};

type Conversation = {
  createdAt: number;
  id: string;
  messages: Message[];
  title: string;
  updatedAt: number;
};

type RuntimeStatus = {
  blockers: string[];
  model: string;
  mode: "connected" | "staging";
  ready: boolean;
};

const STORAGE_KEY = "lumi-chat-history-v0.1";
const ACTIVE_KEY = "lumi-chat-active-v0.1";

const GATES = [
  {
    detail: "Segmented 80B run is active; this product branch is isolated.",
    label: "Base pretraining",
    state: "active",
  },
  {
    detail: "License, provenance, PII, secrets, duplicate and leakage filters.",
    label: "SFT data contract",
    state: "ready",
  },
  {
    detail: "Frozen adapters remain checkpoint-bound before final scoring.",
    label: "Benchmark harness",
    state: "ready",
  },
  {
    detail: "Six owned probes cover injection, privacy and safe behavior.",
    label: "Safety suite",
    state: "ready",
  },
  {
    detail: "Blocked until base, SFT, benchmark and safety hashes are sealed.",
    label: "Public serving",
    state: "blocked",
  },
] as const;

function makeId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

function compactTitle(value: string): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length > 42 ? `${compact.slice(0, 39)}…` : compact;
}

function formatAge(timestamp: number): string {
  const minutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60_000));
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function LumiMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`lumi-mark${compact ? " compact" : ""}`} aria-hidden="true">
      <span className="orbit orbit-one" />
      <span className="orbit orbit-two" />
      <span className="mark-core">L</span>
    </div>
  );
}

function Icon({
  name,
  size = 18,
}: {
  name: "arrow" | "chat" | "check" | "grid" | "plus" | "stop" | "trash";
  size?: number;
}) {
  const paths = {
    arrow: <path d="m5 12 7-7 7 7M12 5v14" />,
    chat: <path d="M20 15a4 4 0 0 1-4 4H8l-4 3V7a4 4 0 0 1 4-4h8a4 4 0 0 1 4 4z" />,
    check: <path d="m5 12 4 4L19 6" />,
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="2" />
        <rect x="14" y="3" width="7" height="7" rx="2" />
        <rect x="3" y="14" width="7" height="7" rx="2" />
        <rect x="14" y="14" width="7" height="7" rx="2" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    stop: <rect x="7" y="7" width="10" height="10" rx="2" />,
    trash: (
      <>
        <path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13" />
        <path d="M10 11v5M14 11v5" />
      </>
    ),
  };
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      <g stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7">
        {paths[name]}
      </g>
    </svg>
  );
}

export function LumiStudio() {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [view, setView] = useState<View>("chat");
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
        if (Array.isArray(saved)) setConversations(saved.slice(0, 40));
        const savedActive = localStorage.getItem(ACTIVE_KEY);
        if (typeof savedActive === "string") setActiveId(savedActive);
      } catch {
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem(ACTIVE_KEY);
      } finally {
        setHydrated(true);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations.slice(0, 40)));
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
    else localStorage.removeItem(ACTIVE_KEY);
  }, [activeId, conversations, hydrated]);

  useEffect(() => {
    fetch("/api/status", { cache: "no-store" })
      .then((response) => response.json())
      .then((value: RuntimeStatus) => setRuntime(value))
      .catch(() =>
        setRuntime({
          blockers: ["status_endpoint"],
          mode: "staging",
          model: "lumi-4b-instruct",
          ready: false,
        }),
      );
  }, []);

  const active = useMemo(
    () => conversations.find((item) => item.id === activeId) ?? null,
    [activeId, conversations],
  );

  useEffect(() => {
    scrollRef.current?.scrollTo({
      behavior: "smooth",
      top: scrollRef.current.scrollHeight,
    });
  }, [active?.messages]);

  function newConversation() {
    if (abortRef.current) abortRef.current.abort();
    setActiveId(null);
    setInput("");
    setView("chat");
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function deleteConversation(id: string) {
    const remaining = conversations.filter((item) => item.id !== id);
    setConversations(remaining);
    if (activeId === id) setActiveId(remaining[0]?.id ?? null);
  }

  function updateMessage(conversationId: string, messageId: string, patch: Partial<Message>) {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === conversationId
          ? {
              ...conversation,
              messages: conversation.messages.map((message) =>
                message.id === messageId ? { ...message, ...patch } : message,
              ),
              updatedAt: Date.now(),
            }
          : conversation,
      ),
    );
  }

  async function readStream(
    response: Response,
    onText: (value: string) => void,
  ): Promise<void> {
    if (!response.body) throw new Error("Streaming is unavailable.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") return;
          const payload = JSON.parse(data);
          if (payload.error) throw new Error(payload.error.message ?? "Generation failed.");
          const text = payload.choices?.[0]?.delta?.content;
          if (typeof text === "string") onText(text);
        }
      }
      if (done) return;
    }
  }

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault();
    const prompt = input.trim();
    if (!prompt || running) return;

    const now = Date.now();
    const conversationId = active?.id ?? makeId("conversation");
    const userMessage: Message = {
      content: prompt,
      id: makeId("message"),
      role: "user",
    };
    const assistantMessage: Message = {
      content: "",
      id: makeId("message"),
      pending: true,
      role: "assistant",
    };
    const priorMessages = active?.messages ?? [];
    const nextConversation: Conversation = active
      ? {
          ...active,
          messages: [...active.messages, userMessage, assistantMessage],
          updatedAt: now,
        }
      : {
          createdAt: now,
          id: conversationId,
          messages: [userMessage, assistantMessage],
          title: compactTitle(prompt),
          updatedAt: now,
        };

    setConversations((current) => [
      nextConversation,
      ...current.filter((item) => item.id !== conversationId),
    ]);
    setActiveId(conversationId);
    setInput("");
    setRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/chat", {
        body: JSON.stringify({
          messages: [...priorMessages, userMessage].map(({ content, role }) => ({
            content,
            role,
          })),
          stream: true,
        }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
        signal: controller.signal,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.error?.message ?? `Request failed (${response.status}).`);
      }
      let generated = "";
      await readStream(response, (piece) => {
        generated += piece;
        updateMessage(conversationId, assistantMessage.id, {
          content: generated,
          pending: true,
        });
      });
      updateMessage(conversationId, assistantMessage.id, {
        content: generated || "Lumi returned an empty response.",
        pending: false,
      });
    } catch (error) {
      const stopped = error instanceof DOMException && error.name === "AbortError";
      updateMessage(conversationId, assistantMessage.id, {
        content: stopped
          ? "Generation stopped."
          : `Lumi could not respond: ${error instanceof Error ? error.message : "unknown error"}`,
        pending: false,
      });
    } finally {
      abortRef.current = null;
      setRunning(false);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function stopGeneration() {
    abortRef.current?.abort();
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  return (
    <div className="studio-shell">
      <aside className="sidebar">
        <div className="brand">
          <LumiMark compact />
          <div>
            <strong>Lumi</strong>
            <span>Afnan Technologies</span>
          </div>
        </div>

        <button className="new-chat" onClick={newConversation} type="button">
          <Icon name="plus" />
          <span>New conversation</span>
          <kbd>⌘K</kbd>
        </button>

        <div className="sidebar-label">
          <span>Conversations</span>
          <span>{conversations.length}</span>
        </div>
        <nav className="conversation-list" aria-label="Saved conversations">
          {conversations.length === 0 ? (
            <p className="empty-history">Your local conversations will appear here.</p>
          ) : (
            conversations.map((conversation) => (
              <div
                className={`conversation-row${conversation.id === activeId ? " active" : ""}`}
                key={conversation.id}
              >
                <button
                  className="conversation-open"
                  onClick={() => {
                    setActiveId(conversation.id);
                    setView("chat");
                  }}
                  type="button"
                >
                  <Icon name="chat" size={15} />
                  <span>{conversation.title}</span>
                  <time>{formatAge(conversation.updatedAt)}</time>
                </button>
                <button
                  aria-label={`Delete ${conversation.title}`}
                  className="conversation-delete"
                  onClick={() => deleteConversation(conversation.id)}
                  type="button"
                >
                  <Icon name="trash" size={14} />
                </button>
              </div>
            ))
          )}
        </nav>

        <div className="sidebar-footer">
          <div className={`runtime-pulse ${runtime?.mode ?? "staging"}`} />
          <div>
            <strong>{runtime?.mode === "connected" ? "Model connected" : "Staging mode"}</strong>
            <span>
              {runtime?.mode === "connected"
                ? runtime.model
                : "No final checkpoint attached"}
            </span>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div className="model-identity">
            <span className="model-kicker">LUMI L1</span>
            <strong>{runtime?.model ?? "lumi-4b-instruct"}</strong>
            <span className="model-state">Base training</span>
          </div>
          <nav className="view-switcher" aria-label="Workspace view">
            <button
              className={view === "chat" ? "active" : ""}
              onClick={() => setView("chat")}
              type="button"
            >
              <Icon name="chat" size={16} />
              Chat
            </button>
            <button
              className={view === "readiness" ? "active" : ""}
              onClick={() => setView("readiness")}
              type="button"
            >
              <Icon name="grid" size={15} />
              Readiness
            </button>
          </nav>
          <div className="header-status">
            <span />
            Private staging
          </div>
        </header>

        {view === "chat" ? (
          <section className="chat-view">
            <div className="message-scroll" ref={scrollRef}>
              {!active || active.messages.length === 0 ? (
                <div className="welcome">
                  <LumiMark />
                  <div className="welcome-copy">
                    <span className="eyebrow">PRODUCT SHELL READY</span>
                    <h1>Build beside the model.</h1>
                    <p>
                      The interface, history, streaming and deployment contract are ready
                      to test while Lumi L1 finishes learning.
                    </p>
                  </div>
                  <div className="training-notice">
                    <span className="notice-icon">i</span>
                    <div>
                      <strong>This is not the final model.</strong>
                      <p>
                        Until the sealed 80B checkpoint completes SFT and safety gates,
                        replies come from an explicit staging stream.
                      </p>
                    </div>
                  </div>
                  <div className="prompt-grid">
                    {[
                      ["Explain clearly", "Why does the sky appear blue?"],
                      ["Write code", "Show a safe Python retry helper."],
                      ["বাংলায় বলো", "সহজ ভাষায় নিউরাল নেটওয়ার্ক বোঝাও।"],
                    ].map(([label, prompt]) => (
                      <button
                        key={label}
                        onClick={() => {
                          setInput(prompt);
                          textareaRef.current?.focus();
                        }}
                        type="button"
                      >
                        <span>{label}</span>
                        <p>{prompt}</p>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="messages">
                  {active.messages.map((message) => (
                    <article className={`message ${message.role}`} key={message.id}>
                      <div className="message-avatar">
                        {message.role === "assistant" ? "L" : "You"}
                      </div>
                      <div className="message-body">
                        <div className="message-heading">
                          <strong>{message.role === "assistant" ? "Lumi" : "You"}</strong>
                          {message.pending ? <span>streaming</span> : null}
                        </div>
                        <p>
                          {message.content}
                          {message.pending ? <span className="cursor" /> : null}
                        </p>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </div>

            <div className="composer-zone">
              <form className="composer" onSubmit={sendMessage}>
                <textarea
                  aria-label="Message Lumi"
                  disabled={running}
                  maxLength={65_536}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={onComposerKeyDown}
                  placeholder="Message Lumi…"
                  ref={textareaRef}
                  rows={1}
                  value={input}
                />
                <div className="composer-footer">
                  <div className="composer-mode">
                    <span />
                    {runtime?.mode === "connected" ? "Model stream" : "Staging stream"}
                  </div>
                  <button
                    aria-label={running ? "Stop generation" : "Send message"}
                    className="send"
                    disabled={!running && !input.trim()}
                    onClick={running ? stopGeneration : undefined}
                    type={running ? "button" : "submit"}
                  >
                    <Icon name={running ? "stop" : "arrow"} size={18} />
                  </button>
                </div>
              </form>
              <p className="composer-note">
                History stays in this browser. Lumi may make mistakes after launch.
              </p>
            </div>
          </section>
        ) : (
          <section className="readiness-view">
            <div className="readiness-hero">
              <div>
                <span className="eyebrow">RELEASE CONTROL</span>
                <h1>Nothing ships on hope.</h1>
                <p>
                  Every model, data, benchmark and safety artifact must be sealed by
                  hash before serving can unlock.
                </p>
              </div>
              <div className="readiness-score">
                <strong>0/5</strong>
                <span>production gates passed</span>
              </div>
            </div>

            <div className="gate-grid">
              {GATES.map((gate, index) => (
                <article className={`gate-card ${gate.state}`} key={gate.label}>
                  <div className="gate-number">{String(index + 1).padStart(2, "0")}</div>
                  <div className="gate-copy">
                    <div>
                      <h2>{gate.label}</h2>
                      <span>{gate.state}</span>
                    </div>
                    <p>{gate.detail}</p>
                  </div>
                  {gate.state === "ready" ? (
                    <div className="gate-check">
                      <Icon name="check" size={15} />
                    </div>
                  ) : null}
                </article>
              ))}
            </div>

            <div className="readiness-bottom">
              <article className="contract-card">
                <span className="eyebrow">SERVING CONTRACT</span>
                <h2>Fail-closed by default</h2>
                <p>
                  Public inference remains disabled until the 80B base checkpoint,
                  filtered SFT set, SFT checkpoint, benchmark report, safety report
                  and tokenizer manifest all have exact SHA-256 bindings.
                </p>
                <div className="contract-path">
                  configs/lumi-4b-serving-contract-v0.1.json
                </div>
              </article>
              <article className="suite-card">
                <div className="suite-heading">
                  <div>
                    <span className="eyebrow">OWNED SAFETY CASES</span>
                    <h2>6 probes ready</h2>
                  </div>
                  <strong>v0.1</strong>
                </div>
                <ul>
                  {[
                    "Prompt injection",
                    "Privacy & secrets",
                    "Harmful requests",
                    "Calibration",
                    "Bengali safety",
                    "Over-refusal",
                  ].map((item) => (
                    <li key={item}>
                      <span>
                        <Icon name="check" size={13} />
                      </span>
                      {item}
                    </li>
                  ))}
                </ul>
              </article>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
