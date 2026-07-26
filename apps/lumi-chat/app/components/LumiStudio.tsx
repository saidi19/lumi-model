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

type IconName =
  | "arrow"
  | "chat"
  | "check"
  | "close"
  | "grid"
  | "menu"
  | "plus"
  | "stop"
  | "trash";

const STORAGE_KEY = "lumi-chat-history-v0.1";
const ACTIVE_KEY = "lumi-chat-active-v0.1";

const GATES = [
  {
    detail: "The segmented 80B run is active and isolated from this product workspace.",
    label: "Base pretraining",
    state: "active",
  },
  {
    detail: "License, provenance, PII, secret, duplicate and leakage filters are defined.",
    label: "SFT data contract",
    state: "prepared",
  },
  {
    detail: "Frozen benchmark adapters are ready for the final checkpoint.",
    label: "Evaluation harness",
    state: "prepared",
  },
  {
    detail: "Six owned probes cover injection, privacy and safe behavior.",
    label: "Safety suite",
    state: "prepared",
  },
  {
    detail: "Serving remains locked until every required artifact is hash-bound.",
    label: "Public serving",
    state: "blocked",
  },
] as const;

const STARTER_PROMPTS = [
  {
    label: "Think clearly",
    prompt: "Compare two product ideas and show the trade-offs.",
  },
  {
    label: "Write & edit",
    prompt: "Draft a concise launch email for a new AI product.",
  },
  {
    label: "Build with code",
    prompt: "Show a safe typed API retry helper.",
  },
  {
    label: "বাংলায় বলো",
    prompt: "সহজ ভাষায় নিউরাল নেটওয়ার্ক বোঝাও।",
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
    <span className={`lumi-mark${compact ? " compact" : ""}`} aria-hidden="true">
      L
    </span>
  );
}

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const glyphs: Record<IconName, string> = {
    arrow: "↑",
    chat: "◌",
    check: "✓",
    close: "×",
    grid: "◇",
    menu: "☰",
    plus: "+",
    stop: "■",
    trash: "×",
  };
  return (
    <span
      aria-hidden="true"
      className={`ui-icon icon-${name}`}
      style={{ fontSize: `${size}px` }}
    >
      {glyphs[name]}
    </span>
  );
}

export function LumiStudio() {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
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

  useEffect(() => {
    function handleShortcut(event: globalThis.KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        abortRef.current?.abort();
        setActiveId(null);
        setInput("");
        setView("chat");
        setSidebarOpen(false);
        requestAnimationFrame(() => textareaRef.current?.focus());
      }
    }
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
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
    abortRef.current?.abort();
    setActiveId(null);
    setInput("");
    setView("chat");
    setSidebarOpen(false);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function deleteConversation(id: string) {
    const remaining = conversations.filter((item) => item.id !== id);
    setConversations(remaining);
    if (activeId === id) setActiveId(remaining[0]?.id ?? null);
  }

  function openConversation(id: string) {
    setActiveId(id);
    setView("chat");
    setSidebarOpen(false);
  }

  function updateMessage(
    conversationId: string,
    messageId: string,
    patch: Partial<Message>,
  ) {
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
          if (payload.error) {
            throw new Error(payload.error.message ?? "Generation failed.");
          }
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
    <div className={`studio-shell${sidebarOpen ? " sidebar-open" : ""}`}>
      <button
        aria-label="Close conversations"
        className="sidebar-scrim"
        onClick={() => setSidebarOpen(false)}
        type="button"
      />

      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand">
            <LumiMark compact />
            <div>
              <strong>Lumi</strong>
              <span>Private workspace</span>
            </div>
          </div>
          <button
            aria-label="Close conversations"
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
            type="button"
          >
            <Icon name="close" size={23} />
          </button>
        </div>

        <button className="new-chat" onClick={newConversation} type="button">
          <Icon name="plus" size={20} />
          <span>New chat</span>
          <kbd>⌘ K</kbd>
        </button>

        <div className="sidebar-heading">
          <span>Recent</span>
          <span>{conversations.length}</span>
        </div>
        <nav className="conversation-list" aria-label="Saved conversations">
          {conversations.length === 0 ? (
            <div className="empty-history">
              <span>◌</span>
              <p>Your conversations will stay here on this device.</p>
            </div>
          ) : (
            conversations.map((conversation) => (
              <div
                className={`conversation-row${conversation.id === activeId ? " active" : ""}`}
                key={conversation.id}
              >
                <button
                  className="conversation-open"
                  onClick={() => openConversation(conversation.id)}
                  type="button"
                >
                  <Icon name="chat" size={17} />
                  <span>{conversation.title}</span>
                  <time>{formatAge(conversation.updatedAt)}</time>
                </button>
                <button
                  aria-label={`Delete ${conversation.title}`}
                  className="conversation-delete"
                  onClick={() => deleteConversation(conversation.id)}
                  type="button"
                >
                  <Icon name="trash" size={18} />
                </button>
              </div>
            ))
          )}
        </nav>

        <div className="sidebar-nav">
          <button
            className={view === "chat" ? "active" : ""}
            onClick={() => {
              setView("chat");
              setSidebarOpen(false);
            }}
            type="button"
          >
            <Icon name="chat" size={18} />
            <span>Chat</span>
          </button>
          <button
            className={view === "readiness" ? "active" : ""}
            onClick={() => {
              setView("readiness");
              setSidebarOpen(false);
            }}
            type="button"
          >
            <Icon name="grid" size={18} />
            <span>Release readiness</span>
            <em>0/5</em>
          </button>
        </div>

        <div className="runtime-card">
          <div className={`runtime-dot ${runtime?.mode ?? "staging"}`} />
          <div>
            <strong>
              {runtime?.mode === "connected" ? "Model connected" : "Staging preview"}
            </strong>
            <span>
              {runtime?.mode === "connected"
                ? runtime.model
                : "Final checkpoint not attached"}
            </span>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div className="header-left">
            <button
              aria-label="Open conversations"
              className="mobile-menu"
              onClick={() => setSidebarOpen(true)}
              type="button"
            >
              <Icon name="menu" size={20} />
            </button>
            <div className="model-selector">
              <LumiMark compact />
              <div>
                <strong>Lumi L1</strong>
                <span>{runtime?.model ?? "lumi-4b-instruct"}</span>
              </div>
              <span className="selector-caret">⌄</span>
            </div>
          </div>
          <div className="header-actions">
            <span className="training-pill">
              <i />
              Base training
            </span>
            <button
              className="readiness-shortcut"
              onClick={() => setView(view === "readiness" ? "chat" : "readiness")}
              type="button"
            >
              <Icon name={view === "readiness" ? "chat" : "grid"} size={18} />
              {view === "readiness" ? "Back to chat" : "Readiness"}
            </button>
          </div>
        </header>

        {view === "chat" ? (
          <section className="chat-view">
            <div className="message-scroll" ref={scrollRef}>
              {!active || active.messages.length === 0 ? (
                <div className="welcome">
                  <LumiMark />
                  <span className="welcome-kicker">LUMI L1 · PRIVATE PREVIEW</span>
                  <h1>What can I help you with?</h1>
                  <p>
                    Chat naturally in English or বাংলা. The final model is still
                    training, so replies here are clearly marked staging previews.
                  </p>
                  <div className="prompt-grid">
                    {STARTER_PROMPTS.map(({ label, prompt }) => (
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
                        <b aria-hidden="true">↗</b>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="messages" aria-live="polite">
                  {active.messages.map((message) => (
                    <article className={`message ${message.role}`} key={message.id}>
                      <div className="message-avatar">
                        {message.role === "assistant" ? "L" : "You"}
                      </div>
                      <div className="message-body">
                        <div className="message-heading">
                          <strong>{message.role === "assistant" ? "Lumi" : "You"}</strong>
                          {message.pending ? <span>writing</span> : null}
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
                  placeholder="Message Lumi"
                  ref={textareaRef}
                  rows={1}
                  value={input}
                />
                <div className="composer-footer">
                  <div className="composer-mode">
                    <span />
                    {runtime?.mode === "connected" ? "Live model" : "Staging preview"}
                  </div>
                  <button
                    aria-label={running ? "Stop generation" : "Send message"}
                    className="send"
                    disabled={!running && !input.trim()}
                    onClick={running ? stopGeneration : undefined}
                    type={running ? "button" : "submit"}
                  >
                    <Icon name={running ? "stop" : "arrow"} size={19} />
                  </button>
                </div>
              </form>
              <p className="composer-note">
                <span>History stays on this device</span>
                <i>•</i>
                <span>Enter to send · Shift + Enter for a new line</span>
              </p>
            </div>
          </section>
        ) : (
          <section className="readiness-view">
            <div className="readiness-container">
              <div className="readiness-hero">
                <div>
                  <span className="section-kicker">RELEASE CONTROL</span>
                  <h1>Ready only when the evidence is.</h1>
                  <p>
                    Lumi stays private until the base model, SFT, benchmarks and
                    safety evidence are sealed to exact artifact hashes.
                  </p>
                </div>
                <div className="readiness-summary">
                  <strong>0%</strong>
                  <span>production unlocked</span>
                  <div>
                    <i />
                  </div>
                  <small>3 foundations prepared</small>
                </div>
              </div>

              <div className="gate-grid">
                {GATES.map((gate, index) => (
                  <article className={`gate-card ${gate.state}`} key={gate.label}>
                    <div className="gate-top">
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <em>{gate.state}</em>
                    </div>
                    <h2>{gate.label}</h2>
                    <p>{gate.detail}</p>
                    {gate.state === "prepared" ? (
                      <div className="gate-check">
                        <Icon name="check" size={15} />
                      </div>
                    ) : null}
                  </article>
                ))}
              </div>

              <div className="readiness-bottom">
                <article className="contract-card">
                  <span className="section-kicker">SERVING CONTRACT</span>
                  <h2>Fail-closed by default</h2>
                  <p>
                    Public inference cannot unlock without exact bindings for the 80B
                    base, filtered SFT data, instruct checkpoint, tokenizer, benchmark
                    report and safety report.
                  </p>
                  <code>configs/lumi-4b-serving-contract-v0.1.json</code>
                </article>
                <article className="suite-card">
                  <div>
                    <span className="section-kicker">OWNED SAFETY SUITE</span>
                    <strong>6 probes</strong>
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
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
