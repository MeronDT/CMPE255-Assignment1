import { useEffect, useRef, useState } from "react";
import { sendChat } from "../api/client";
import type { ChatMessage } from "../types";

const SUGGESTED_PROMPTS = [
  "What is the capital of France?",
  "Give three tips for staying healthy.",
  "Write a short poem about the ocean.",
  "Explain what a computer is to a child.",
];

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [lastStats, setLastStats] = useState<{ ms: number; tokPerSec: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async (text: string) => {
    if (!text.trim() || sending) return;
    const next = [...messages, { role: "user" as const, content: text }];
    setMessages(next);
    setInput("");
    setSending(true);
    setError(null);
    try {
      const res = await sendChat(next);
      setMessages([...next, { role: "assistant", content: res.reply }]);
      setLastStats({ ms: res.generation_ms, tokPerSec: res.tokens_per_sec });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Chat failed");
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="mx-auto flex h-[calc(100vh-140px)] max-w-3xl flex-col rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex-1 space-y-4 overflow-y-auto p-5">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <div className="text-4xl">🦙</div>
            <p className="text-sm text-slate-400">
              NanoLlama — a ~30M-parameter model, pretrained on TinyStories and instruction-tuned on Alpaca. Don't expect
              GPT-4; this is a toy scale, by design (see the Model Insights tab).
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTED_PROMPTS.map((p) => (
                <button
                  key={p}
                  onClick={() => send(p)}
                  className="rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:border-indigo-300 hover:text-indigo-600"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm ${
                m.role === "user" ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-800"
              }`}
            >
              {m.content || <span className="italic text-slate-400">(empty response)</span>}
            </div>
          </div>
        ))}

        {sending && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-slate-100 px-4 py-2.5 text-sm text-slate-400">generating…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <div className="border-t border-red-100 bg-red-50 px-4 py-2 text-xs text-red-600">{error}</div>}
      {lastStats && !sending && (
        <div className="border-t border-slate-100 px-4 py-1.5 text-right text-xs text-slate-400">
          {lastStats.ms.toFixed(0)}ms · {lastStats.tokPerSec.toFixed(0)} tok/s
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2 border-t border-slate-100 p-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask NanoLlama something…"
          className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none"
        />
        <button
          type="submit"
          disabled={sending || !input.trim()}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
        >
          Send
        </button>
      </form>
    </div>
  );
}
