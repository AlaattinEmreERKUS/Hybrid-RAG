import { useRef, useState } from "react";
import { postQuery, uploadFile } from "../api/client";
import { useGraphStore } from "../store/useGraphStore";
import type { QueryResponse } from "../types";

interface Msg {
  role: "user" | "assistant";
  text: string;
  retrieval?: QueryResponse;
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const applyRetrieval = useGraphStore((s) => s.applyRetrieval);

  async function send() {
    if (!input.trim() || busy) return;
    const question = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setBusy(true);
    try {
      const rsp = await postQuery(question);
      setMessages((m) => [...m, { role: "assistant", text: rsp.answer, retrieval: rsp }]);
      applyRetrieval(rsp); // light up graph immediately on answer
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: `Error: ${e}` }]);
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(file: File | undefined) {
    if (!file || uploading) return;
    setUploading(true);
    setMessages((m) => [...m, { role: "user", text: `📄 Uploading ${file.name}…` }]);
    try {
      const r = await uploadFile(file);
      setMessages((m) => [
        ...m,
        { role: "assistant", text: `Ingested "${file.name}" — ${r.chunks} chunks (${r.status}). Reload graph to see new nodes.` },
      ]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: `Upload failed: ${e}` }]);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="flex h-full flex-col bg-slate-900">
      <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
        <h1 className="text-sm font-semibold text-white">Hybrid GraphRAG</h1>
        <button
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          className="rounded-lg bg-slate-800 px-3 py-1.5 text-xs text-slate-200 ring-1 ring-slate-700 hover:ring-indigo-500 disabled:opacity-50"
        >
          {uploading ? "Ingesting…" : "+ Upload doc"}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.md,.txt"
          className="hidden"
          onChange={(e) => onUpload(e.target.files?.[0])}
        />
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="text-sm text-slate-500">
            Upload a document, then ask a question. Click any answer to highlight
            its sources in the graph.
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            onClick={() => m.retrieval && applyRetrieval(m.retrieval)}
            className={`max-w-[85%] whitespace-pre-wrap rounded-xl px-4 py-2 text-sm ${
              m.role === "user"
                ? "ml-auto bg-indigo-600 text-white"
                : "cursor-pointer bg-slate-800 text-slate-100 ring-1 ring-slate-700 hover:ring-yellow-400"
            }`}
            title={m.retrieval ? "Click to highlight sources in graph" : undefined}
          >
            {m.text}
          </div>
        ))}
        {busy && <div className="animate-pulse text-sm text-slate-400">retrieving…</div>}
      </div>

      <div className="flex gap-2 border-t border-slate-700 p-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask the knowledge graph…"
          className="flex-1 rounded-lg bg-slate-800 px-3 py-2 text-sm text-white outline-none ring-1 ring-slate-700 focus:ring-indigo-500"
        />
        <button
          onClick={send}
          disabled={busy}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
