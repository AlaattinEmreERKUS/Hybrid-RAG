import { useEffect } from "react";
import ChatPanel from "./components/ChatPanel";
import GraphCanvas from "./components/GraphCanvas";
import { fetchGraph } from "./api/client";
import { useGraphStore } from "./store/useGraphStore";

export default function App() {
  const setGraph = useGraphStore((s) => s.setGraph);
  const clearHighlight = useGraphStore((s) => s.clearHighlight);

  useEffect(() => {
    fetchGraph(300).then(setGraph).catch(console.error);
  }, [setGraph]);

  return (
    <div className="grid h-screen grid-cols-[420px_1fr]">
      <ChatPanel />
      <div className="relative border-l border-slate-700">
        <GraphCanvas />
        <button
          onClick={() => fetchGraph(300).then(setGraph).catch(console.error)}
          className="absolute right-3 top-3 rounded-lg bg-slate-800/80 px-3 py-1.5 text-xs text-slate-200 ring-1 ring-slate-700 hover:ring-indigo-500"
        >
          ⟳ Reload graph
        </button>
        <button
          onClick={clearHighlight}
          className="absolute right-3 top-12 rounded-lg bg-slate-800/80 px-3 py-1.5 text-xs text-slate-200 ring-1 ring-slate-700 hover:ring-yellow-400"
        >
          ✕ Clear highlight
        </button>
      </div>
    </div>
  );
}
