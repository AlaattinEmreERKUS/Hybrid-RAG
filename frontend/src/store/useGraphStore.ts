import { create } from "zustand";
import type { GraphPayload, QueryResponse } from "../types";

interface GraphState {
  graph: GraphPayload;
  highlightNodes: Set<string>;
  highlightEdges: Set<string>;
  setGraph: (g: GraphPayload) => void;
  /** Merge a query's retrieved subgraph into the canvas + light it up. */
  applyRetrieval: (r: QueryResponse) => void;
  clearHighlight: () => void;
}

export const useGraphStore = create<GraphState>((set, get) => ({
  graph: { nodes: [], links: [] },
  highlightNodes: new Set<string>(),
  highlightEdges: new Set<string>(),

  setGraph: (graph) => set({ graph }),

  applyRetrieval: (r) => {
    const { graph } = get();
    const nodeIds = new Set(graph.nodes.map((n) => n.id));
    const linkIds = new Set(graph.links.map((l) => l.id));
    set({
      graph: {
        nodes: [...graph.nodes, ...r.subgraph.nodes.filter((n) => !nodeIds.has(n.id))],
        links: [...graph.links, ...r.subgraph.links.filter((l) => !linkIds.has(l.id))],
      },
      highlightNodes: new Set(r.retrieval.nodeIds),
      highlightEdges: new Set(r.retrieval.edgeIds),
    });
  },

  clearHighlight: () =>
    set({ highlightNodes: new Set<string>(), highlightEdges: new Set<string>() }),
}));
