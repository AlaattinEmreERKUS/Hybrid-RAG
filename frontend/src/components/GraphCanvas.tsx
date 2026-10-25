import { useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useGraphStore } from "../store/useGraphStore";

const TYPE_COLORS: Record<string, string> = {
  PERSON: "#f472b6",
  ORGANIZATION: "#60a5fa",
  TECHNOLOGY: "#34d399",
  CONCEPT: "#a78bfa",
  PRODUCT: "#fbbf24",
  LOCATION: "#fb923c",
  EVENT: "#f87171",
  DEFAULT: "#94a3b8",
};

export default function GraphCanvas() {
  const fgRef = useRef<any>(null);
  const { graph, highlightNodes, highlightEdges } = useGraphStore();

  // force-graph mutates its data objects (adds x/y) — give it stable copies
  const data = useMemo(
    () => ({
      nodes: graph.nodes.map((n) => ({ ...n })),
      links: graph.links.map((l) => ({ ...l })),
    }),
    [graph]
  );

  const dimmed = highlightNodes.size > 0;

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={data}
      backgroundColor="#0f172a"
      cooldownTicks={120}
      onEngineStop={() => fgRef.current?.zoomToFit(400, 60)}
      nodeCanvasObject={(node: any, ctx, scale) => {
        const hot = highlightNodes.has(node.id);
        const r = 3 + Math.min(node.degree ?? 1, 12) * 0.6;
        ctx.globalAlpha = dimmed && !hot ? 0.15 : 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = TYPE_COLORS[node.entityType ?? ""] ?? TYPE_COLORS.DEFAULT;
        ctx.fill();
        if (hot) {
          ctx.strokeStyle = "#fde047";
          ctx.lineWidth = 2 / scale;
          ctx.stroke();
        }
        if (scale > 1.4 || hot) {
          ctx.font = `${Math.max(10 / scale, 2)}px Inter, sans-serif`;
          ctx.fillStyle = hot ? "#fde047" : "#cbd5e1";
          ctx.textAlign = "center";
          ctx.fillText(node.label, node.x, node.y + r + 4 / scale);
        }
        ctx.globalAlpha = 1;
      }}
      nodePointerAreaPaint={(node: any, color, ctx) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI);
        ctx.fill();
      }}
      linkColor={(l: any) => (highlightEdges.has(l.id) ? "#fde047" : "#334155")}
      linkWidth={(l: any) => (highlightEdges.has(l.id) ? 2.5 : 0.5)}
      linkDirectionalParticles={(l: any) => (highlightEdges.has(l.id) ? 3 : 0)}
      linkDirectionalParticleSpeed={0.006}
      linkLabel={(l: any) => l.predicate}
    />
  );
}
