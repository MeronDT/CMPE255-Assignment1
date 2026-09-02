import { useEffect, useMemo, useRef, useState } from "react";
import { type WorkflowDef } from "./api";

type Status = { status: "pending" | "running" | "success" | "failed" | "skipped"; [k: string]: any };

const STATUS_COLOR: Record<string, string> = {
  pending: "#2a3550", running: "#e6a24d", success: "#4dd6b6", failed: "#e65c5c", skipped: "#8a97b5",
};

/** Client-side layering (longest-path-from-root), independent of the backend's
 * own Kahn's-algorithm layering -- used purely for visual layout before a run
 * even starts, when there's no execution result to read layers from yet. */
function computeLayers(tasks: WorkflowDef["tasks"]): Map<string, number> {
  const layer = new Map<string, number>();
  const byId = new Map(tasks.map((t) => [t.id, t]));
  function depth(id: string, seen: Set<string>): number {
    if (layer.has(id)) return layer.get(id)!;
    if (seen.has(id)) throw new Error("cycle detected in client-side layering");
    seen.add(id);
    const task = byId.get(id)!;
    const d = task.dependsOn.length === 0 ? 0 : Math.max(...task.dependsOn.map((dep) => depth(dep, seen))) + 1;
    layer.set(id, d);
    return d;
  }
  for (const t of tasks) depth(t.id, new Set());
  return layer;
}

const NODE_W = 170, NODE_H = 56, COL_GAP = 220, ROW_GAP = 80, PAD = 30;

export default function DagView({ workflow, statuses }: { workflow: WorkflowDef; statuses: Record<string, Status> }) {
  const layers = useMemo(() => computeLayers(workflow.tasks), [workflow]);
  const byLayer = useMemo(() => {
    const m = new Map<number, string[]>();
    for (const t of workflow.tasks) {
      const l = layers.get(t.id)!;
      if (!m.has(l)) m.set(l, []);
      m.get(l)!.push(t.id);
    }
    return m;
  }, [layers, workflow]);

  const positions = useMemo(() => {
    const pos = new Map<string, { x: number; y: number }>();
    for (const [layerIdx, ids] of byLayer.entries()) {
      ids.forEach((id, row) => {
        pos.set(id, { x: PAD + layerIdx * COL_GAP, y: PAD + row * ROW_GAP });
      });
    }
    return pos;
  }, [byLayer]);

  const maxLayer = Math.max(...Array.from(layers.values()));
  const maxRows = Math.max(...Array.from(byLayer.values()).map((v) => v.length));
  const width = PAD * 2 + (maxLayer + 1) * COL_GAP;
  const height = PAD * 2 + maxRows * ROW_GAP;

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={width} height={height} style={{ background: "#0f1420", borderRadius: 8 }}>
        {/* edges */}
        {workflow.tasks.map((t) =>
          t.dependsOn.map((dep) => {
            const from = positions.get(dep)!, to = positions.get(t.id)!;
            const x1 = from.x + NODE_W, y1 = from.y + NODE_H / 2;
            const x2 = to.x, y2 = to.y + NODE_H / 2;
            const midX = (x1 + x2) / 2;
            return (
              <path
                key={`${dep}->${t.id}`}
                d={`M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`}
                stroke="#2a3550" strokeWidth={2} fill="none"
              />
            );
          })
        )}
        {/* nodes */}
        {workflow.tasks.map((t) => {
          const p = positions.get(t.id)!;
          const s = statuses[t.id]?.status ?? "pending";
          return (
            <g key={t.id} transform={`translate(${p.x}, ${p.y})`}>
              <rect width={NODE_W} height={NODE_H} rx={8} fill="#161d2e" stroke={STATUS_COLOR[s]} strokeWidth={2} />
              <text x={12} y={22} fill="#e6ebf5" fontSize={13} fontWeight={600}>{t.label.length > 20 ? t.label.slice(0, 20) + "…" : t.label}</text>
              <text x={12} y={40} fill={STATUS_COLOR[s]} fontSize={11}>{s}</text>
              {s === "running" && (
                <circle cx={NODE_W - 16} cy={16} r={5} fill={STATUS_COLOR[s]}>
                  <animate attributeName="opacity" values="1;0.3;1" dur="1s" repeatCount="indefinite" />
                </circle>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
