/**
 * Kahn's algorithm for topological sort, layered so that all nodes in one
 * layer are mutually independent and can execute in parallel -- not just a
 * flat ordering. Also does real cycle detection (a workflow with a cycle has
 * no valid execution order at all, and must be rejected before running).
 */
import { NodeId, TaskDef } from "./types";

export interface TopoResult {
  ok: true;
  layers: NodeId[][];
}
export interface CycleError {
  ok: false;
  cycleNodes: NodeId[];
}

export function topologicalLayers(tasks: TaskDef[]): TopoResult | CycleError {
  const inDegree = new Map<string, number>();
  const dependents = new Map<string, NodeId[]>(); // node -> nodes that depend on it
  const byId = new Map<string, TaskDef>();

  for (const t of tasks) {
    byId.set(t.id, t);
    inDegree.set(t.id, t.dependsOn.length);
    if (!dependents.has(t.id)) dependents.set(t.id, []);
  }
  for (const t of tasks) {
    for (const dep of t.dependsOn) {
      if (!byId.has(dep)) {
        throw new Error(`Task "${t.id}" depends on unknown task "${dep}"`);
      }
      dependents.get(dep)!.push(t.id);
    }
  }

  const layers: NodeId[][] = [];
  let frontier = tasks.filter((t) => inDegree.get(t.id) === 0).map((t) => t.id);
  const visited = new Set<string>();

  while (frontier.length > 0) {
    layers.push(frontier);
    frontier.forEach((id) => visited.add(id));
    const next: NodeId[] = [];
    for (const id of frontier) {
      for (const dep of dependents.get(id) ?? []) {
        const remaining = (inDegree.get(dep) ?? 0) - 1;
        inDegree.set(dep, remaining);
        if (remaining === 0) next.push(dep);
      }
    }
    frontier = next;
  }

  if (visited.size !== tasks.length) {
    const cycleNodes = tasks.map((t) => t.id).filter((id) => !visited.has(id));
    return { ok: false, cycleNodes };
  }

  return { ok: true, layers };
}

/** Small self-test run at module load, per the "verify before trusting"
 * discipline used throughout this repo -- a broken topological sort would
 * silently corrupt every workflow run, so it's checked once, cheaply, here. */
function selfTest() {
  const T = (id: string, deps: string[]): TaskDef => ({
    id: NodeId(id), label: id, dependsOn: deps.map(NodeId), durationMs: 1,
  });
  const result = topologicalLayers([T("a", []), T("b", ["a"]), T("c", ["a"]), T("d", ["b", "c"])]);
  if (!result.ok) throw new Error("DAG self-test failed: expected a valid topological order");
  const flat = result.layers.map((l) => [...l].sort());
  const expected = [["a"], ["b", "c"], ["d"]];
  if (JSON.stringify(flat) !== JSON.stringify(expected)) {
    throw new Error(`DAG self-test failed: got ${JSON.stringify(flat)}, expected ${JSON.stringify(expected)}`);
  }
  const cyclic = topologicalLayers([T("x", ["y"]), T("y", ["x"])]);
  if (cyclic.ok) throw new Error("DAG self-test failed: expected cycle detection to trigger");
}
selfTest();
