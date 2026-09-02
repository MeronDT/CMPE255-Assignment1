/**
 * Executes a workflow's tasks layer by layer (from topologicalLayers): every
 * task within a layer runs in parallel (Promise.all, since they're mutually
 * independent by construction), and a layer only starts once the previous
 * layer has fully settled. If a task fails, every downstream task that
 * (transitively) depends on it is marked "skipped" rather than run --
 * genuine dependency-aware behavior, not just a flat parallel-run-everything.
 */
import { NodeId, TaskDef, WorkflowDef, RunId, RunState, NodeStatus } from "./types";
import { topologicalLayers } from "./dag";

export type StatusListener = (runId: RunId, nodeId: NodeId, status: NodeStatus) => void;

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeDescendants(tasks: TaskDef[]): Map<string, Set<string>> {
  const dependents = new Map<string, string[]>();
  for (const t of tasks) {
    for (const dep of t.dependsOn) {
      if (!dependents.has(dep)) dependents.set(dep, []);
      dependents.get(dep)!.push(t.id);
    }
  }
  const descendants = new Map<string, Set<string>>();
  const dfs = (id: string): Set<string> => {
    if (descendants.has(id)) return descendants.get(id)!;
    const result = new Set<string>();
    for (const child of dependents.get(id) ?? []) {
      result.add(child);
      for (const grandchild of dfs(child)) result.add(grandchild);
    }
    descendants.set(id, result);
    return result;
  };
  for (const t of tasks) dfs(t.id);
  return descendants;
}

export async function executeWorkflow(
  workflow: WorkflowDef,
  runId: RunId,
  onStatus: StatusListener
): Promise<RunState> {
  const topo = topologicalLayers(workflow.tasks);
  if (!topo.ok) {
    throw new Error(`Workflow "${workflow.name}" has a cycle involving: ${topo.cycleNodes.join(", ")}`);
  }

  const statuses: Record<string, NodeStatus> = {};
  for (const t of workflow.tasks) statuses[t.id] = { status: "pending" };

  const descendants = computeDescendants(workflow.tasks);
  const failedOrSkipped = new Set<string>();
  const startedAt = Date.now();

  for (const layer of topo.layers) {
    await Promise.all(
      layer.map(async (nodeId) => {
        const task = workflow.tasks.find((t) => t.id === nodeId)!;

        // Skip if any (transitive) ancestor already failed or was skipped.
        const shouldSkip = task.dependsOn.some((dep) => failedOrSkipped.has(dep));
        if (shouldSkip) {
          const s: NodeStatus = { status: "skipped", reason: "an upstream dependency failed" };
          statuses[nodeId] = s;
          failedOrSkipped.add(nodeId);
          onStatus(runId, nodeId, s);
          return;
        }

        const running: NodeStatus = { status: "running", startedAt: Date.now() };
        statuses[nodeId] = running;
        onStatus(runId, nodeId, running);

        await sleep(task.durationMs);

        if (task.simulateFailure) {
          const failed: NodeStatus = {
            status: "failed",
            startedAt: running.startedAt,
            finishedAt: Date.now(),
            error: `Task "${task.label}" failed (simulated)`,
          };
          statuses[nodeId] = failed;
          failedOrSkipped.add(nodeId);
          onStatus(runId, nodeId, failed);
        } else {
          const success: NodeStatus = {
            status: "success",
            startedAt: running.startedAt,
            finishedAt: Date.now(),
            output: { message: `${task.label} completed` },
          };
          statuses[nodeId] = success;
          onStatus(runId, nodeId, success);
        }
      })
    );
  }

  return {
    runId,
    workflowId: workflow.id,
    startedAt,
    finishedAt: Date.now(),
    statuses,
    executionOrder: topo.layers,
  };
}
