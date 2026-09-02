import express from "express";
import cors from "cors";
import { WebSocketServer, WebSocket } from "ws";
import { createServer } from "http";
import { WORKFLOWS } from "./workflows";
import { executeWorkflow } from "./executor";
import { RunId, WorkflowId, NodeId, NodeStatus } from "./types";
import { randomUUID } from "crypto";

const app = express();
app.use(cors());
app.use(express.json());

const server = createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });
const clients = new Set<WebSocket>();
wss.on("connection", (ws) => {
  clients.add(ws);
  ws.on("close", () => clients.delete(ws));
});

function broadcast(msg: unknown) {
  const payload = JSON.stringify(msg);
  for (const client of clients) {
    if (client.readyState === WebSocket.OPEN) client.send(payload);
  }
}

const runHistory: Record<string, unknown> = {};

app.get("/api/health", (_req, res) => res.json({ status: "ok" }));

app.get("/api/workflows", (_req, res) => {
  res.json(WORKFLOWS.map((w) => ({ id: w.id, name: w.name, description: w.description, nTasks: w.tasks.length })));
});

app.get("/api/workflows/:id", (req, res) => {
  const wf = WORKFLOWS.find((w) => w.id === req.params.id);
  if (!wf) return res.status(404).json({ error: "not found" });
  res.json(wf);
});

app.get("/api/runs", (_req, res) => {
  res.json(Object.values(runHistory));
});

app.post("/api/workflows/:id/run", async (req, res) => {
  const wf = WORKFLOWS.find((w) => w.id === req.params.id);
  if (!wf) return res.status(404).json({ error: "not found" });

  const runId = RunId(randomUUID());
  res.json({ runId });

  const onStatus = (rid: RunId, nodeId: NodeId, status: NodeStatus) => {
    broadcast({ type: "status", runId: rid, nodeId, status });
  };

  broadcast({ type: "run-started", runId, workflowId: wf.id });
  try {
    const result = await executeWorkflow(wf, runId, onStatus);
    runHistory[runId] = result;
    broadcast({ type: "run-finished", runId, result });
  } catch (e: any) {
    broadcast({ type: "run-error", runId, error: e.message });
  }
});

const PORT = 8009;
server.listen(PORT, () => {
  console.log(`FlowForge backend listening on http://localhost:${PORT} (WS: /ws)`);
});
