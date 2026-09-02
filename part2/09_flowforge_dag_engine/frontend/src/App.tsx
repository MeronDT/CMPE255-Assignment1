import { useEffect, useRef, useState } from "react";
import { api, type WorkflowSummary, type WorkflowDef } from "./api";
import DagView from "./DagView";

type Status = { status: string; [k: string]: any };

function useWorkflows() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  useEffect(() => { api.workflows().then(setWorkflows).catch(console.error); }, []);
  return workflows;
}

export default function App() {
  const workflows = useWorkflows();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [workflow, setWorkflow] = useState<WorkflowDef | null>(null);
  const [statuses, setStatuses] = useState<Record<string, Status>>({});
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const activeRunId = useRef<string | null>(null);

  useEffect(() => {
    if (!selectedId && workflows.length > 0) setSelectedId(workflows[0].id);
  }, [workflows, selectedId]);

  useEffect(() => {
    if (!selectedId) return;
    api.workflow(selectedId).then((wf) => {
      setWorkflow(wf);
      const initial: Record<string, Status> = {};
      wf.tasks.forEach((t) => { initial[t.id] = { status: "pending" }; });
      setStatuses(initial);
      setLog([]);
    });
  }, [selectedId]);

  useEffect(() => {
    const ws = new WebSocket(api.wsUrl);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "status" && msg.runId === activeRunId.current) {
        setStatuses((prev) => ({ ...prev, [msg.nodeId]: msg.status }));
        setLog((prev) => [...prev, `${msg.nodeId}: ${msg.status.status}`]);
      }
      if (msg.type === "run-finished" && msg.runId === activeRunId.current) {
        setRunning(false);
        setLog((prev) => [...prev, `Run finished.`]);
      }
    };
    return () => ws.close();
  }, []);

  const runWorkflow = async () => {
    if (!selectedId || !workflow) return;
    const initial: Record<string, Status> = {};
    workflow.tasks.forEach((t) => { initial[t.id] = { status: "pending" }; });
    setStatuses(initial);
    setLog(["Run started..."]);
    setRunning(true);
    const { runId } = await api.runWorkflow(selectedId);
    activeRunId.current = runId;
  };

  const nSuccess = Object.values(statuses).filter((s) => s.status === "success").length;
  const nFailed = Object.values(statuses).filter((s) => s.status === "failed").length;
  const nSkipped = Object.values(statuses).filter((s) => s.status === "skipped").length;

  return (
    <div className="app">
      <div className="sidebar">
        <h1>FlowForge</h1>
        <div className="sub">DAG Execution Engine</div>
        <nav>
          <div className="section-label">Workflows</div>
          {workflows.map((w) => (
            <button key={w.id} className={selectedId === w.id ? "active" : ""} onClick={() => setSelectedId(w.id)}>{w.name}</button>
          ))}
        </nav>
      </div>
      <div className="main" style={{ maxWidth: "100%" }}>
        {workflow && (
          <>
            <h2>{workflow.name}</h2>
            <p className="lede">{workflow.description}</p>
            <div className="grid cols-4">
              <div className="card"><h3>Tasks</h3><div className="stat-value">{workflow.tasks.length}</div></div>
              <div className="card"><h3>Success</h3><div className="stat-value" style={{ color: "#4dd6b6" }}>{nSuccess}</div></div>
              <div className="card"><h3>Failed</h3><div className="stat-value" style={{ color: "#e65c5c" }}>{nFailed}</div></div>
              <div className="card"><h3>Skipped</h3><div className="stat-value" style={{ color: "#8a97b5" }}>{nSkipped}</div></div>
            </div>
            <div className="sim-controls" style={{ margin: "16px 0" }}>
              <button className="btn" onClick={runWorkflow} disabled={running}>{running ? "Running..." : "Run Workflow"}</button>
            </div>
            <div className="card">
              <h4>DAG Visualization</h4>
              <DagView workflow={workflow} statuses={statuses} />
            </div>
            <div className="card" style={{ marginTop: 14 }}>
              <h4>Execution Log</h4>
              <div style={{ fontFamily: "monospace", fontSize: 12.5, maxHeight: 200, overflowY: "auto" }}>
                {log.map((l, i) => <div key={i}>{l}</div>)}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
