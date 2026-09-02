import { WorkflowDef, WorkflowId, NodeId } from "./types";

const T = (id: string, label: string, deps: string[], durationMs: number, fail = false) => ({
  id: NodeId(id), label, dependsOn: deps.map(NodeId), durationMs, simulateFailure: fail,
});

export const WORKFLOWS: WorkflowDef[] = [
  {
    id: WorkflowId("cicd-pipeline"),
    name: "CI/CD Pipeline",
    description: "Checkout -> parallel lint/typecheck -> test -> parallel deploy to staging+prod. A realistic branching-and-merging DAG, not a linear chain.",
    tasks: [
      T("checkout", "Checkout code", [], 600),
      T("lint", "Lint", ["checkout"], 900),
      T("typecheck", "Typecheck", ["checkout"], 1100),
      T("test", "Run tests", ["lint", "typecheck"], 1500),
      T("build", "Build artifact", ["test"], 1200),
      T("deploy-staging", "Deploy to staging", ["build"], 800),
      T("deploy-prod", "Deploy to production", ["build"], 900),
    ],
  },
  {
    id: WorkflowId("etl-pipeline"),
    name: "ETL Data Pipeline",
    description: "Extract from 3 independent sources in parallel, transform once all land, load, then run validation and notify in parallel.",
    tasks: [
      T("extract-crm", "Extract: CRM", [], 1000),
      T("extract-orders", "Extract: Orders DB", [], 1300),
      T("extract-events", "Extract: Event stream", [], 900),
      T("transform", "Transform & join", ["extract-crm", "extract-orders", "extract-events"], 1800),
      T("load", "Load to warehouse", ["transform"], 1000),
      T("validate", "Data quality validation", ["load"], 700),
      T("notify", "Notify stakeholders", ["load"], 400),
    ],
  },
  {
    id: WorkflowId("failure-demo"),
    name: "Failure Propagation Demo",
    description: "One task is scripted to fail, to demonstrate that downstream-dependent tasks are correctly SKIPPED (not silently run) while independent branches still complete.",
    tasks: [
      T("setup", "Setup environment", [], 500),
      T("risky-migration", "Risky DB migration", ["setup"], 800, true),
      T("run-migrated-tests", "Tests against migrated schema", ["risky-migration"], 700),
      T("independent-lint", "Independent: lint (unaffected)", ["setup"], 600),
      T("independent-docs", "Independent: build docs (unaffected)", ["setup"], 500),
    ],
  },
];
