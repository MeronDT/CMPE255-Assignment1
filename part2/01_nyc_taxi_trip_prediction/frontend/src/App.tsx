import { useState } from "react";
import { EstimatorPage } from "./components/EstimatorPage";
import { ModelInsightsPage } from "./components/ModelInsightsPage";

type Tab = "estimator" | "insights";

function App() {
  const [tab, setTab] = useState<Tab>("estimator");

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-6xl px-4 pb-16 pt-8">
        <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">NYC Taxi Trip Estimator</h1>
            <p className="text-sm text-slate-500">CRISP-DM end-to-end: real TLC trip data → LightGBM models → live estimate</p>
          </div>
          <nav className="flex rounded-xl bg-slate-100 p-1">
            <button
              onClick={() => setTab("estimator")}
              className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                tab === "estimator" ? "bg-white text-indigo-600 shadow-sm" : "text-slate-500 hover:text-slate-700"
              }`}
            >
              Trip Estimator
            </button>
            <button
              onClick={() => setTab("insights")}
              className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                tab === "insights" ? "bg-white text-indigo-600 shadow-sm" : "text-slate-500 hover:text-slate-700"
              }`}
            >
              Model Insights
            </button>
          </nav>
        </header>

        {tab === "estimator" ? <EstimatorPage /> : <ModelInsightsPage />}
      </div>
    </div>
  );
}

export default App;
