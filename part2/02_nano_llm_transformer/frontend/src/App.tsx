import { useState } from "react";
import { ChatPage } from "./components/ChatPage";
import { ModelInsightsPage } from "./components/ModelInsightsPage";
import { AutoResearchPage } from "./components/AutoResearchPage";

type Tab = "chat" | "insights" | "autoresearch";

function App() {
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-6xl px-4 pb-16 pt-8">
        <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">🦙 NanoLlama</h1>
            <p className="text-sm text-slate-500">A from-scratch autoregressive SFT LLM, sized to fit a laptop GPU</p>
          </div>
          <nav className="flex rounded-xl bg-slate-100 p-1">
            {(
              [
                ["chat", "Chat"],
                ["insights", "Model Insights"],
                ["autoresearch", "AutoResearch"],
              ] as [Tab, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                  tab === key ? "bg-white text-indigo-600 shadow-sm" : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </header>

        {tab === "chat" && <ChatPage />}
        {tab === "insights" && <ModelInsightsPage />}
        {tab === "autoresearch" && <AutoResearchPage />}
      </div>
    </div>
  );
}

export default App;
