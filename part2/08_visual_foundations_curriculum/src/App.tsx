import { useState } from "react";
import NaiveBayes from "./components/NaiveBayes";
import ModelEvaluation from "./components/ModelEvaluation";
import Calculus from "./components/Calculus";
import Backprop from "./components/Backprop";
import InterviewPrep from "./components/InterviewPrep";

type Page = "intro" | "naive-bayes" | "evaluation" | "calculus" | "backprop" | "interview";

const PAGES: { id: Page; label: string }[] = [
  { id: "naive-bayes", label: "1. Naive Bayes" },
  { id: "evaluation", label: "2. Model Evaluation" },
  { id: "calculus", label: "3. Calculus & Gradient Descent" },
  { id: "backprop", label: "4. Chain Rule & Backprop" },
];

function Intro({ onStart }: { onStart: () => void }) {
  return (
    <div>
      <h2>Data Science Visual Foundations Curriculum</h2>
      <p className="lede">
        Four foundational data science &amp; ML concepts, taught with deep intuition, rigorous math, and
        live interactive simulations you can manipulate yourself — not just read about.
      </p>
      <div className="card">
        <h4>What's Inside</h4>
        <ul>
          <li><strong>Naive Bayes</strong> — Bayes' theorem, the independence assumption, a live one-word spam classifier</li>
          <li><strong>Model Evaluation</strong> — confusion matrices, Type I/II errors, ROC-AUC, cost matrices, precision/recall tradeoffs, all on one interactive dataset</li>
          <li><strong>Differential Calculus &amp; Gradient Descent</strong> — what a derivative actually means, and a live gradient-descent simulation you can run, step, and break (try a too-high learning rate)</li>
          <li><strong>Chain Rule &amp; Backpropagation</strong> — the calculus behind how neural networks learn, traced through a single neuron's forward and backward pass</li>
        </ul>
        <p>Every section ends with a short quiz. A dedicated <strong>Interview Prep</strong> page collects common data science interview questions (and model answers) for all four topics.</p>
        <button className="btn" onClick={onStart}>Start with Naive Bayes &rarr;</button>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("intro");

  return (
    <div className="app">
      <div className="sidebar">
        <h1>Visual Foundations</h1>
        <div className="sub">Data Science Curriculum</div>
        <nav>
          <button className={page === "intro" ? "active" : ""} onClick={() => setPage("intro")}>Introduction</button>
          <div className="section-label">Curriculum</div>
          {PAGES.map((p) => (
            <button key={p.id} className={page === p.id ? "active" : ""} onClick={() => setPage(p.id)}>{p.label}</button>
          ))}
          <div className="section-label">Reference</div>
          <button className={page === "interview" ? "active" : ""} onClick={() => setPage("interview")}>Interview Prep</button>
        </nav>
      </div>
      <div className="main">
        {page === "intro" && <Intro onStart={() => setPage("naive-bayes")} />}
        {page === "naive-bayes" && <NaiveBayes />}
        {page === "evaluation" && <ModelEvaluation />}
        {page === "calculus" && <Calculus />}
        {page === "backprop" && <Backprop />}
        {page === "interview" && <InterviewPrep />}
      </div>
    </div>
  );
}
