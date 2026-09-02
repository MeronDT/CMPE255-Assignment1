# Design Doc — Data Science Visual Foundations Curriculum

## 1. Overview

A client-side-only educational React site. No backend, no dataset, no CRISP-DM
lifecycle — see [docs/01_project_scope.md](./docs/01_project_scope.md) for why that's
a deliberate scope decision, not an oversight.

## 2. Architecture

- **Vite + React + TypeScript**, `base: './'` in `vite.config.ts` for GitHub-Pages
  path portability.
- **KaTeX** (`react-katex`) for math rendering — chosen over plain unicode/monospace
  formulas since the brief explicitly asked for "rigorous math," and KaTeX renders
  proper mathematical typesetting (fractions, summations, arg max, etc.) rather than
  an approximation.
- **Recharts** reused from Projects 03/04/06/07 for the ROC curve (the one chart that
  benefits from a real charting library; everything else — the derivative tangent line,
  the gradient descent animation — is hand-drawn inline SVG for full control over the
  live-simulation interaction).
- **No routing library** — four topic pages plus an intro and interview-prep page,
  handled with simple `useState` page switching in `App.tsx`. A router would be
  over-engineering for six static views with no deep-linking requirement.

## 3. Component Structure

```
src/
  App.tsx                 -- sidebar nav + page switch
  components/
    Quiz.tsx               -- reusable quiz component (question, options, explanation)
    NaiveBayes.tsx          -- Bayes' theorem + live spam-classifier simulation
    ModelEvaluation.tsx     -- confusion matrix + ROC curve + cost matrix, one shared threshold
    Calculus.tsx            -- derivative/tangent-line demo + gradient descent simulation
    Backprop.tsx             -- chain rule + single-neuron forward/backward pass trace
    InterviewPrep.tsx        -- static Q&A reference, all four topics
```

## 4. Live Simulations: Design Notes

- **Naive Bayes**: fully parametric (prior, two likelihoods, evidence toggle) — every
  control feeds directly into Bayes' theorem computed live, no pre-baked lookup table.
- **Model Evaluation**: a single synthetic 1,000-point dataset (500 true positive /
  500 true negative scores, overlapping Gaussians, seeded PRNG for reproducibility)
  drives the confusion matrix, precision/recall, ROC curve, and cost calculator from
  one shared threshold slider — deliberately, so a learner sees all four evaluation
  concepts respond to the *same* underlying change, not four disconnected demos.
- **Gradient Descent**: a genuine iterative simulation (not an animation faked with
  CSS) — each "Step" click computes the real gradient at the current point and applies
  the real update rule; "Run" auto-steps on an interval; the learning-rate slider lets
  a learner directly cause and observe divergence, which is the actual pedagogical
  point (not just showing successful convergence).
- **Backpropagation**: traces one neuron's forward pass (z, a, L) and backward pass
  (every chain-rule link: ∂L/∂a, ∂a/∂z, ∂L/∂z, ∂L/∂w) as literal displayed numbers that
  recompute live as weight/bias sliders move — chosen over a bigger network diagram so
  every single intermediate value stays visible and traceable at once.

## 5. Verification

Built and served the production `dist/` output standalone (not just the dev server) to
confirm GitHub-Pages readiness — 0 console errors. Playwright-verified all six pages
(Introduction + 4 topics + Interview Prep) plus direct interaction (moving a simulation
slider, answering a quiz question) with 0 console errors and 0 failed requests.
