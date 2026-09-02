# Project Scope

## What this is

**Data Science Visual Foundations Curriculum** — an interactive educational site teaching
four foundational data science / ML concepts with deep intuition, rigorous math (rendered
via KaTeX), and live simulations the learner directly manipulates, not just static
diagrams. Unlike Projects 01–07, this is not a data-mining project and does not follow
the CRISP-DM lifecycle — it's a teaching tool, scoped and structured accordingly.

## Topics Covered

1. **Naive Bayes** — Bayes' theorem derivation, the conditional-independence assumption,
   a live one-word spam classifier with adjustable priors/likelihoods.
2. **Model Evaluation** — confusion matrix, Type I/II errors, ROC-AUC (with a live ROC
   curve computed from a synthetic overlapping-distribution dataset), cost-matrix
   business-cost minimization, and the precision/recall tradeoff — all driven by one
   draggable decision-threshold slider.
3. **Differential Calculus & Gradient Descent** — what a derivative means geometrically
   (live tangent-line demo), then a runnable/steppable gradient-descent simulation on a
   quadratic loss surface, including a deliberately-breakable "too-high learning rate"
   case to show divergence, not just convergence.
4. **Chain Rule & Backpropagation** — the chain rule stated formally, then traced through
   a single neuron's full forward and backward pass (every intermediate value and every
   local gradient shown explicitly), to make concrete how backprop is just the chain rule
   applied systematically with intermediate results reused.

Each topic ends with a 3-4 question quiz (immediate feedback, explanation on every
answer). A separate **Interview Prep** page collects common data science interview
questions and model answers across all four topics.

## Why no backend / no CRISP-DM framework here

This project teaches concepts, not a dataset-driven business problem — there's no data
to mine, clean, or model, so a CRISP-DM writeup and a FastAPI backend would be
performative rather than useful. Everything runs client-side (React state), including
the "live simulations," which are pure math computed in the browser.

## GitHub Pages Deployment

Built with Vite (`base: './'` for relative-path portability under any GitHub Pages
project path). `npm run build` produces a static `dist/` directory deployable as-is —
verified locally by serving `dist/` and confirming 0 console errors. A GitHub Actions
workflow (`.github/workflows/deploy-curriculum.yml`, repo root) builds and publishes
this project's `dist/` to GitHub Pages on pushes to `main` that touch this directory.
