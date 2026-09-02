# Data Science Visual Foundations Curriculum

An interactive educational site teaching four foundational data science / ML concepts —
Naive Bayes, model evaluation, differential calculus & gradient descent, and the chain
rule & backpropagation — with deep intuition, rigorous math (KaTeX-rendered), live
manipulable simulations, per-topic quizzes, and an interview-prep reference page.

## Running it locally

```bash
npm install
npm run dev       # dev server, http://localhost:5173 by default
npm run build     # production build -> dist/, GitHub-Pages-ready (relative asset paths)
```

## GitHub Pages

A workflow (`.github/workflows/deploy-curriculum.yml` at the repo root) builds and
publishes this project's `dist/` to GitHub Pages automatically on pushes to `main`
that touch this directory.

**Note on repo privacy**: this repository is intentionally kept private. GitHub Pages
on a *private* repository requires GitHub Pro/Team/Enterprise — on the free tier,
Pages either isn't available for private repos or would require making the repo (or
this specific Pages site) public. The workflow above is ready to run the moment Pages
is enabled (Settings → Pages → Source: GitHub Actions) — that's a manual step for
whoever owns the repo to decide on, not something done automatically here, since it
touches the repo's visibility/access settings.

Until then, `npm run build && npm run preview` (or serving `dist/` with any static
file server) reproduces the exact deployed output locally.

Full project scope in [docs/01_project_scope.md](./docs/01_project_scope.md).
