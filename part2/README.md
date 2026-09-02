# CMPE 255 — HW1 Part 2: Do Data Science with a Chat or Coding Assistance

A collection of full-stack applications built by replicating assignment prompts through an AI coding agent (Claude Code), to observe and compare how agentic tools translate a prompt into a working system.

---

## 📁 Projects Index

| # | Project & Directory | Stack | Backend Port | Frontend Port |
|---|---|---|:---:|:---:|
| **0** | [**Flow — Dynamic Todo Workspace**](./00_dynamic_todo_workspace) | React 19 + TS + Vite + Tailwind / Express + `node:sqlite` | `4000` | `5173` |
| **1** | [**NYC Taxi Trip Duration & Fare Prediction**](./01_nyc_taxi_trip_prediction) | CRISP-DM · Python/LightGBM / FastAPI / React + Leaflet + Recharts | `8001` | `5174` |
| **2** | [**NanoLlama — Autoregressive SFT LLM**](./02_nano_llm_transformer) | CRISP-DM · PyTorch (RoPE/RMSNorm/SwiGLU) / FastAPI / React + Recharts | `8002` | `5175` |
| **3** | [**Customer Intelligence & Segmentation Clustering**](./03_customer_segmentation_clustering) | CRISP-DM · scikit-learn (K-Means) + AutoResearch / FastAPI / React + Recharts | `8003` | `5176` |
| **4** | [**Market Basket Pattern Mining**](./04_market_basket_pattern_mining) | CRISP-DM · mlxtend (Apriori/FP-Growth) + AutoResearch / FastAPI / React + Recharts | `8004` | `5177` |
| **5** | [**Data Science Skills Mastery Lab**](./05_data_science_skills_lab) | 46 vetted third-party Claude Code skills, live-executed / FastAPI / React + Recharts | `8005` | `5185` |
| **6** | [**Autonomous Anomaly Detection Platform**](./06_anomaly_detection_platform) | CRISP-DM · Isolation Forest/LOF/OC-SVM + AutoResearch / FastAPI / React + Recharts | `8006` | `5178` |
| **7** | [**AutoGluon Multi-Layer Stacking Platform**](./07_autogluon_stacking_platform) | CRISP-DM · AutoGluon Tabular (regression + classification) / FastAPI / React + Recharts | `8007` | `5179` |
| **8** | [**Data Science Visual Foundations Curriculum**](./08_visual_foundations_curriculum) | Client-side only · React + KaTeX + inline SVG live simulations, GitHub-Pages-ready | — | `5180` |
| **9** | [**FlowForge DAG Engine**](./09_flowforge_dag_engine) | 25 vetted `mattpocock/skills`-informed TypeScript patterns · Express + WebSocket / React + custom SVG DAG viz | `8009` | `5189` |
| **10** | [**CRISP-DM Master's Data Science Platform**](./10_crispdm_masters_platform) | CRISP-DM capstone · clustering + anomaly detection + supervised ML + association rules + LSH / FastAPI / React + Recharts | `8010` | `5182` |
| **11** | [**Enterprise Data Science Audit Platform**](./11_enterprise_audit_platform) | Evidence-backed audit of every project above / FastAPI / React + Recharts | `8011` | `5184` |
| **12** | [**Time Series Forecasting Engine**](./12_time_series_forecasting_engine) | CRISP-DM · Prophet/SARIMA/Holt-Winters + AutoResearch / FastAPI / React + Recharts | `8012` | `5183` |

Projects 3, 4, 6, 7, 10, 12 share the CRISP-DM + admin-dashboard + AutoResearch pattern established in Projects 1–2; each project's own `README.md` explains its specific dataset, findings, and any honest negative/surprising results (all of them have at least one — see each `RESEARCH_REPORT.md`). Project 11 audits all of them with real, evidence-backed checks rather than a subjective writeup.

---

## 🧠 Implementation Plans

Per-project architecture, data model, key technical decisions, and verification/acceptance criteria are cataloged in [`IMPLEMENTATION_PLANS.md`](./IMPLEMENTATION_PLANS.md), linking out to each project's own `DESIGN_DOC.md`.

## 🚀 Running a Project

Each project is self-contained under its own directory, with its own `README.md`. Projects 0 is pure Node (`client/` + `server/`); Projects 1–7 and 9–12 pair a Python data/ML pipeline (`scripts/`) with a FastAPI `backend/` and a React `frontend/`; Project 8 is a static client-side site with no backend. Navigate into a project and follow its README's Quick Start, e.g.:

```bash
cd 03_customer_segmentation_clustering
python scripts/01_eda.py && python scripts/02_prepare_data.py && python scripts/03_train_models.py
python scripts/04_autoresearch.py && python scripts/05_evaluate.py

cd backend && python -m uvicorn app.main:app --port 8003    # backend
cd ../frontend && npm install && npm run dev -- --port 5176  # frontend
```

Every project's exact run steps are in its own `README.md`; ports above match what each project's scripts/backend/frontend actually use, so multiple projects can run simultaneously without colliding.

Project 2's trained checkpoints are committed to the repo, so serving it needs no GPU; reproducing its training from scratch does (tested on 4GB VRAM). Projects 3/4/10/12 share the UCI/Kaggle Online Retail dataset (reused deliberately across different mining tasks, not duplicated); the large raw dataset files are gitignored and re-downloadable via each project's `scripts/01_*.py` (no Kaggle credentials required — sourced from each dataset's original public host).

## 🔐 Third-Party Skill Installs (Projects 5 & 9)

Projects 5 and 9 install skills from third-party GitHub repositories
(`param087/agent-ml-skills`, `nimrodfisher/data-analytics-skills`,
`mattpocock/skills`). Each was vetted before installation — install-mechanism
review, license check, and cross-referencing dataset/skill *names* (never code)
against the course's reference repository — documented in each project's own
`docs/00_skill_vetting.md`.
