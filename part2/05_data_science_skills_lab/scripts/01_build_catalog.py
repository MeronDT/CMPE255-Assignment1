"""Builds a full catalog of all 46 installed skills by parsing their real
SKILL.md YAML frontmatter -- not a hand-typed list that could drift from what's
actually installed.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / ".claude" / "skills"
DOCS = ROOT / "docs" / "eda"
DOCS.mkdir(parents=True, exist_ok=True)

# Which source repo each skill came from, and the CRISP-DM phase it best maps to
# -- both are structural facts about the installed files (a skill's containing
# repo) or a categorization judgment call, not invented per-skill descriptions
# (those are read from each skill's own frontmatter below).
PARAM087_SKILLS = {
    "data-cleaning", "experiment-tracking", "exploratory-data-analysis", "feature-engineering",
    "hyperparameter-tuning", "imbalanced-data", "llm-finetuning", "ml-debugging", "model-evaluation",
    "model-serving", "pandas-patterns", "pytorch-training-loop", "rag-pipeline", "reproducible-ml",
    "sklearn-pipelines",
}

PHASE_MAP = {
    "programmatic-eda": "Data Understanding", "exploratory-data-analysis": "Data Understanding",
    "data-catalog-entry": "Data Understanding", "schema-mapper": "Data Understanding",
    "data-cleaning": "Data Preparation", "feature-engineering": "Data Preparation",
    "data-quality-audit": "Data Preparation", "query-validation": "Data Preparation",
    "sklearn-pipelines": "Modeling", "hyperparameter-tuning": "Modeling", "pytorch-training-loop": "Modeling",
    "llm-finetuning": "Modeling", "rag-pipeline": "Modeling", "reproducible-ml": "Modeling",
    "experiment-tracking": "Modeling", "pandas-patterns": "Modeling",
    "model-evaluation": "Evaluation", "imbalanced-data": "Evaluation", "ml-debugging": "Evaluation",
    "ab-test-analysis": "Evaluation", "peer-review-template": "Evaluation", "analysis-qa-checklist": "Evaluation",
    "model-serving": "Deployment",
}
DEFAULT_PHASE = "Business Analysis & Communication"

DEMONSTRATED = {
    "programmatic-eda", "exploratory-data-analysis", "data-cleaning", "feature-engineering",
    "sklearn-pipelines", "hyperparameter-tuning", "model-evaluation", "imbalanced-data",
    "segmentation-analysis", "cohort-analysis", "business-metrics-calculator", "ab-test-analysis",
    "time-series-analysis",
}


def parse_frontmatter(text: str) -> dict:
    m = re.match(r"^---\s*[\r\n]([\s\S]*?)[\r\n]---", text)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm


def main():
    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
        fm = parse_frontmatter(text)
        name = skill_dir.name
        source = "param087/agent-ml-skills" if name in PARAM087_SKILLS else "nimrodfisher/data-analytics-skills"
        skills.append({
            "name": name,
            "description": fm.get("description", "").strip('"'),
            "source_repo": source,
            "crispdm_phase": PHASE_MAP.get(name, DEFAULT_PHASE),
            "live_demonstrated": name in DEMONSTRATED,
        })

    by_source = {}
    by_phase = {}
    for s in skills:
        by_source[s["source_repo"]] = by_source.get(s["source_repo"], 0) + 1
        by_phase[s["crispdm_phase"]] = by_phase.get(s["crispdm_phase"], 0) + 1

    catalog = {
        "n_total_skills": len(skills),
        "n_live_demonstrated": sum(1 for s in skills if s["live_demonstrated"]),
        "by_source_repo": by_source,
        "by_crispdm_phase": by_phase,
        "skills": skills,
    }
    with open(DOCS / "skill_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Cataloged {len(skills)} skills ({catalog['n_live_demonstrated']} live-demonstrated)")
    print(json.dumps(by_source, indent=2))
    print(json.dumps(by_phase, indent=2))


if __name__ == "__main__":
    main()
