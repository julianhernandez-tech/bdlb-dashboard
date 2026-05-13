# BDLB Run Dashboard

Monitoring and control dashboard for **BDLB (Backward-Design Lesson Builder)** — a multi-agent AI pipeline that turns a STAAR-style math question image into a full backward-designed lesson (60 authored items, tier explanations, teaching article, final `lesson.html` bundle).

The pipeline runs 16 specialized agents across 7 phases. This dashboard makes those runs visible, inspectable, and editable — without touching the pipeline itself.

---

## Build status

| Step | Scope | Status |
|------|------------------------------------------|--------|
| 1    | UI shell — tabs, sidebar, static flowchart, placeholders | ✅ Done |
| 2    | GitHub connection — PAT input, repo auth, run discovery | ⏳ Next |
| 3    | Artifact downloader — pull run files into `runs/{run_id}/` | ⏳ |
| 4    | Live flowchart — color nodes by `state.json` phase status | ⏳ |
| 5    | Readable cards — per-phase output (seed, plan, items, article, QC) | ⏳ |
| 6    | Cost reporter — parse `events.jsonl` → $ per agent / phase / run | ⏳ |
| 7    | Editor + push — file browser, edit any artifact, commit via GitHub API | ⏳ |

---

## Quickstart

```bash
git clone https://github.com/julianhernandez-tech/bdlb-dashboard.git
cd bdlb-dashboard

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

### What you should see (Step 1)

- **Sidebar** — Run Selector (`No runs loaded`), GitHub status (`● Not connected`), About blurb
- **🗺️ Workflow tab** — 7-node Mermaid flowchart (Preflight → Seed Extraction → Backward Design → Tier Specification → Item Authoring → Article Authoring → Assembly), all neutral grey + placeholder preview card
- **💰 Costs tab** — Four `—` metric cards, empty dataframe with 6 columns, info caption
- **✏️ Editor tab** — File selector + Pull/Push buttons (disabled), code block + Save button (disabled)

No API calls are made in Step 1. The app is pure static UI.

---

## Pipeline source repo

The BDLB pipeline itself lives at <https://github.com/julianhernandez-tech/bdlb>. Once Step 2 lands, this dashboard will read runs from there:

- `bdlb/agents/` — 16 agent spec markdown files
- `bdlb/schemas/` — 9 JSON schemas
- `bdlb/runs/{run_id}/` — one folder per lesson run
- `bdlb/build_state.json` — phase tracker
- `bdlb/build_events.jsonl` — append-only event log
- `bdlb/consistency_report.json` — cross-spec invariant check

---

## Tech stack

Python · Streamlit · streamlit-mermaid · PyGithub · streamlit-ace · pandas
