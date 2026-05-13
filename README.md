# BDLB Run Dashboard

Monitoring and control dashboard for **BDLB (Backward-Design Lesson Builder)** — a multi-agent AI pipeline that turns a STAAR-style math question image into a full backward-designed lesson (60 authored items, tier explanations, teaching article, final `lesson.html` bundle).

This dashboard reads runs from the [`bdlb`](https://github.com/julianhernandez-tech/bdlb) pipeline repo, visualizes each phase's status, parses cost data, and lets you edit any artifact and push the change back to GitHub.

---

## Live URL

Once deployed on Streamlit Community Cloud:
**https://bdlb-dashboard-julian.streamlit.app** (or whatever subdomain you chose)

## What it does

- **🗺️ Workflow** — 7-node Mermaid flowchart colored by phase status from `runs/{run_id}/build_state.json` (green = completed, yellow = in progress, red = failed, grey = pending). Click any phase to expand its key artifacts: seed image, lesson plan, items, article drafts, QC reports, final `lesson.html`.
- **💰 Costs** — Parses `runs/{run_id}/build_events.jsonl`, maps `model + tokens_in + tokens_out` to dollar cost using a built-in pricing table, and shows per-agent and per-phase breakdowns.
- **✏️ Editor** — Browse every file in the active run, edit in-page (Ace editor with JSON/Markdown/HTML/etc. syntax highlighting), commit the change back to `julianhernandez-tech/bdlb` with one click.

## Pipeline phases

```
P0 Preflight → P1 Seed Extraction → P2 Backward Design → P3 Tier Specification
→ P4 Item Authoring → P5 Article Authoring → P6 Assembly
```

## Setup on Streamlit Cloud

1. The dashboard is already deployed — see live URL above.
2. **Add your GitHub PAT as a secret** so the dashboard can read the private pipeline repo and push edits:
   - Generate a fine-grained PAT at <https://github.com/settings/personal-access-tokens/new>
     - Repository access: `julianhernandez-tech/bdlb` (and this dashboard repo if private)
     - Repository permissions: **Contents → Read and write**, **Metadata → Read**
     - Expiration: whatever you prefer
   - Copy the token (starts with `github_pat_…`)
   - Go to your Streamlit Cloud app → **⋮ menu → Settings → Secrets**
   - Paste:
     ```toml
     GITHUB_TOKEN = "github_pat_xxxxxxxxxxxxxxxxxxxx"
     ```
   - Save. The app reboots in ~10s and the sidebar dot turns green: "● Connected as julianhernandez-tech".
3. As soon as the pipeline writes its first run folder (`bdlb/runs/<run_id>/build_state.json`), it appears in the Run Selector dropdown.

If you don't want to store a secret, you can also paste a token into the sidebar's "Paste a token manually" expander for a session-only override.

## Local development

```bash
git clone https://github.com/julianhernandez-tech/bdlb-dashboard.git
cd bdlb-dashboard

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Optional: export your PAT for local testing
export GITHUB_TOKEN=github_pat_xxxxxxxxxxxx

streamlit run app.py
```

App opens at <http://localhost:8501>.

---

## Pipeline repo layout (read by this dashboard)

```
bdlb/
├── agents/                       # 16 agent spec markdown files
├── schemas/                      # 9 JSON schemas
├── runs/
│   └── {run_id}/
│       ├── build_state.json      # phase tracker — read by Workflow tab
│       ├── build_events.jsonl    # event log — read by Costs tab
│       ├── consistency_report.json
│       ├── seed/                 # P1 outputs
│       ├── lesson_plan/          # P2 outputs
│       ├── tier_spec/            # P3 outputs
│       ├── items/                # P4 outputs (60 authored items)
│       ├── article/              # P5 outputs (teaching article + drafts)
│       └── lesson.html           # P6 final bundle
```

### Accepted `build_state.json` shapes

The dashboard handles any of these shapes (so you can iterate on the pipeline's state schema without breaking the UI):

```jsonc
// Shape A — explicit per-phase dict
{ "phases": { "P0": {"status": "completed"}, "P1": {"status": "in_progress"}, ... } }

// Shape B — completed list + current pointer
{ "phases_completed": ["P0", "P1"], "current_phase": "P2", "failed_phases": [] }

// Shape C — single-phase pointer
{ "phase": "P3", "status": "in_progress" }
```

### Accepted `build_events.jsonl` shapes

Any line containing both a `model` field and at least one of (`tokens_in`/`input_tokens`/`usage.input_tokens`) and (`tokens_out`/`output_tokens`/`usage.output_tokens`) will be costed.

---

## Build status

| Step | Scope | Status |
|------|------------------------------------------|--------|
| 1    | UI shell — tabs, sidebar, static flowchart, placeholders | ✅ |
| 2    | GitHub connection — secret + manual-token fallback, repo handshake | ✅ |
| 3    | Run discovery + artifact downloader | ✅ |
| 4    | Live flowchart coloring from `build_state.json` | ✅ |
| 5    | Per-phase readable output cards | ✅ |
| 6    | Cost reporter (`events.jsonl` → $ per agent/phase/run) | ✅ |
| 7    | Editor: file browser, edit, commit back to `bdlb` repo | ✅ |

---

## Tech stack

Python · Streamlit · streamlit-mermaid · streamlit-ace · PyGithub · pandas
