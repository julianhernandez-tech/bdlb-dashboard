"""
BDLB Run Dashboard — generation engine + monitoring

Monitoring and control dashboard for BDLB (Backward-Design Lesson Builder),
a multi-agent AI pipeline that turns a STAAR-style math question image into a
full backward-designed lesson.

Pipeline source repo:   julianhernandez-tech/bdlb
Dashboard repo:         julianhernandez-tech/bdlb-dashboard

Authentication:
  Reads a GitHub Personal Access Token from st.secrets["GITHUB_TOKEN"]
  (configured in Streamlit Cloud → Settings → Secrets), or falls back to the
  GITHUB_TOKEN env var for local development.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_mermaid import st_mermaid

try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except Exception:
    ACE_AVAILABLE = False

from github import Github, GithubException, UnknownObjectException

import engine  # generation engine: orchestrator loop + provider adapters

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIPELINE_REPO = "julianhernandez-tech/bdlb"
DASHBOARD_VERSION = "v0.8"

# The 7 lesson-build phases (P0–P6) the dashboard visualizes.
# These are the per-lesson pipeline phases, distinct from the B0–B4 phases
# that built the pipeline itself.
PHASES: list[tuple[str, str, str]] = [
    ("P0", "Preflight",          "preflight"),
    ("P1", "Seed Extraction",    "seed_extraction"),
    ("P2", "Backward Design",    "backward_design"),
    ("P3", "Tier Specification", "tier_specification"),
    ("P4", "Item Authoring",     "item_authoring"),
    ("P5", "Article Authoring",  "article_authoring"),
    ("P6", "Assembly",           "assembly"),
]
PHASE_KEY_BY_ID = {pid: key for pid, _, key in PHASES}
PHASE_LABEL_BY_ID = {pid: label for pid, label, _ in PHASES}

# Model pricing in USD per 1M tokens (input, output).
# Used by the cost reporter to convert events.jsonl token counts to dollars.
# Update freely as new models are added to the pipeline.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4":           (15.00, 75.00),
    "claude-sonnet-4":         ( 3.00, 15.00),
    "claude-haiku-4":          ( 0.80,  4.00),
    "claude-3-7-sonnet":       ( 3.00, 15.00),
    "claude-3-5-sonnet":       ( 3.00, 15.00),
    "claude-3-5-haiku":        ( 0.80,  4.00),
    "gpt-5":                   ( 5.00, 15.00),
    "gpt-4o":                  ( 2.50, 10.00),
    "gpt-4o-mini":             ( 0.15,  0.60),
    "o1":                      (15.00, 60.00),
    "o3-mini":                 ( 1.10,  4.40),
    "gemini-2.5-pro":          ( 1.25, 10.00),
    "gemini-2.0-flash":        ( 0.10,  0.40),
}
DEFAULT_PRICING = (3.00, 15.00)  # Fallback if model name doesn't match

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BDLB Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# GitHub client
# ---------------------------------------------------------------------------
def get_github_token() -> str | None:
    """Resolve a GitHub PAT from Streamlit secrets, env var, or session state."""
    # 1. Streamlit Cloud secrets (production)
    try:
        token = st.secrets.get("GITHUB_TOKEN")  # type: ignore[attr-defined]
        if token:
            return str(token).strip()
    except Exception:
        pass
    # 2. Environment variable (local dev)
    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token:
        return env_token.strip()
    # 3. Session-state override (user pasted into UI)
    return st.session_state.get("manual_token")


@st.cache_resource(show_spinner=False)
def _github_client(token: str) -> Github:
    return Github(token, per_page=100)


def github_client() -> Github | None:
    token = get_github_token()
    if not token:
        return None
    return _github_client(token)


@st.cache_data(ttl=60, show_spinner=False)
def gh_check_connection(token_fingerprint: str) -> dict[str, Any]:
    """Verify the token works and return basic user info. Cached by token hash."""
    client = github_client()
    if client is None:
        return {"ok": False, "error": "No token provided"}
    try:
        user = client.get_user()
        login = user.login
        # Also verify access to the pipeline repo
        try:
            repo = client.get_repo(PIPELINE_REPO)
            repo_ok = True
            repo_private = repo.private
        except UnknownObjectException:
            repo_ok = False
            repo_private = None
        return {
            "ok": True,
            "login": login,
            "repo_accessible": repo_ok,
            "repo_private": repo_private,
        }
    except GithubException as e:
        return {"ok": False, "error": f"GitHub error: {e.data.get('message', str(e))}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _token_fingerprint() -> str:
    """Short stable hash of the token for cache keying without leaking it."""
    t = get_github_token()
    if not t:
        return "none"
    return f"len{len(t)}_{t[:4]}_{t[-4:]}"


# ---------------------------------------------------------------------------
# Run discovery & artifact loading (Steps 3 + 4 + 5)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def list_runs(token_fingerprint: str) -> list[str]:
    """List run_id folder names under bdlb/runs/. Empty list if no runs/ dir yet."""
    client = github_client()
    if client is None:
        return []
    try:
        repo = client.get_repo(PIPELINE_REPO)
        contents = repo.get_contents("runs")
        if not isinstance(contents, list):
            contents = [contents]
        return sorted(
            [c.name for c in contents if c.type == "dir"],
            reverse=True,  # newest first if names are timestamped
        )
    except UnknownObjectException:
        return []
    except GithubException:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def list_run_files(token_fingerprint: str, run_id: str) -> list[dict[str, Any]]:
    """Recursively list every file in runs/{run_id}/."""
    client = github_client()
    if client is None:
        return []
    try:
        repo = client.get_repo(PIPELINE_REPO)
    except GithubException:
        return []

    out: list[dict[str, Any]] = []

    def walk(path: str) -> None:
        try:
            items = repo.get_contents(path)
        except UnknownObjectException:
            return
        if not isinstance(items, list):
            items = [items]
        for it in items:
            if it.type == "dir":
                walk(it.path)
            else:
                out.append({
                    "path": it.path,
                    "name": it.name,
                    "size": it.size,
                    "sha": it.sha,
                    "download_url": it.download_url,
                })

    walk(f"runs/{run_id}")
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_file_text(token_fingerprint: str, path: str) -> tuple[str | None, str | None]:
    """Return (text_content, sha) for a file path in the pipeline repo."""
    client = github_client()
    if client is None:
        return None, None
    try:
        repo = client.get_repo(PIPELINE_REPO)
        f = repo.get_contents(path)
        if isinstance(f, list):
            return None, None
        raw = base64.b64decode(f.content)
        try:
            return raw.decode("utf-8"), f.sha
        except UnicodeDecodeError:
            return None, f.sha  # binary
    except (UnknownObjectException, GithubException):
        return None, None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_file_binary(token_fingerprint: str, download_url: str) -> bytes | None:
    """Fetch raw bytes for binary files (images, etc.) via download_url."""
    import requests
    try:
        r = requests.get(download_url, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def parse_state(run_id: str) -> dict[str, Any]:
    """Read runs/{run_id}/build_state.json. Returns {} if missing."""
    text, _ = fetch_file_text(_token_fingerprint(), f"runs/{run_id}/build_state.json")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def parse_events(run_id: str) -> list[dict[str, Any]]:
    """Read runs/{run_id}/build_events.jsonl as a list of dicts."""
    text, _ = fetch_file_text(_token_fingerprint(), f"runs/{run_id}/build_events.jsonl")
    if not text:
        return []
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def phase_statuses(state: dict[str, Any]) -> dict[str, str]:
    """
    Derive a {phase_id: status} mapping from build_state.json.

    Status values: completed | in_progress | failed | pending

    The dashboard accepts several state.json shapes:
      - {"phases": {"P0": {"status": "completed"}, ...}}
      - {"phases_completed": ["P0", "P1"], "current_phase": "P2", "failed_phases": []}
      - {"phase": "P3", "status": "in_progress"}  (single-phase pointer)
    """
    statuses: dict[str, str] = {pid: "pending" for pid, _, _ in PHASES}
    if not state:
        return statuses

    # Shape A: explicit per-phase dict
    phases_obj = state.get("phases")
    if isinstance(phases_obj, dict):
        for pid in statuses:
            entry = phases_obj.get(pid) or phases_obj.get(PHASE_KEY_BY_ID[pid])
            if isinstance(entry, dict):
                s = str(entry.get("status", "")).lower()
                if s in {"completed", "done", "success"}:
                    statuses[pid] = "completed"
                elif s in {"in_progress", "running", "active"}:
                    statuses[pid] = "in_progress"
                elif s in {"failed", "error"}:
                    statuses[pid] = "failed"
            elif isinstance(entry, str):
                s = entry.lower()
                if s in {"completed", "done", "success"}:
                    statuses[pid] = "completed"

    # Shape B: phases_completed list + current_phase pointer
    completed = state.get("phases_completed") or []
    for pid in completed:
        if pid in statuses:
            statuses[pid] = "completed"
    failed_list = state.get("specs_failed") or state.get("failed_phases") or []
    for pid in failed_list:
        if pid in statuses:
            statuses[pid] = "failed"
    current = state.get("current_phase") or state.get("phase")
    if current in statuses and statuses[current] == "pending":
        statuses[current] = "in_progress"

    return statuses


# ---------------------------------------------------------------------------
# Cost reporter (Step 6)
# ---------------------------------------------------------------------------
def model_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return USD cost for a single LLM call."""
    key = (model or "").lower().strip()
    # Match by substring so "claude-3-5-sonnet-20241022" still resolves
    pricing = None
    for known, p in MODEL_PRICING.items():
        if known in key:
            pricing = p
            break
    if pricing is None:
        pricing = DEFAULT_PRICING
    p_in, p_out = pricing
    return (tokens_in / 1_000_000) * p_in + (tokens_out / 1_000_000) * p_out


def build_cost_table(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate token-usage events from events.jsonl into a per-call cost table."""
    rows = []
    for ev in events:
        # Accept multiple event shapes for token usage.
        # Recognized: event in {"agent_call", "llm_call", "token_usage"} OR
        # presence of tokens_in / input_tokens fields.
        toks_in = (
            ev.get("tokens_in")
            or ev.get("input_tokens")
            or (ev.get("usage", {}) or {}).get("input_tokens")
            or 0
        )
        toks_out = (
            ev.get("tokens_out")
            or ev.get("output_tokens")
            or (ev.get("usage", {}) or {}).get("output_tokens")
            or 0
        )
        if not toks_in and not toks_out:
            continue
        rows.append({
            "Phase":     ev.get("phase", "—"),
            "Agent":     ev.get("agent", ev.get("event", "—")),
            "Model":     ev.get("model", "—"),
            "Tokens In":  int(toks_in),
            "Tokens Out": int(toks_out),
            "Cost ($)":  round(model_cost(ev.get("model", ""), int(toks_in), int(toks_out)), 4),
        })
    if not rows:
        return pd.DataFrame(columns=["Phase", "Agent", "Model", "Tokens In", "Tokens Out", "Cost ($)"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Run Selector")

    conn = gh_check_connection(_token_fingerprint())
    connected = bool(conn.get("ok"))

    if connected:
        runs = list_runs(_token_fingerprint())
    else:
        runs = []

    if runs:
        active_run = st.selectbox("Active Run", runs, key="active_run")
    else:
        placeholder = "No runs found in bdlb/runs/" if connected else "No runs loaded"
        st.selectbox("Active Run", [placeholder], disabled=True)
        active_run = None

    if st.button("🔄 Refresh runs", use_container_width=True, disabled=not connected):
        list_runs.clear()
        list_run_files.clear()
        fetch_file_text.clear()
        st.rerun()

    st.divider()

    st.markdown("### 🔗 GitHub")
    if connected:
        repo_note = " · repo ✓" if conn.get("repo_accessible") else " · repo ✗"
        st.markdown(
            f"<span style='color:#2ea043;font-size:18px;'>●</span> "
            f"<span style='color:#2ea043;'>Connected as <b>{conn.get('login')}</b>{repo_note}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='color:#888;font-size:18px;'>●</span> "
            "<span style='color:#888;'>Not connected</span>",
            unsafe_allow_html=True,
        )
        if conn.get("error") and get_github_token():
            st.caption(f"⚠️ {conn['error']}")

    with st.expander("Paste a token manually" if not connected else "Replace token", expanded=False):
        st.caption(
            "Tokens are read from Streamlit Cloud secrets as `GITHUB_TOKEN`. "
            "You can also paste one here for a session-only override."
        )
        new_tok = st.text_input("GitHub PAT", type="password", key="pat_input",
                                placeholder="ghp_… or github_pat_…")
        c1, c2 = st.columns(2)
        if c1.button("Save", use_container_width=True):
            if new_tok.strip():
                st.session_state["manual_token"] = new_tok.strip()
                gh_check_connection.clear()
                _github_client.clear()
                st.rerun()
        if c2.button("Clear", use_container_width=True):
            st.session_state.pop("manual_token", None)
            gh_check_connection.clear()
            _github_client.clear()
            st.rerun()

    st.divider()

    st.markdown("### ℹ️ About")
    st.caption(
        f"BDLB Run Dashboard {DASHBOARD_VERSION} — monitors Backward-Design "
        f"Lesson Builder pipeline runs stored in `{PIPELINE_REPO}`."
    )


# ---------------------------------------------------------------------------
# Phase output card (used in Tab 1)
# ---------------------------------------------------------------------------
def render_phase_card(run_id: str, phase_id: str, status: str) -> None:
    """Render a readable card showing key artifacts for the chosen phase."""
    files = list_run_files(_token_fingerprint(), run_id)
    if not files:
        st.warning("No files found in this run folder yet.")
        return

    phase_key = PHASE_KEY_BY_ID[phase_id].lower()
    relevant = [
        f for f in files
        if phase_key in f["path"].lower() or f"/{phase_id.lower()}/" in f["path"].lower()
    ]

    spotlight_by_phase = {
        "P1": ["seed_image", "seed_question", "seed.json"],
        "P2": ["lesson_plan", "backward_design"],
        "P3": ["tier_spec"],
        "P4": ["items", "item_"],
        "P5": ["article", "article_draft"],
        "P6": ["lesson.html", "lesson_assembly"],
    }
    spotlight_keys = spotlight_by_phase.get(phase_id, [])
    spotlights = [
        f for f in files
        if any(k in f["name"].lower() or k in f["path"].lower() for k in spotlight_keys)
    ]

    badge = {
        "completed": "🟢", "in_progress": "🟡", "failed": "🔴", "pending": "⚪",
    }[status]
    st.markdown(f"##### {badge} {phase_id} — {PHASE_LABEL_BY_ID[phase_id]}")

    if not relevant and not spotlights:
        st.caption("No artifacts found for this phase yet.")
        return

    for f in (spotlights or relevant[:5]):
        with st.expander(f"📄 {f['path']}  ·  {f['size']:,} bytes", expanded=False):
            if f["name"].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                img = fetch_file_binary(_token_fingerprint(), f["download_url"])
                if img:
                    st.image(img, use_column_width=True)
                else:
                    st.caption("Could not load image.")
            else:
                text, _ = fetch_file_text(_token_fingerprint(), f["path"])
                if text is None:
                    st.caption("Binary file — not previewable.")
                elif f["name"].lower().endswith(".json"):
                    try:
                        st.json(json.loads(text))
                    except json.JSONDecodeError:
                        st.code(text[:5000], language="json")
                elif f["name"].lower().endswith((".html", ".htm")):
                    st.code(text[:5000], language="html")
                    st.caption("Showing first 5,000 chars.")
                elif f["name"].lower().endswith(".md"):
                    st.markdown(text[:5000])
                else:
                    st.code(text[:5000])

    if relevant:
        with st.expander(f"All files matching this phase ({len(relevant)})", expanded=False):
            st.dataframe(
                pd.DataFrame([{"Path": f["path"], "Size": f["size"]} for f in relevant]),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# API key + spec-loader helpers (used by the Run tab)
# ---------------------------------------------------------------------------
def get_api_keys() -> dict[str, str]:
    """Resolve provider API keys from Streamlit secrets or env vars."""
    keys: dict[str, str] = {}
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        v = None
        try:
            v = st.secrets.get(k)  # type: ignore[attr-defined]
        except Exception:
            pass
        if not v:
            v = os.environ.get(k)
        if v:
            keys[k] = str(v).strip()
    return keys


def agent_spec_loader_github() -> engine.AgentSpecLoader:
    """Build an AgentSpecLoader that reads agent .md specs from the bdlb repo."""
    def _load(path: str) -> str:
        text, _ = fetch_file_text(_token_fingerprint(), path)
        if text is None:
            raise RuntimeError(f"Agent spec not found: {path}")
        return text
    return engine.AgentSpecLoader(_load)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_run, tab_workflow, tab_costs, tab_editor = st.tabs(
    ["▶️ Run", "🗺️ Workflow", "💰 Costs", "✏️ Editor"]
)

# ---------------------------------------------------------------------------
# Tab 0 — Run (generation engine)
# ---------------------------------------------------------------------------
with tab_run:
    st.subheader("Run a new lesson build")

    api_keys = get_api_keys()
    have_anthropic = "ANTHROPIC_API_KEY" in api_keys
    have_openai = "OPENAI_API_KEY" in api_keys
    have_google = "GOOGLE_API_KEY" in api_keys

    # Provider status row
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.markdown(f"{'🟢' if have_anthropic else '⚪'} Anthropic")
    pc2.markdown(f"{'🟢' if have_openai else '⚪'} OpenAI")
    pc3.markdown(f"{'🟢' if have_google else '⚪'} Gemini")
    pc4.markdown(f"{'🟢' if connected else '⚪'} GitHub")

    if not (have_anthropic or have_openai or have_google):
        st.warning(
            "No provider API keys configured. Add at least one to Streamlit Cloud → "
            "Settings → Secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)."
        )

    if not connected:
        st.warning("GitHub is not connected — agent .md specs cannot be loaded. "
                   "Add GITHUB_TOKEN to Streamlit secrets or paste a token in the sidebar.")

    st.markdown("#### Inputs")
    ic1, ic2 = st.columns([2, 1])
    with ic1:
        seed_file = st.file_uploader(
            "Seed question image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="seed_upload",
        )
    with ic2:
        grade_hint = st.selectbox(
            "Grade hint (optional)",
            options=[None, 3, 4, 5, 6, 7, 8],
            format_func=lambda g: "(none)" if g is None else f"Grade {g}",
            key="grade_hint",
        )

    # Build model dropdown from configured providers
    available_models: list[str] = []
    if have_anthropic:
        available_models += engine.ANTHROPIC_MODELS
    if have_openai:
        available_models += engine.OPENAI_MODELS
    if have_google:
        available_models += engine.GEMINI_MODELS

    st.markdown("#### Model assignment")
    if not available_models:
        st.info("Configure at least one provider key to assign models.")
        orchestrator_model = None
        agent_model_map: dict[str, str] = {}
    else:
        # Sensible defaults: Claude Sonnet for orchestrator (good reasoner, JSON-friendly);
        # Gemini for vision agents (cheap + multimodal); Claude Sonnet for authoring.
        def _default(preferred: list[str]) -> str:
            for p in preferred:
                for m in available_models:
                    if p in m:
                        return m
            return available_models[0]

        default_orch = _default(["claude-sonnet-4", "claude-3-5-sonnet", "gpt-4o", "gemini-2.5-pro"])
        default_vision = _default(["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o"])
        default_author = _default(["claude-sonnet-4", "claude-3-5-sonnet", "gpt-4o"])
        default_qc = _default(["gpt-4o-mini", "claude-3-5-haiku", "gemini-2.0-flash", "claude-sonnet-4"])

        mc1, mc2 = st.columns(2)
        with mc1:
            orchestrator_model = st.selectbox(
                "Orchestrator model (router)",
                options=available_models,
                index=available_models.index(default_orch) if default_orch in available_models else 0,
                key="orch_model",
                help="This model receives the run state each turn and emits JSON dispatch decisions.",
            )
            vision_model = st.selectbox(
                "Vision agents (seed extractor, image-*)",
                options=available_models,
                index=available_models.index(default_vision) if default_vision in available_models else 0,
                key="vision_model",
            )
        with mc2:
            author_model = st.selectbox(
                "Authoring agents (lesson plan, items, article)",
                options=available_models,
                index=available_models.index(default_author) if default_author in available_models else 0,
                key="author_model",
            )
            qc_model = st.selectbox(
                "QC agents (deterministic + smart QC)",
                options=available_models,
                index=available_models.index(default_qc) if default_qc in available_models else 0,
                key="qc_model",
            )

        agent_model_map = {}

        def model_for_agent(agent_name: str) -> str:
            n = agent_name.lower()
            if n in engine.VISION_AGENTS:
                return vision_model
            if "qc" in n:
                return qc_model
            return author_model

    st.markdown("#### Run options")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        max_turns = st.number_input("Max orchestrator turns", min_value=1, max_value=200,
                                    value=40, step=5, help="Safety stop for the orchestrator loop.")
    with rc2:
        run_id_override = st.text_input("Run ID (optional)",
                                        placeholder=f"e.g. test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    with rc3:
        push_on_complete = st.checkbox("Push to bdlb/runs/ on complete", value=False,
                                       help="When the run finishes, commit every produced file to the bdlb pipeline repo.")

    # --- Manual gating controls ---
    st.markdown("#### 🚫 Manual approval gates")
    st.caption(
        "Pause the pipeline before each phase you want to inspect. The run will stop, "
        "show you everything produced so far, and wait for you to click **▶️ Continue**. "
        "Use **Manual mode** to also pause after every single orchestrator turn."
    )
    mg_cols = st.columns([1, 3])
    with mg_cols[0]:
        pause_every_turn_ui = st.checkbox(
            "🔍 Manual mode",
            value=False,
            help="Pause after EVERY orchestrator turn (one dispatch wave at a time). Strongest control — useful while you tune prompts.",
        )
    with mg_cols[1]:
        phase_opts = ["P1", "P2", "P3", "P4", "P5", "P6"]
        pause_before_ui = st.multiselect(
            "⏸️ Pause before phase\u2026",
            options=phase_opts,
            default=[],
            help="When the orchestrator transitions into one of these phases, the pipeline halts and waits for your approval.",
            disabled=pause_every_turn_ui,
        )

    # --- Buttons row ---
    is_resumable = st.session_state.get("run_paused_state") is not None
    can_run = bool(seed_file and orchestrator_model and connected) and not is_resumable
    btn_cols = st.columns([3, 1])
    with btn_cols[0]:
        start = st.button("▶️ Start run", type="primary", disabled=not can_run, use_container_width=True)
    with btn_cols[1]:
        cancel_paused = st.button("🗑️ Discard paused run",
                                  disabled=not is_resumable, use_container_width=True)
    if cancel_paused:
        for k in ("run_paused_state", "run_log", "run_seed_bytes", "run_seed_name",
                  "run_orch_model", "run_pause_before", "run_pause_every",
                  "run_max_turns", "run_push_on_complete"):
            st.session_state.pop(k, None)
        st.rerun()

    if is_resumable:
        paused: engine.RunState = st.session_state["run_paused_state"]
        cur_phase = paused.build_state.get("current_phase", "?")
        st.info(
            f"⏸️ **Paused** — run `{paused.run_id}` is waiting at phase **{cur_phase}** "
            f"(turn {paused.turn}, ${paused.total_cost_usd:,.4f} so far). "
            "Review the cards below, then continue."
        )

    # Live stream container
    live = st.container(border=True)

    # --- Continue button (only when paused) ---
    continue_clicked = False
    if is_resumable:
        continue_clicked = st.button("▶️ Continue from where we paused",
                                     type="primary", use_container_width=True)

    # Resolve whether we should run this script invocation
    should_run = (start and can_run) or continue_clicked

    # --- Helper: render any log entry into the live container ---
    # Defined here so it's available BOTH for live streaming (in `should_run` below)
    # AND for re-rendering after a Streamlit rerun while paused.
    def _render_log_entry(entry: dict, container):
        t = entry["type"]
        if t == "info":
            container.info(entry["msg"])
        elif t == "success":
            container.success(entry["msg"])
        elif t == "warning":
            container.warning(entry["msg"])
        elif t == "error":
            container.error(entry["msg"])
        elif t == "orchestrator_turn":
            r = entry["record"]
            title = f"🧠 Turn {r.turn} — Orchestrator  ·  {r.model}  ·  ${r.cost_usd:.4f}"
            with container.expander(title, expanded=False):
                if r.error:
                    st.error(r.error)
                st.caption("Decision (parsed):")
                st.json(r.decision_parsed)
                with st.expander("Prompt sent (system)", expanded=False):
                    st.code(r.prompt_system[:4000] + ("\n…[truncated]" if len(r.prompt_system) > 4000 else ""))
                with st.expander("Prompt sent (user)", expanded=False):
                    st.code(r.prompt_user)
                with st.expander("Raw response", expanded=False):
                    st.code(r.decision_raw)
        elif t == "agent_call":
            r2 = entry["record"]
            icon = "❌" if r2.error else "✅"
            title = (f"{icon} Turn {r2.turn} — {r2.agent} ({r2.task_name})  ·  "
                     f"{r2.model}  ·  {r2.tokens_in:,} in / {r2.tokens_out:,} out  ·  "
                     f"${r2.cost_usd:.4f}  ·  {r2.elapsed_ms/1000:.1f}s")
            with container.expander(title, expanded=False):
                if r2.error:
                    st.error(r2.error)
                if r2.output_path:
                    st.caption(f"Output path: `{r2.output_path}`")
                st.caption("Output:")
                try:
                    obj = json.loads(r2.output_text)
                    st.json(obj)
                except Exception:
                    st.code(r2.output_text[:6000] +
                            ("\n…[truncated]" if len(r2.output_text) > 6000 else ""))
                with st.expander("Prompt sent (system, agent spec)", expanded=False):
                    st.code(r2.prompt_system[:4000] +
                            ("\n…[truncated]" if len(r2.prompt_system) > 4000 else ""))
                with st.expander("Prompt sent (user, inputs)", expanded=False):
                    st.code(r2.prompt_user)

    # --- If paused AND not about to resume, render the saved snapshot now ---
    # This is what was missing: after pause + Streamlit rerun, the log was empty.
    # Now we always show what happened so far while the run sits idle.
    if is_resumable and not should_run:
        paused_state: engine.RunState = st.session_state["run_paused_state"]
        paused_log: list[dict] = st.session_state.get("run_log", [])
        paused_seed = st.session_state.get("run_seed_bytes")
        paused_seed_name = st.session_state.get("run_seed_name", paused_state.seed_image_name)

        live.markdown(f"##### 🌱 Seed: `{paused_seed_name}`  ·  run_id: `{paused_state.run_id}`  ·  ⏸️ paused")
        if paused_seed:
            live.image(paused_seed, width=320)
        live.divider()

        snap_cols = live.columns(4)
        snap_cols[0].metric("Turn", paused_state.turn)
        snap_cols[1].metric("Phase", paused_state.build_state.get("current_phase", "—"))
        snap_cols[2].metric("Cost", f"${paused_state.total_cost_usd:,.4f}")
        snap_cols[3].metric("Tokens",
                            f"{paused_state.total_tokens_in + paused_state.total_tokens_out:,}")

        snap_log_area = live.container()
        snap_log_area.markdown(f"**📜 What's happened so far ({len(paused_log)} events):**")
        for entry in paused_log:
            _render_log_entry(entry, snap_log_area)

        # Also show the files produced so far so the user can audit before continuing
        if paused_state.files:
            with live.expander(f"📁 Files produced so far ({len(paused_state.files)})", expanded=False):
                for path in sorted(paused_state.files.keys()):
                    content = paused_state.files[path]
                    with st.expander(path, expanded=False):
                        if path.endswith(".json"):
                            try:
                                st.json(json.loads(content))
                            except Exception:
                                st.code(content[:10000] + ("\n…[truncated]" if len(content) > 10000 else ""))
                        else:
                            st.code(content[:10000] + ("\n…[truncated]" if len(content) > 10000 else ""))

    if should_run:
        import asyncio

        resuming = continue_clicked
        if resuming:
            saved_state: engine.RunState = st.session_state["run_paused_state"]
            seed_bytes = st.session_state.get("run_seed_bytes") or saved_state.seed_image_bytes or b""
            seed_name = st.session_state.get("run_seed_name") or saved_state.seed_image_name
            run_id = saved_state.run_id
            # Re-read the prior settings (so the model picks etc. don't matter on resume)
            orchestrator_model_eff = st.session_state.get("run_orch_model", orchestrator_model)
            pause_before_eff = set(st.session_state.get("run_pause_before", pause_before_ui))
            pause_every_eff = bool(st.session_state.get("run_pause_every", pause_every_turn_ui))
            max_turns_eff = int(st.session_state.get("run_max_turns", max_turns))
            push_on_complete_eff = bool(st.session_state.get("run_push_on_complete", push_on_complete))
            log_entries: list[dict] = st.session_state.get("run_log", [])
        else:
            seed_bytes = seed_file.getvalue()
            seed_name = seed_file.name
            run_id = (run_id_override.strip()
                      or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            orchestrator_model_eff = orchestrator_model
            pause_before_eff = set(pause_before_ui)
            pause_every_eff = bool(pause_every_turn_ui)
            max_turns_eff = int(max_turns)
            push_on_complete_eff = bool(push_on_complete)
            saved_state = None
            log_entries = []
            # Stash these so a future resume uses the same settings
            st.session_state["run_seed_bytes"] = seed_bytes
            st.session_state["run_seed_name"] = seed_name
            st.session_state["run_orch_model"] = orchestrator_model_eff
            st.session_state["run_pause_before"] = list(pause_before_eff)
            st.session_state["run_pause_every"] = pause_every_eff
            st.session_state["run_max_turns"] = max_turns_eff
            st.session_state["run_push_on_complete"] = push_on_complete_eff

        # Show the seed image up top
        live.markdown(f"##### 🌱 Seed: `{seed_name}`  ·  run_id: `{run_id}`"
                      + ("  ·  🔄 resuming" if resuming else ""))
        if seed_bytes:
            live.image(seed_bytes, width=320)

        # Metrics row that updates as events stream in
        live.divider()
        metric_row = live.columns(4)
        m_turn = metric_row[0].empty()
        m_phase = metric_row[1].empty()
        m_cost = metric_row[2].empty()
        m_tokens = metric_row[3].empty()
        if saved_state is not None:
            m_turn.metric("Turn", saved_state.turn)
            m_phase.metric("Phase", saved_state.build_state.get("current_phase", "—"))
            m_cost.metric("Cost", f"${saved_state.total_cost_usd:,.4f}")
            m_tokens.metric("Tokens",
                            f"{saved_state.total_tokens_in + saved_state.total_tokens_out:,}")
        else:
            m_turn.metric("Turn", 0)
            m_phase.metric("Phase", "—")
            m_cost.metric("Cost", "$0.00")
            m_tokens.metric("Tokens", "0")

        log_area = live.container()

        # Replay any prior log entries from a previous (paused) chunk
        if log_entries:
            log_area.markdown(f"**📜 Resuming — replaying {len(log_entries)} prior events:**")
        for entry in log_entries:
            _render_log_entry(entry, log_area)

        spec_loader = agent_spec_loader_github()

        async def _drive():
            async for ev in engine.run_pipeline(
                run_id=run_id,
                seed_image_bytes=seed_bytes,
                seed_image_name=seed_name,
                grade_hint=grade_hint,
                spec_loader=spec_loader,
                orchestrator_model=orchestrator_model_eff,
                model_for_agent=model_for_agent,
                api_keys=api_keys,
                max_turns=max_turns_eff,
                pause_before_phases=pause_before_eff,
                pause_every_turn=pause_every_eff,
                resume_state=saved_state,
            ):
                yield ev

        # Drain the async generator
        loop = asyncio.new_event_loop()
        was_paused = False
        try:
            asyncio.set_event_loop(loop)
            agen = _drive()
            final_state: engine.RunState | None = saved_state
            while True:
                try:
                    ev = loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
                final_state = ev.get("state", final_state)
                # Update metrics
                if final_state is not None:
                    m_turn.metric("Turn", final_state.turn)
                    m_phase.metric("Phase", final_state.build_state.get("current_phase", "—"))
                    m_cost.metric("Cost", f"${final_state.total_cost_usd:,.4f}")
                    m_tokens.metric("Tokens",
                                    f"{final_state.total_tokens_in + final_state.total_tokens_out:,}")

                t = ev["type"]
                if t == "started":
                    entry = {"type": "info", "msg": "Run started."}
                    log_entries.append(entry); _render_log_entry(entry, log_area)
                elif t == "orchestrator_turn":
                    entry = {"type": "orchestrator_turn", "record": ev["record"]}
                    log_entries.append(entry); _render_log_entry(entry, log_area)
                elif t == "phase_change":
                    entry = {"type": "success", "msg": f"→ Phase {ev['phase']} ({ev['status']})"}
                    log_entries.append(entry); _render_log_entry(entry, log_area)
                elif t == "agent_call":
                    entry = {"type": "agent_call", "record": ev["record"]}
                    log_entries.append(entry); _render_log_entry(entry, log_area)
                elif t == "paused":
                    was_paused = True
                    entry = {"type": "warning", "msg": f"⏸️ Paused: {ev['reason']}"}
                    log_entries.append(entry); _render_log_entry(entry, log_area)
                elif t == "error":
                    entry = {"type": "error", "msg": ev["message"]}
                    log_entries.append(entry); _render_log_entry(entry, log_area)
                elif t == "done":
                    entry = {"type": "success", "msg": "🏁 Run complete."}
                    log_entries.append(entry); _render_log_entry(entry, log_area)
        finally:
            loop.close()

        # Persist or clear paused state
        if was_paused and final_state is not None:
            # Strip seed bytes from the saved state to keep session_state light;
            # we kept seed_bytes separately in run_seed_bytes for resume.
            final_state.seed_image_bytes = None
            st.session_state["run_paused_state"] = final_state
            st.session_state["run_log"] = log_entries
            st.rerun()
        else:
            # Run finished (success or error) — clear any prior pause
            for k in ("run_paused_state", "run_log", "run_seed_bytes", "run_seed_name",
                      "run_orch_model", "run_pause_before", "run_pause_every",
                      "run_max_turns", "run_push_on_complete"):
                st.session_state.pop(k, None)

        # Optional: push run artifacts back to GitHub bdlb/runs/{run_id}/
        # Only on actual completion, not on pause.
        if final_state and not was_paused and push_on_complete_eff and connected and final_state.done:
            client = github_client()
            if client is not None:
                try:
                    repo = client.get_repo(PIPELINE_REPO)
                    pushed = 0
                    failed = 0
                    # Write build_state.json
                    bs_path = f"runs/{final_state.run_id}/build_state.json"
                    try:
                        repo.create_file(bs_path, f"Run {final_state.run_id} state",
                                         json.dumps(final_state.build_state, indent=2))
                        pushed += 1
                    except GithubException:
                        failed += 1
                    # Write events as JSONL
                    ev_path = f"runs/{final_state.run_id}/build_events.jsonl"
                    try:
                        repo.create_file(ev_path, f"Run {final_state.run_id} events",
                                         "\n".join(json.dumps(e) for e in final_state.events))
                        pushed += 1
                    except GithubException:
                        failed += 1
                    # Write each produced file
                    for path, content in final_state.files.items():
                        if path.endswith("/" + final_state.seed_image_name):
                            continue  # skip the placeholder for the seed image
                        norm = path[5:] if path.startswith("bdlb/") else path  # strip leading 'bdlb/'
                        try:
                            repo.create_file(norm, f"Run {final_state.run_id} artifact", content)
                            pushed += 1
                        except GithubException:
                            failed += 1
                    # Push seed image as binary
                    try:
                        seed_path = f"runs/{final_state.run_id}/seed/{final_state.seed_image_name}"
                        repo.create_file(seed_path, "Seed image",
                                         final_state.seed_image_bytes or b"")
                    except GithubException:
                        failed += 1

                    if failed == 0:
                        live.success(f"✅ Pushed {pushed} files to {PIPELINE_REPO}/runs/{final_state.run_id}/")
                    else:
                        live.warning(f"Pushed {pushed} files — {failed} failed (some may have already existed).")
                except Exception as e:
                    live.error(f"Push failed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab 1 — Workflow
# ---------------------------------------------------------------------------
with tab_workflow:
    st.subheader("Pipeline Workflow")
    if active_run:
        st.caption(f"Run: `{active_run}`")

    state = parse_state(active_run) if active_run else {}
    statuses = phase_statuses(state)

    color_map = {
        "completed":   ("#d1f4d6", "#2ea043"),   # green
        "in_progress": ("#fff4c2", "#d4a017"),   # yellow
        "failed":      ("#fbd5d5", "#d1242f"),   # red
        "pending":     ("#e0e0e0", "#9e9e9e"),   # grey
    }

    lines = ["flowchart LR"]
    for pid, label, _key in PHASES:
        safe_label = label.replace('"', "'")
        lines.append(f'    {pid}["{safe_label}<br/><i>Phase {pid[1]}</i>"]')
    lines.append("    P6_PAD[\" \"]")
    lines.append("    P0 --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P6_PAD")
    for pid, _, _ in PHASES:
        fill, stroke = color_map[statuses[pid]]
        lines.append(f"    style {pid} fill:{fill},stroke:{stroke},stroke-width:2px,color:#222")
    lines.append("    style P6_PAD fill:transparent,stroke:transparent,color:transparent")
    lines.append("    linkStyle 6 stroke:transparent")
    st_mermaid("\n".join(lines), height="320px")

    # Legend
    lc1, lc2, lc3, lc4, lc5 = st.columns([1, 1, 1, 1, 4])
    lc1.markdown("<span style='color:#2ea043'>●</span> Completed", unsafe_allow_html=True)
    lc2.markdown("<span style='color:#d4a017'>●</span> In progress", unsafe_allow_html=True)
    lc3.markdown("<span style='color:#d1242f'>●</span> Failed", unsafe_allow_html=True)
    lc4.markdown("<span style='color:#9e9e9e'>●</span> Pending", unsafe_allow_html=True)

    st.divider()

    # Phase selector → output card
    st.markdown("#### Phase output")
    if not active_run:
        st.info(
            "👆 Once runs appear in `bdlb/runs/`, select one in the sidebar and "
            "pick a phase below to preview its output — seed image, lesson plan, "
            "items, article drafts, QC reports, and final lesson."
        )
    else:
        phase_choice = st.selectbox(
            "Select a phase",
            options=[pid for pid, _, _ in PHASES],
            format_func=lambda pid: f"{pid} — {PHASE_LABEL_BY_ID[pid]}  ·  {statuses[pid]}",
            key="phase_choice",
        )
        render_phase_card(active_run, phase_choice, statuses[phase_choice])


# ---------------------------------------------------------------------------
# Tab 2 — Costs
# ---------------------------------------------------------------------------
with tab_costs:
    st.subheader("Run Cost Summary")

    if not active_run:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Runs", len(runs) if connected else "—")
        col2.metric("Total Tokens", "—")
        col3.metric("Total Cost", "—")
        col4.metric("Avg Cost / Run", "—")
        st.markdown("#### Per-Agent Breakdown")
        st.dataframe(
            pd.DataFrame(columns=["Phase", "Agent", "Model", "Tokens In", "Tokens Out", "Cost ($)"]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Cost data is parsed from `bdlb/runs/{run_id}/events.jsonl`. "
            "Select a run in the sidebar to load."
        )
    else:
        events = parse_events(active_run)
        cost_df = build_cost_table(events)

        total_tokens = int(cost_df["Tokens In"].sum() + cost_df["Tokens Out"].sum()) if not cost_df.empty else 0
        total_cost = float(cost_df["Cost ($)"].sum()) if not cost_df.empty else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Runs", len(runs))
        col2.metric("Total Tokens", f"{total_tokens:,}" if total_tokens else "—")
        col3.metric("Total Cost", f"${total_cost:,.2f}" if total_cost else "—")
        col4.metric("Avg Cost / Run", f"${total_cost:,.2f}" if total_cost else "—")

        st.markdown("#### Per-Agent Breakdown")
        if cost_df.empty:
            st.info(
                "No token-usage events found in this run's `build_events.jsonl`. "
                "Cost rows are extracted from events containing model + token fields."
            )
            st.dataframe(
                pd.DataFrame(columns=["Phase", "Agent", "Model", "Tokens In", "Tokens Out", "Cost ($)"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.dataframe(cost_df, use_container_width=True, hide_index=True)

            st.markdown("#### Aggregated by phase")
            by_phase = (
                cost_df.groupby("Phase", as_index=False)
                       .agg(**{
                           "Tokens In": ("Tokens In", "sum"),
                           "Tokens Out": ("Tokens Out", "sum"),
                           "Cost ($)": ("Cost ($)", "sum"),
                       })
                       .sort_values("Phase")
            )
            st.dataframe(by_phase, use_container_width=True, hide_index=True)

        st.caption(
            "Costs computed from token counts × `MODEL_PRICING` table in `app.py`. "
            "Update pricing there as new models are added."
        )


# ---------------------------------------------------------------------------
# Tab 3 — Editor
# ---------------------------------------------------------------------------
with tab_editor:
    st.subheader("Artifact Editor")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("#### 📂 Run Files")

        if not connected:
            st.info("Connect to GitHub (sidebar) to load run files.")
            files: list[dict[str, Any]] = []
        elif not active_run:
            st.info("Select a run in the sidebar to load its files.")
            files = []
        else:
            files = list_run_files(_token_fingerprint(), active_run)
            if not files:
                st.warning(f"No files found under `runs/{active_run}/`.")

        if files:
            file_paths = [f["path"] for f in files]
            selected_path = st.selectbox("Select file", file_paths, key="edit_file_select")
        else:
            st.selectbox("Select file", ["No files loaded"], disabled=True)
            selected_path = None

        if st.button("📥 Reload from GitHub", use_container_width=True,
                     disabled=not (connected and selected_path)):
            fetch_file_text.clear()
            st.rerun()

    with right_col:
        st.markdown("#### 📝 File Editor")

        if not (connected and active_run and selected_path):
            st.code("# No file selected", language="json")
            st.button("💾 Save & push to GitHub", disabled=True)
        else:
            text, sha = fetch_file_text(_token_fingerprint(), selected_path)

            if text is None:
                st.warning("This file is binary or unreadable as text — cannot edit here.")
            else:
                ext = selected_path.rsplit(".", 1)[-1].lower()
                lang_map = {
                    "json": "json", "jsonl": "json", "md": "markdown",
                    "py": "python", "html": "html", "htm": "html",
                    "js": "javascript", "css": "css", "yml": "yaml", "yaml": "yaml",
                }
                lang = lang_map.get(ext, "text")

                if ACE_AVAILABLE:
                    edited = st_ace(
                        value=text,
                        language=lang,
                        theme="github",
                        key=f"ace_{selected_path}",
                        height=480,
                        auto_update=True,
                        wrap=True,
                        show_gutter=True,
                    )
                else:
                    edited = st.text_area(
                        "Editor", value=text, height=480, key=f"ta_{selected_path}",
                    )

                with st.form("save_form", clear_on_submit=False):
                    commit_msg = st.text_input(
                        "Commit message",
                        value=f"Edit {selected_path} via BDLB Dashboard",
                    )
                    submitted = st.form_submit_button("💾 Save & push to GitHub")
                    if submitted:
                        if edited == text:
                            st.info("No changes to save.")
                        else:
                            client = github_client()
                            try:
                                repo = client.get_repo(PIPELINE_REPO)  # type: ignore[union-attr]
                                repo.update_file(
                                    path=selected_path,
                                    message=commit_msg or f"Edit {selected_path}",
                                    content=edited,
                                    sha=sha,
                                )
                                fetch_file_text.clear()
                                st.success(f"✅ Pushed to `{PIPELINE_REPO}`")
                                st.balloons()
                            except GithubException as e:
                                msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
                                st.error(f"GitHub rejected the commit: {msg}")
                            except Exception as e:
                                st.error(f"Push failed: {type(e).__name__}: {e}")
