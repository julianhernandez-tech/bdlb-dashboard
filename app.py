"""
BDLB Run Dashboard — Step 1 of 7: UI Shell

Monitoring and control dashboard for BDLB (Backward-Design Lesson Builder),
a multi-agent AI pipeline that takes a STAAR-style math question image as input
and produces a full backward-designed lesson.

This is Step 1: pure UI shell with static placeholders. No API calls.
"""

import streamlit as st
import pandas as pd
from streamlit_mermaid import st_mermaid

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BDLB Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # STEP 3 — populate Active Run selectbox from discovered run folders
    st.markdown("### 📁 Run Selector")
    st.selectbox("Active Run", ["No runs loaded"])

    st.divider()

    # STEP 2 — replace status dot with live GitHub connection state and
    # enable the Connect button (PAT input, repo auth, run discovery)
    st.markdown("### 🔗 GitHub")
    st.markdown(
        "<span style='color:#888;font-size:18px;'>●</span> "
        "<span style='color:#888;'>Not connected</span>",
        unsafe_allow_html=True,
    )
    st.button("Connect to GitHub", disabled=True)

    st.divider()

    st.markdown("### ℹ️ About")
    st.caption(
        "BDLB Run Dashboard v0.1 — monitors Backward-Design Lesson Builder "
        "pipeline runs stored in julianhernandez-tech/bdlb."
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_workflow, tab_costs, tab_editor = st.tabs(
    ["🗺️ Workflow", "💰 Costs", "✏️ Editor"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Workflow
# ---------------------------------------------------------------------------
with tab_workflow:
    st.subheader("Pipeline Workflow")

    # STEP 4 — color nodes by state.json phase status
    # (completed=green, failed=red, in_progress=yellow, pending=grey)
    mermaid_diagram = """
flowchart LR
    P0["Preflight<br/><i>Phase 0</i>"]
    P1["Seed Extraction<br/><i>Phase 1</i>"]
    P2["Backward Design<br/><i>Phase 2</i>"]
    P3["Tier Specification<br/><i>Phase 3</i>"]
    P4["Item Authoring<br/><i>Phase 4</i>"]
    P5["Article Authoring<br/><i>Phase 5</i>"]
    P6["Assembly<br/><i>Phase 6</i>"]

    P6_PAD[" "]

    P0 --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P6_PAD

    classDef neutral fill:#e0e0e0,stroke:#9e9e9e,stroke-width:1px,color:#333;
    classDef invisible fill:transparent,stroke:transparent,color:transparent;
    class P0,P1,P2,P3,P4,P5,P6 neutral;
    class P6_PAD invisible;
    linkStyle 6 stroke:transparent;
"""
    st_mermaid(mermaid_diagram, height="320px")

    # STEP 5 — wire phase click → render readable output card
    # (seed image, lesson plan, items, article drafts, QC reports, final lesson)
    st.markdown(
        """
        <div style="
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            padding: 24px;
            background-color: #fafafa;
            text-align: center;
            color: #666;
            margin-top: 12px;
        ">
            👆 Select a phase node to preview its output — seed image, lesson plan,
            items, article drafts, QC reports, and final lesson will appear here.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tab 2 — Costs
# ---------------------------------------------------------------------------
with tab_costs:
    st.subheader("Run Cost Summary")

    # STEP 6 — parse events.jsonl, map model+tokens → cost using pricing config
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", "—")
    col2.metric("Total Tokens", "—")
    col3.metric("Total Cost", "—")
    col4.metric("Avg Cost / Run", "—")

    st.markdown("#### Per-Agent Breakdown")

    # STEP 6 — populate dataframe from parsed events.jsonl
    empty_costs_df = pd.DataFrame(
        columns=["Phase", "Agent", "Model", "Tokens In", "Tokens Out", "Cost ($)"]
    )
    st.dataframe(empty_costs_df, use_container_width=True, hide_index=True)

    st.caption(
        "Cost data is parsed from `bdlb/runs/{run_id}/events.jsonl`. "
        "Connect to GitHub in Step 2 to load runs."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Editor
# ---------------------------------------------------------------------------
with tab_editor:
    st.subheader("Artifact Editor")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("#### 📂 Run Files")

        # STEP 7 — replace with actual file tree from local runs/ folder
        st.selectbox("Select file", ["No files loaded"])

        # STEP 7 — wire Pull/Push buttons to GitHub API
        st.button("📥 Pull from GitHub", disabled=True, use_container_width=True)
        st.button("📤 Push to GitHub", disabled=True, use_container_width=True)

    with right_col:
        st.markdown("#### 📝 File Editor")

        # STEP 7 — file browser from local runs/ folder, ace editor,
        # commit back via GitHub API
        st.code("# No file selected", language="json")
        st.button("💾 Save Changes", disabled=True)
