"""
BDLB Generation Engine — orchestrator-loop runtime

Architecture
------------
This module turns a `bdlb-orchestrator.md` spec into a live pipeline runner.
The orchestrator is treated as a Claude API call that, given the current run
state, returns a JSON decision describing the next dispatches. The engine
executes those dispatches concurrently (each one is itself an LLM call
against a specialist agent's `.md` spec) and feeds their outputs back into
the next orchestrator turn.

  while not state.done:
      decision = call_orchestrator(state)              # 1 API call
      outputs  = gather(call_agent(d) for d in decision.dispatches)   # N API calls in parallel
      state = apply(decision, outputs)
      yield event for the dashboard

The engine is provider-agnostic: each phase (orchestrator + each agent) can
target any of Anthropic / OpenAI / Google Gemini via a unified adapter.

This file exposes one async generator `run_pipeline(...)` that yields events
the dashboard streams to the UI.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator, Callable

# Provider SDKs — imported lazily so the dashboard can still run with only
# some keys configured.
try:
    from anthropic import Anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

try:
    import google.generativeai as genai
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens, input / output)
# ---------------------------------------------------------------------------
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-20250514":      (15.00, 75.00),
    "claude-opus-4":               (15.00, 75.00),
    "claude-sonnet-4-20250514":    ( 3.00, 15.00),
    "claude-sonnet-4":             ( 3.00, 15.00),
    "claude-3-7-sonnet-20250219":  ( 3.00, 15.00),
    "claude-3-5-sonnet-20241022":  ( 3.00, 15.00),
    "claude-3-5-haiku-20241022":   ( 0.80,  4.00),
    # OpenAI
    "gpt-5":                       ( 5.00, 15.00),
    "gpt-4o":                      ( 2.50, 10.00),
    "gpt-4o-mini":                 ( 0.15,  0.60),
    "o1":                          (15.00, 60.00),
    "o3-mini":                     ( 1.10,  4.40),
    # Google
    "gemini-2.5-pro":              ( 1.25, 10.00),
    "gemini-2.0-flash":            ( 0.10,  0.40),
    "gemini-1.5-pro":              ( 1.25,  5.00),
    "gemini-1.5-flash":            ( 0.075, 0.30),
}
DEFAULT_PRICING = (3.00, 15.00)


def model_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    key = (model or "").lower().strip()
    pricing = DEFAULT_PRICING
    for known, p in PRICING.items():
        if known in key:
            pricing = p
            break
    return (tokens_in / 1_000_000) * pricing[0] + (tokens_out / 1_000_000) * pricing[1]


# ---------------------------------------------------------------------------
# Provider catalog
# ---------------------------------------------------------------------------
ANTHROPIC_MODELS = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]
OPENAI_MODELS = [
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o3-mini",
]
GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


def detect_provider(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    return "unknown"


# ---------------------------------------------------------------------------
# Unified LLM call
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str
    provider: str
    elapsed_ms: int
    raw: Any = None  # the full provider response for debugging


def _now_ms() -> int:
    return int(time.time() * 1000)


def call_llm(
    *,
    model: str,
    system: str,
    user: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/png",
    api_keys: dict[str, str],
    max_tokens: int = 8192,
    temperature: float = 0.2,
    response_format_json: bool = False,
) -> LLMResult:
    """Synchronously call any supported model with a unified signature.

    `system` is the system prompt (the agent's .md spec body).
    `user` is the user-turn content.
    `image_bytes` is optional vision input — used by Phase 0/1 (seed extraction)
    and image-renderer/image-qc agents.
    `response_format_json` requests JSON output when supported.
    """
    provider = detect_provider(model)
    t0 = _now_ms()

    if provider == "anthropic":
        if not _ANTHROPIC_OK:
            raise RuntimeError("anthropic SDK not installed")
        key = api_keys.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        client = Anthropic(api_key=key)

        content: list[dict] = []
        if image_bytes:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_mime,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                },
            })
        content.append({"type": "text", "text": user})

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        ti = getattr(resp.usage, "input_tokens", 0)
        to = getattr(resp.usage, "output_tokens", 0)
        return LLMResult(
            text=text, tokens_in=ti, tokens_out=to,
            cost_usd=model_cost(model, ti, to),
            model=model, provider="anthropic",
            elapsed_ms=_now_ms() - t0, raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )

    if provider == "openai":
        if not _OPENAI_OK:
            raise RuntimeError("openai SDK not installed")
        key = api_keys.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        client = OpenAI(api_key=key)

        user_content: list[dict] | str
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            user_content = [
                {"type": "text", "text": user},
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_mime};base64,{b64}"}},
            ]
        else:
            user_content = user

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}

        # Newer OpenAI models (o-series, gpt-5, and any future reasoning models)
        # require `max_completion_tokens` instead of `max_tokens` and don't accept
        # custom temperature. Detect by trying once; on the known 400 errors,
        # rewrite kwargs and retry.
        m_lower = model.lower()
        needs_new_params = (
            m_lower.startswith("o1")
            or m_lower.startswith("o3")
            or m_lower.startswith("o4")
            or m_lower.startswith("gpt-5")
        )
        if needs_new_params:
            kwargs.pop("temperature", None)
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        def _do_call(kw: dict[str, Any]):
            return client.chat.completions.create(**kw)

        try:
            resp = _do_call(kwargs)
        except Exception as e:
            msg = str(e)
            mutated = False
            if "max_tokens" in msg and "max_completion_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                mutated = True
            if "temperature" in msg and "unsupported" in msg.lower():
                kwargs.pop("temperature", None)
                mutated = True
            if mutated:
                resp = _do_call(kwargs)
            else:
                raise
        text = resp.choices[0].message.content or ""
        ti = getattr(resp.usage, "prompt_tokens", 0) or 0
        to = getattr(resp.usage, "completion_tokens", 0) or 0
        return LLMResult(
            text=text, tokens_in=ti, tokens_out=to,
            cost_usd=model_cost(model, ti, to),
            model=model, provider="openai",
            elapsed_ms=_now_ms() - t0, raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )

    if provider == "google":
        if not _GEMINI_OK:
            raise RuntimeError("google-generativeai SDK not installed")
        key = api_keys.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not configured")
        genai.configure(api_key=key)

        parts: list = [user]
        if image_bytes:
            parts.insert(0, {"mime_type": image_mime, "data": image_bytes})

        gen_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if response_format_json:
            gen_kwargs["response_mime_type"] = "application/json"

        model_obj = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=gen_kwargs,
        )
        resp = model_obj.generate_content(parts)
        text = resp.text or ""
        try:
            ti = resp.usage_metadata.prompt_token_count
            to = resp.usage_metadata.candidates_token_count
        except Exception:
            ti, to = 0, 0
        return LLMResult(
            text=text, tokens_in=ti, tokens_out=to,
            cost_usd=model_cost(model, ti, to),
            model=model, provider="google",
            elapsed_ms=_now_ms() - t0, raw=None,
        )

    raise RuntimeError(f"Unknown provider for model: {model}")


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------
@dataclass
class AgentCallRecord:
    """One agent invocation's full record — what the dashboard streams."""
    turn: int
    agent: str
    task_name: str
    model: str
    provider: str
    prompt_system: str
    prompt_user: str
    output_text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    elapsed_ms: int
    output_path: str | None = None
    error: str | None = None


@dataclass
class OrchestratorTurnRecord:
    turn: int
    model: str
    provider: str
    prompt_system: str
    prompt_user: str
    decision_raw: str
    decision_parsed: dict[str, Any]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    elapsed_ms: int
    error: str | None = None


@dataclass
class RunState:
    run_id: str
    seed_image_bytes: bytes | None = None
    seed_image_name: str = "seed.png"
    grade_hint: int | None = None
    files: dict[str, str] = field(default_factory=dict)         # path -> text content
    build_state: dict[str, Any] = field(default_factory=lambda: {"phases_completed": [], "current_phase": "P0", "current_phase_status": "pending"})
    events: list[dict[str, Any]] = field(default_factory=list)  # mirrors build_events.jsonl
    turn: int = 0
    done: bool = False
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    def file_list(self) -> list[str]:
        return sorted(self.files.keys())


# ---------------------------------------------------------------------------
# Agent spec cache (loaded from GitHub or local)
# ---------------------------------------------------------------------------
class AgentSpecLoader:
    """Loads agent .md specs from a callable source (GitHub or local fs)."""

    def __init__(self, loader: Callable[[str], str]):
        self._loader = loader
        self._cache: dict[str, str] = {}

    def load(self, agent_name: str) -> str:
        if agent_name in self._cache:
            return self._cache[agent_name]
        path = f"agents/{agent_name}.md"
        text = self._loader(path)
        self._cache[agent_name] = text
        return text


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
ORCHESTRATOR_AGENT_NAME = "bdlb-orchestrator"
MAX_TURNS_DEFAULT = 40   # safety stop


def _extract_json_block(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM response, tolerating fences."""
    # Try fenced ```json blocks first
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try any fenced block
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try first {...} balanced
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object in response")
    # naive balanced-brace scan
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON in response")


def _build_orchestrator_user_message(state: RunState, last_agent_outputs: list[AgentCallRecord]) -> str:
    """Compose the user-turn the orchestrator receives each loop iteration."""
    bs = json.dumps(state.build_state, indent=2)
    flist = "\n".join(f"- {p}" for p in state.file_list()) or "  (none yet)"
    prior = ""
    if last_agent_outputs:
        bits = []
        for rec in last_agent_outputs:
            head = rec.output_text[:4000]
            bits.append(f"### {rec.agent} ({rec.task_name})\n```\n{head}\n```")
        prior = "\n\n## Previous turn agent outputs\n" + "\n\n".join(bits)
    return f"""EXECUTION_MODE: dashboard_api

## Run context
run_id: {state.run_id}
seed_image: bdlb/runs/{state.run_id}/seed/{state.seed_image_name}  (provided as image input)
grade_hint: {state.grade_hint if state.grade_hint else "(not provided)"}

## Current build_state.json
```json
{bs}
```

## Files already written under bdlb/runs/{state.run_id}/
{flist}
{prior}

## Your task
Decide the next dispatch wave. Respond with exactly one JSON object as defined in the
"Execution mode" section of your spec. No prose outside the JSON.
"""


def _build_agent_user_message(dispatch: dict[str, Any], state: RunState) -> str:
    """Compose the user-turn for a specialist agent call."""
    inputs_block = json.dumps(dispatch.get("inputs", {}), indent=2)

    # Inline the contents of any input files referenced by path so the agent
    # can act on them in a single API call (no file I/O available).
    referenced_files = []
    for v in (dispatch.get("inputs") or {}).values():
        if isinstance(v, str) and v.startswith("bdlb/runs/") and v in state.files:
            referenced_files.append((v, state.files[v]))

    files_section = ""
    if referenced_files:
        files_section = "\n\n## Inlined input file contents\n"
        for path, content in referenced_files:
            head = content[:6000]
            files_section += f"\n### {path}\n```\n{head}\n```\n"

    return f"""You are the **{dispatch['agent']}** agent. Your full spec is your system prompt.

EXECUTION_MODE: dashboard_api
(You are being invoked via plain LLM API by the BDLB Run Dashboard. You do NOT have
file I/O or tool access. Write your full output as your response text.)

## Run context
run_id: {state.run_id}
task_name: {dispatch.get('task_name', '(unnamed)')}

## Inputs from the orchestrator
```json
{inputs_block}
```

## Output target (the runtime will save your response here)
{dispatch.get('output_path', f"bdlb/runs/{state.run_id}/{dispatch['agent']}.json")}
{files_section}

## Instructions
Follow your spec exactly. Produce ONLY the file contents that should be saved at the
output path. If your spec mandates a JSON file, output ONLY the JSON object (no prose,
no markdown fences). If your spec mandates HTML or Markdown, output ONLY that.
"""


async def _call_agent_async(
    *,
    spec_loader: AgentSpecLoader,
    dispatch: dict[str, Any],
    state: RunState,
    seed_image_bytes: bytes | None,
    model_for_agent: Callable[[str], str],
    api_keys: dict[str, str],
    turn: int,
    is_vision: bool,
) -> AgentCallRecord:
    agent = dispatch["agent"]
    task_name = dispatch.get("task_name", agent)
    model = model_for_agent(agent)
    provider = detect_provider(model)

    try:
        system = spec_loader.load(agent)
    except Exception as e:
        return AgentCallRecord(
            turn=turn, agent=agent, task_name=task_name,
            model=model, provider=provider,
            prompt_system="", prompt_user="",
            output_text="", tokens_in=0, tokens_out=0,
            cost_usd=0.0, elapsed_ms=0,
            error=f"Could not load spec: {e}",
        )

    user = _build_agent_user_message(dispatch, state)

    def _sync_call() -> LLMResult:
        return call_llm(
            model=model, system=system, user=user,
            image_bytes=seed_image_bytes if is_vision else None,
            api_keys=api_keys,
            response_format_json=False,  # specs decide their own format
        )

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, _sync_call)
    except Exception as e:
        return AgentCallRecord(
            turn=turn, agent=agent, task_name=task_name,
            model=model, provider=provider,
            prompt_system=system, prompt_user=user,
            output_text="", tokens_in=0, tokens_out=0,
            cost_usd=0.0, elapsed_ms=0,
            error=f"{type(e).__name__}: {e}",
        )

    return AgentCallRecord(
        turn=turn, agent=agent, task_name=task_name,
        model=res.model, provider=res.provider,
        prompt_system=system, prompt_user=user,
        output_text=res.text,
        tokens_in=res.tokens_in, tokens_out=res.tokens_out,
        cost_usd=res.cost_usd, elapsed_ms=res.elapsed_ms,
        output_path=dispatch.get("output_path"),
    )


VISION_AGENTS = {
    "seed-question-extractor",
    "image-qc",
    "image-renderer",
    "image-spec-generator",
    "article-render-qc",
}


async def run_pipeline(
    *,
    run_id: str,
    seed_image_bytes: bytes,
    seed_image_name: str,
    grade_hint: int | None,
    spec_loader: AgentSpecLoader,
    orchestrator_model: str,
    model_for_agent: Callable[[str], str],
    api_keys: dict[str, str],
    max_turns: int = MAX_TURNS_DEFAULT,
) -> AsyncIterator[dict[str, Any]]:
    """Run the pipeline end-to-end, yielding events for the dashboard.

    Yielded event shapes:
      {"type": "started", "state": RunState}
      {"type": "orchestrator_turn", "record": OrchestratorTurnRecord, "state": RunState}
      {"type": "agent_call", "record": AgentCallRecord, "state": RunState}
      {"type": "phase_change", "phase": str, "status": str, "state": RunState}
      {"type": "error", "message": str, "state": RunState}
      {"type": "done", "state": RunState}
    """
    state = RunState(
        run_id=run_id,
        seed_image_bytes=seed_image_bytes,
        seed_image_name=seed_image_name,
        grade_hint=grade_hint,
    )
    # Seed image is treated as a "written file" so the orchestrator sees it
    state.files[f"bdlb/runs/{run_id}/seed/{seed_image_name}"] = "(binary image — provided as vision input to vision-capable agents)"

    yield {"type": "started", "state": state}

    last_outputs: list[AgentCallRecord] = []
    prev_phase = state.build_state.get("current_phase")

    while not state.done and state.turn < max_turns:
        state.turn += 1
        # --- 1. Call orchestrator ---
        try:
            orchestrator_system = spec_loader.load(ORCHESTRATOR_AGENT_NAME)
        except Exception as e:
            yield {"type": "error", "message": f"Could not load orchestrator spec: {e}", "state": state}
            return

        orchestrator_user = _build_orchestrator_user_message(state, last_outputs)
        t0 = _now_ms()
        try:
            # Run sync call in executor so we don't block the event loop
            loop = asyncio.get_running_loop()
            orch_res = await loop.run_in_executor(
                None,
                lambda: call_llm(
                    model=orchestrator_model,
                    system=orchestrator_system,
                    user=orchestrator_user,
                    image_bytes=seed_image_bytes if state.turn == 1 else None,
                    api_keys=api_keys,
                    response_format_json=True,
                    max_tokens=4096,
                ),
            )
        except Exception as e:
            rec = OrchestratorTurnRecord(
                turn=state.turn, model=orchestrator_model,
                provider=detect_provider(orchestrator_model),
                prompt_system=orchestrator_system, prompt_user=orchestrator_user,
                decision_raw="", decision_parsed={},
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                elapsed_ms=_now_ms() - t0,
                error=f"{type(e).__name__}: {e}",
            )
            yield {"type": "orchestrator_turn", "record": rec, "state": state}
            yield {"type": "error", "message": str(e), "state": state}
            return

        state.total_cost_usd += orch_res.cost_usd
        state.total_tokens_in += orch_res.tokens_in
        state.total_tokens_out += orch_res.tokens_out

        try:
            decision = _extract_json_block(orch_res.text)
        except Exception as e:
            rec = OrchestratorTurnRecord(
                turn=state.turn, model=orchestrator_model, provider=orch_res.provider,
                prompt_system=orchestrator_system, prompt_user=orchestrator_user,
                decision_raw=orch_res.text, decision_parsed={},
                tokens_in=orch_res.tokens_in, tokens_out=orch_res.tokens_out,
                cost_usd=orch_res.cost_usd, elapsed_ms=orch_res.elapsed_ms,
                error=f"Could not parse JSON decision: {e}",
            )
            yield {"type": "orchestrator_turn", "record": rec, "state": state}
            yield {"type": "error", "message": "Orchestrator returned malformed JSON; aborting.", "state": state}
            return

        rec = OrchestratorTurnRecord(
            turn=state.turn, model=orchestrator_model, provider=orch_res.provider,
            prompt_system=orchestrator_system, prompt_user=orchestrator_user,
            decision_raw=orch_res.text, decision_parsed=decision,
            tokens_in=orch_res.tokens_in, tokens_out=orch_res.tokens_out,
            cost_usd=orch_res.cost_usd, elapsed_ms=orch_res.elapsed_ms,
        )
        yield {"type": "orchestrator_turn", "record": rec, "state": state}

        # Apply state_updates from orchestrator
        su = decision.get("state_updates", {}) or {}
        if isinstance(su.get("phases_completed"), list):
            state.build_state["phases_completed"] = su["phases_completed"]
        if su.get("current_phase"):
            state.build_state["current_phase"] = su["current_phase"]
        if su.get("current_phase_status"):
            state.build_state["current_phase_status"] = su["current_phase_status"]
        if su.get("notes"):
            state.build_state["notes"] = su["notes"]

        if state.build_state.get("current_phase") != prev_phase:
            yield {
                "type": "phase_change",
                "phase": state.build_state.get("current_phase"),
                "status": state.build_state.get("current_phase_status"),
                "state": state,
            }
            prev_phase = state.build_state.get("current_phase")

        if decision.get("done"):
            state.done = True
            yield {"type": "done", "state": state}
            return

        # --- 2. Execute dispatches concurrently ---
        dispatches = decision.get("dispatches", []) or []
        if not dispatches:
            # Orchestrator said neither dispatch nor done — treat as error
            yield {"type": "error", "message": "Orchestrator returned no dispatches and done=false.", "state": state}
            return

        tasks = []
        for d in dispatches:
            is_vision = d.get("agent") in VISION_AGENTS
            tasks.append(_call_agent_async(
                spec_loader=spec_loader,
                dispatch=d,
                state=state,
                seed_image_bytes=seed_image_bytes if is_vision else None,
                model_for_agent=model_for_agent,
                api_keys=api_keys,
                turn=state.turn,
                is_vision=is_vision,
            ))
        results: list[AgentCallRecord] = await asyncio.gather(*tasks, return_exceptions=False)

        # Persist outputs into state.files and stream events
        for r in results:
            state.total_cost_usd += r.cost_usd
            state.total_tokens_in += r.tokens_in
            state.total_tokens_out += r.tokens_out
            if r.output_path and r.output_text and not r.error:
                state.files[r.output_path] = r.output_text
            state.events.append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "turn": r.turn,
                "agent": r.agent,
                "task_name": r.task_name,
                "model": r.model,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_usd": r.cost_usd,
                "error": r.error,
            })
            yield {"type": "agent_call", "record": r, "state": state}

        last_outputs = results

    if not state.done:
        yield {"type": "error", "message": f"Hit max_turns ({max_turns}) without completion.", "state": state}
