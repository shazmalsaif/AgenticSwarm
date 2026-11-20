from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

KNOWN_PROVIDERS: set[str] = {
    "google",
    "gemini",
    "google_genai",
    "openai",
    "openrouter",
    "deterministic",
    "local",
    "offline",
}


def parse_provider_and_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Parse provider and model_name without breaking OpenRouter ':free' model suffixes."""
    parsed_provider = (provider or "").strip().lower() or None
    parsed_model = (model_name or "").strip() or None

    if parsed_model and ":" in parsed_model:
        prefix, rest = parsed_model.split(":", 1)
        pref_clean = prefix.strip().lower()
        if pref_clean in KNOWN_PROVIDERS:
            if not parsed_provider:
                parsed_provider = pref_clean
            parsed_model = rest.strip() or None

    if parsed_provider in ("google", "gemini"):
        parsed_provider = "google_genai"

    if not parsed_provider and parsed_model:
        if "/" in parsed_model or parsed_model.endswith(":free"):
            parsed_provider = "openrouter"

    return parsed_provider, parsed_model


AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "openrouter:openrouter/free",
        "provider": "openrouter",
        "model_name": "openrouter/free",
        "label": "OpenRouter Free • Auto Free Router (Recommended)",
    },
    {
        "id": "openrouter:baidu/ernie-5.5-flash-preview:free",
        "provider": "openrouter",
        "model_name": "baidu/ernie-5.5-flash-preview:free",
        "label": "OpenRouter Free • Baidu ERNIE 5.5 Flash Preview",
    },
    {
        "id": "openrouter:google/gemma-3-27b-it:free",
        "provider": "openrouter",
        "model_name": "google/gemma-3-27b-it:free",
        "label": "OpenRouter Free • Gemma 3 27B IT",
    },
    {
        "id": "openrouter:meta-llama/llama-3.3-70b-instruct:free",
        "provider": "openrouter",
        "model_name": "meta-llama/llama-3.3-70b-instruct:free",
        "label": "OpenRouter Free • Llama 3.3 70B Instruct",
    },
    {
        "id": "openrouter:deepseek/deepseek-chat-v3-0324:free",
        "provider": "openrouter",
        "model_name": "deepseek/deepseek-chat-v3-0324:free",
        "label": "OpenRouter Free • DeepSeek V3 0324",
    },
    {
        "id": "google_genai:gemini-3.8-flash",
        "provider": "google_genai",
        "model_name": "gemini-3.8-flash",
        "label": "Gemini 3.8 Flash — Latest & Recommended",
    },
    {
        "id": "google_genai:gemini-3.8-pro",
        "provider": "google_genai",
        "model_name": "gemini-3.8-pro",
        "label": "Gemini 3.8 Pro — Deep Reasoning",
    },
    {
        "id": "google_genai:gemini-2.5-flash",
        "provider": "google_genai",
        "model_name": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
    },
    {
        "id": "google_genai:gemini-2.5-pro",
        "provider": "google_genai",
        "model_name": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
    },
    {
        "id": "google_genai:gemini-2.0-flash",
        "provider": "google_genai",
        "model_name": "gemini-2.0-flash",
        "label": "Gemini 2.0 Flash",
    },
    {
        "id": "openrouter:google/gemini-2.5-flash",
        "provider": "openrouter",
        "model_name": "google/gemini-2.5-flash",
        "label": "OpenRouter • Gemini 2.5 Flash",
    },
    {
        "id": "openrouter:anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "model_name": "anthropic/claude-3.5-sonnet",
        "label": "OpenRouter • Claude 3.5 Sonnet",
    },
    {
        "id": "openrouter:openai/gpt-4o-mini",
        "provider": "openrouter",
        "model_name": "openai/gpt-4o-mini",
        "label": "OpenRouter • GPT-4o Mini",
    },
    {
        "id": "openai:gpt-4o-mini",
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "label": "OpenAI • GPT-4o Mini",
    },
    {
        "id": "openai:gpt-4o",
        "provider": "openai",
        "model_name": "gpt-4o",
        "label": "OpenAI • GPT-4o",
    },
]

_DISCOVERED_GOOGLE_MODELS_CACHE: Optional[list[str]] = None
_DISCOVERED_OPENROUTER_FREE_MODELS_CACHE: Optional[list[str]] = None
_UNAVAILABLE_MODELS: set[str] = set()
_WORKING_GOOGLE_MODEL: Optional[str] = None
_WORKING_OPENROUTER_MODEL: Optional[str] = None


def _discover_openrouter_free_models() -> list[str]:
    """Query OpenRouter live catalog for currently active ':free' models, sorting non-reasoning instruct models first."""
    global _DISCOVERED_OPENROUTER_FREE_MODELS_CACHE
    if _DISCOVERED_OPENROUTER_FREE_MODELS_CACHE is not None:
        return list(_DISCOVERED_OPENROUTER_FREE_MODELS_CACHE)

    found: list[str] = []
    try:
        import httpx

        resp = httpx.get(f"{OPENROUTER_BASE_URL}/models", timeout=8.0)
        if resp.status_code < 400:
            items = resp.json().get("data") or []
            raw_ids = [
                str(m.get("id") or "").strip()
                for m in items
                if isinstance(m, dict) and str(m.get("id") or "").strip().endswith(":free")
            ]

            def _rank(mid: str) -> tuple[int, str]:
                low = mid.lower()
                is_reasoning = any(tok in low for tok in ("think", "-r1", "reason", "qwq", "preview"))
                score = 50 if is_reasoning else 0
                if any(tok in low for tok in ("gemma-3-27b", "llama-3.3-70b", "deepseek-chat", "mistral", "qwen")):
                    score -= 20
                elif any(tok in low for tok in ("llama", "gemma", "qwen")):
                    score -= 10
                if "instruct" in low or "-it:" in low:
                    score -= 5
                return (score, low)

            found = sorted(dict.fromkeys(raw_ids), key=_rank)[:12]
    except Exception as exc:
        logger.debug("[AgentSwarm LLM] OpenRouter model discovery skipped: %s", exc)

    if found:
        _DISCOVERED_OPENROUTER_FREE_MODELS_CACHE = found
    return list(found)


def _discover_google_models(api_key: Optional[str] = None) -> list[str]:
    global _DISCOVERED_GOOGLE_MODELS_CACHE
    if _DISCOVERED_GOOGLE_MODELS_CACHE is not None:
        return list(_DISCOVERED_GOOGLE_MODELS_CACHE)

    resolved_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not resolved_key:
        return []

    found: list[str] = []
    try:
        from google import genai

        client = genai.Client(api_key=resolved_key)
        for m in client.models.list():
            name = str(getattr(m, "name", "") or "")
            if name.startswith("models/"):
                name = name[len("models/"):]
            if "gemini" in name.lower() and "embedding" not in name.lower() and "imagen" not in name.lower():
                if name not in found:
                    found.append(name)
    except Exception as exc:
        logger.debug("[AgentSwarm LLM] Google model discovery skipped: %s", exc)

    if found:
        _DISCOVERED_GOOGLE_MODELS_CACHE = found
    return list(found)


def _get_configured_providers() -> dict[str, bool]:
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH, override=True)
    return {
        "google_genai": bool((os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()),
        "openrouter": bool((os.getenv("OPENROUTER_API_KEY") or "").strip()),
        "openai": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
    }


def list_available_models() -> dict[str, Any]:
    configured_providers = _get_configured_providers()
    env_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if env_provider in ("google", "gemini"):
        env_provider = "google_genai"

    if env_provider and configured_providers.get(env_provider):
        default_provider = env_provider
    elif configured_providers["openrouter"]:
        default_provider = "openrouter"
    elif configured_providers["google_genai"]:
        default_provider = "google_genai"
    elif configured_providers["openai"]:
        default_provider = "openai"
    else:
        default_provider = "openrouter"

    discovered_google = _discover_google_models() if configured_providers["google_genai"] else []

    env_model = (os.getenv("LLM_MODEL") or "").strip()
    if env_model:
        default_model = env_model
    elif default_provider == "openrouter":
        default_model = (os.getenv("OPENROUTER_MODEL") or "openrouter/free").strip()
    elif default_provider == "google_genai":
        explicit_gemini = (os.getenv("GEMINI_MODEL") or "").strip()
        if explicit_gemini:
            default_model = explicit_gemini
        elif discovered_google:
            preferred = [
                m for m in ("gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.8-pro", "gemini-2.5-pro")
                if m in discovered_google
            ]
            default_model = preferred[0] if preferred else discovered_google[0]
        else:
            default_model = "gemini-3.8-flash"
    elif default_provider == "openai":
        default_model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    else:
        default_model = ""

    models_out = [{**entry, "configured": bool(configured_providers.get(entry["provider"], False))} for entry in AVAILABLE_MODELS]
    known_ids = {m["id"] for m in models_out}
    for g_model in discovered_google[:8]:
        gid = f"google_genai:{g_model}"
        if gid not in known_ids:
            models_out.append(
                {
                    "id": gid,
                    "provider": "google_genai",
                    "model_name": g_model,
                    "label": f"{g_model} (Verified Key Model)",
                    "configured": True,
                }
            )
            known_ids.add(gid)

    return {
        "default_provider": default_provider,
        "default_model": default_model,
        "configured_providers": configured_providers,
        "models": models_out,
    }


def _extract_json_payload(raw_text: str) -> Any:
    """Extract and parse a JSON object or array from raw LLM output."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.S | re.I).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    for block in re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", cleaned, flags=re.S):
        block_clean = block.strip()
        try:
            parsed = json.loads(block_clean)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass
        for idx, ch in enumerate(block_clean):
            if ch in "{[":
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(block_clean[idx:])
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except Exception:
                    continue

    brace_positions = [idx for idx, ch in enumerate(cleaned) if ch in "{["]
    if brace_positions:
        first_idx = brace_positions[0]
        last_char = "}" if cleaned[first_idx] == "{" else "]"
        last_idx = cleaned.rfind(last_char)
        if last_idx > first_idx:
            try:
                parsed = json.loads(cleaned[first_idx : last_idx + 1])
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                pass
        decoder = json.JSONDecoder()
        for idx in brace_positions:
            try:
                parsed, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                continue

    raise ValueError(f"Could not extract valid JSON object or array from response: {cleaned[:240]}")


def _build_compact_schema_hint(schema: type[BaseModel]) -> str:
    """Return a concise JSON skeleton example so free models adhere cleanly without $defs noise."""
    name = getattr(schema, "__name__", "")
    if name == "TestPlanResult":
        return json.dumps(
            {
                "target_url": "https://example.com",
                "summary": "Concise summary of high-level QA goals for this site",
                "goals": [
                    {
                        "goal_id": "GOAL-01",
                        "title": "Navigate all primary sections",
                        "objective": "Direction only: explore top navigation links and verify each section renders visibly.",
                        "category": "navigation",
                        "priority": "critical | high | medium | low",
                        "success_criteria": "All discovered navigation sections become visible without errors.",
                    }
                ],
            },
            indent=2,
        )
    if name == "NextActionDecision":
        return json.dumps(
            {
                "thought": "Brief reasoning based on live visible elements and recent steps",
                "decision": "act | goal_met | goal_failed",
                "action_type": "click | fill | press | scroll | select | wait_for_visible | assert_visible | assert_url | none",
                "locator_strategy": "role | label | text | placeholder | testid",
                "role": "button | link | textbox | tab | checkbox | combobox",
                "name": "Exact accessible name, label, or text from visible elements list",
                "value": "Input text for fill/select/scroll, or null",
                "description": "Concise human description of this single action",
                "completion_reason": "Required when decision is goal_met or goal_failed",
            },
            indent=2,
        )
    if name == "PersonaSuiteResult":
        return json.dumps(
            {
                "target_url": "https://example.com",
                "personas": [
                    {
                        "persona_id": "PRS-01",
                        "name": "Persona Name",
                        "role": "User Role",
                        "behavior_profile": "methodical",
                        "goals": ["Primary goal"],
                        "synthetic_data": {"email": "qa.synthetic@example.com"},
                        "target_scenarios": ["GOAL-01"],
                        "risk_focus": "Specific edge case focus",
                    }
                ],
            },
            indent=2,
        )
    if name == "TriageDiagnosis":
        return json.dumps(
            {
                "failure_id": "FAIL-001",
                "category": "NAVIGATION_MISSING | OVERLAY_BLOCKING | SELECTOR_DRIFT | BACKEND_API_ERROR | NETWORK_TIMEOUT | ASSERTION_FAILURE | CONSOLE_EXCEPTION",
                "severity": "critical | high | medium | low",
                "confidence": 0.9,
                "root_cause_summary": "Concise root cause",
                "verdict": "Short verdict",
                "needs_healing": False,
                "recommended_next_agent": "healing_agent | human_review | retry_verification",
                "evidence_highlights": ["Supporting clue"],
            },
            indent=2,
        )
    if name == "HealingSuggestion":
        framework_patch = {
            "framework": "playwright | cypress | selenium",
            "healed_selector": "Resilient accessible locator",
            "code_snippet": "Wait for blocking overlay hidden if present, then use getByRole/getByLabel/getByText",
            "explanation": "Why this locator works and never silently skips blockers",
        }
        return json.dumps(
            {
                "failure_id": "FAIL-001",
                "failed_selector": "#broken-selector",
                "primary_healed_locator": "getByRole('button', { name: 'Submit' })",
                "locator_strategy": "aria_role",
                "confidence": 0.9,
                "rationale": "Why this accessible locator resists DOM drift",
                "playwright": {**framework_patch, "framework": "playwright"},
                "cypress": {**framework_patch, "framework": "cypress"},
                "selenium": {**framework_patch, "framework": "selenium"},
            },
            indent=2,
        )
    return json.dumps(schema.model_json_schema(), indent=2)


def _normalize_priority(raw_val: Any, default: str = "medium") -> str:
    val = str(raw_val or "").strip().lower()
    if val in ("critical", "high", "medium", "low"):
        return val
    if val in ("p0", "blocker", "urgent", "highest", "sev1", "s1"):
        return "critical"
    if val in ("p1", "important", "major", "sev2", "s2"):
        return "high"
    if val in ("p2", "normal", "moderate", "standard", "sev3", "s3"):
        return "medium"
    if val in ("p3", "p4", "minor", "trivial", "lowest", "sev4", "s4"):
        return "low"
    return default


def _to_str_list(raw_val: Any) -> list[str]:
    if isinstance(raw_val, str):
        s = raw_val.strip()
        return [s] if s else []
    if isinstance(raw_val, list):
        return [str(x).strip() for x in raw_val if x is not None and str(x).strip()]
    return []


def _unwrap_payload_dict(parsed: Any, wrapper_keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    current: dict[str, Any] = dict(parsed)
    for _ in range(2):
        unwrapped = False
        for key in wrapper_keys:
            inner = current.get(key)
            if isinstance(inner, dict):
                current = dict(inner)
                unwrapped = True
                break
        if unwrapped:
            continue
        if len(current) == 1:
            only_val = next(iter(current.values()))
            if isinstance(only_val, dict):
                current = dict(only_val)
                continue
        break
    return current


def _strip_has_text_css(raw_name: str, raw_role: Optional[str]) -> tuple[str, Optional[str], Optional[str]]:
    """Convert accidental ':has-text("Foo")' output into pure accessible name + strategy."""
    m = re.match(r"^(?:([a-zA-Z0-9_-]+):has-text|:has-text)\(\s*['\"]([^'\"]+)['\"]\s*\)$", raw_name.strip())
    if not m:
        return raw_name, raw_role, None
    tag = (m.group(1) or "").lower()
    txt = m.group(2).strip()
    if tag in ("button", "a") and not raw_role:
        return txt, ("link" if tag == "a" else "button"), "role"
    return txt, raw_role, ("role" if raw_role else "text")


def _coerce_structured_payload(schema: type[BaseModel], parsed: Any) -> Any:
    """Normalize minor free-model JSON variations before Pydantic validation."""
    name = getattr(schema, "__name__", "")

    if name == "TestPlanResult":
        if isinstance(parsed, list):
            data: dict[str, Any] = {"goals": parsed}
        elif isinstance(parsed, dict):
            data = _unwrap_payload_dict(parsed, ("test_plan", "plan", "testPlan", "result", "data", "output", "response"))
            if "goals" not in data:
                for alt_key in ("test_goals", "objectives", "scenarios", "test_scenarios", "cases"):
                    if alt_key in data:
                        data["goals"] = data[alt_key]
                        break
                else:
                    if isinstance(parsed.get("goals"), list):
                        data["goals"] = parsed["goals"]
                    elif isinstance(parsed.get("scenarios"), list):
                        data["goals"] = parsed["scenarios"]
        else:
            raise ValueError("TestPlanResult payload must be a JSON object or array.")

        raw_goals = data.get("goals")
        if not isinstance(raw_goals, list) or not raw_goals:
            raise ValueError("LLM output did not include any test goals.")

        norm_goals: list[dict[str, Any]] = []
        for idx, raw_g in enumerate(raw_goals[:6], start=1):
            if isinstance(raw_g, str):
                g_text = raw_g.strip()
                if not g_text:
                    continue
                norm_goals.append(
                    {
                        "goal_id": f"GOAL-{idx:02d}",
                        "title": g_text[:72],
                        "objective": g_text,
                        "category": "e2e",
                        "priority": "high" if idx == 1 else "medium",
                        "success_criteria": f"{g_text} succeeds without errors.",
                    }
                )
                continue
            if not isinstance(raw_g, dict):
                continue
            title = str(raw_g.get("title") or raw_g.get("name") or raw_g.get("goal") or f"Goal {idx}").strip()
            objective = str(raw_g.get("objective") or raw_g.get("description") or raw_g.get("goal") or title).strip()
            norm_goals.append(
                {
                    "goal_id": str(raw_g.get("goal_id") or raw_g.get("scenario_id") or raw_g.get("id") or f"GOAL-{idx:02d}").strip()
                    or f"GOAL-{idx:02d}",
                    "title": title,
                    "objective": objective,
                    "category": str(raw_g.get("category") or "e2e").strip() or "e2e",
                    "priority": _normalize_priority(raw_g.get("priority"), default="high" if idx == 1 else "medium"),
                    "success_criteria": str(
                        raw_g.get("success_criteria")
                        or raw_g.get("expected_result")
                        or raw_g.get("expected_outcome")
                        or f"Goal '{title}' verified without runtime or overlay errors."
                    ).strip(),
                }
            )

        if not norm_goals:
            raise ValueError("LLM output did not contain any valid test goals.")

        summary = str(data.get("summary") or "").strip() or (
            "Goal-driven QA plan: " + ", ".join(g["title"] for g in norm_goals[:4])
        )
        return {
            "target_url": str(data.get("target_url") or data.get("url") or "").strip(),
            "summary": summary,
            "goals": norm_goals,
            "scenarios": [],
            "generated_by": str(data.get("generated_by") or "langchain"),
        }

    if name == "NextActionDecision":
        if not isinstance(parsed, dict):
            raise ValueError("NextActionDecision payload must be a JSON object.")
        data = _unwrap_payload_dict(parsed, ("decision_result", "next_action", "action_decision", "result", "data", "output"))

        raw_dec = str(data.get("decision") or data.get("status") or "act").strip().lower()
        raw_act = str(data.get("action_type") or data.get("action") or data.get("type") or "click").strip().lower()

        if raw_dec in ("goal_met", "done", "complete", "completed", "passed", "pass", "success") or raw_act == "goal_met":
            decision = "goal_met"
        elif raw_dec in ("goal_failed", "failed", "fail", "stuck", "impossible", "blocked") or raw_act == "goal_failed":
            decision = "goal_failed"
        else:
            decision = "act"

        act_map = {
            "click": "click",
            "press_button": "click",
            "tap": "click",
            "submit": "click",
            "fill": "fill",
            "type": "fill",
            "input": "fill",
            "enter_text": "fill",
            "press": "press",
            "key": "press",
            "scroll": "scroll",
            "scroll_down": "scroll",
            "scroll_up": "scroll",
            "select": "select",
            "wait_for_visible": "wait_for_visible",
            "wait_for_selector": "wait_for_visible",
            "wait": "wait_for_visible",
            "assert_visible": "assert_visible",
            "verify": "assert_visible",
            "check": "assert_visible",
            "assert_url": "assert_url",
            "none": "none",
        }
        action_type = "none" if decision != "act" else act_map.get(raw_act, "assert_visible")

        raw_strat = str(data.get("locator_strategy") or data.get("strategy") or "").strip().lower()
        strat_map = {
            "role": "role",
            "aria": "role",
            "getbyrole": "role",
            "label": "label",
            "getbylabel": "label",
            "text": "text",
            "getbytext": "text",
            "placeholder": "placeholder",
            "getbyplaceholder": "placeholder",
            "testid": "testid",
            "test_id": "testid",
            "getbytestid": "testid",
        }
        role = str(data.get("role") or "").strip().lower() or None
        raw_name = data.get("name") if data.get("name") is not None else data.get("selector")
        clean_name = str(raw_name).strip() if raw_name is not None and str(raw_name).strip() else None
        inferred_strat: Optional[str] = None
        if clean_name:
            clean_name, role, inferred_strat = _strip_has_text_css(clean_name, role)

        locator_strategy = strat_map.get(raw_strat) or inferred_strat or ("role" if role else ("text" if clean_name else None))
        raw_val = data.get("value") if data.get("value") is not None else data.get("input_value")
        val_str = str(raw_val) if raw_val is not None else None
        if raw_act == "scroll_up" and not val_str:
            val_str = "-600"

        return {
            "thought": str(data.get("thought") or data.get("reasoning") or "").strip(),
            "decision": decision,
            "action_type": action_type,
            "locator_strategy": locator_strategy,
            "role": role,
            "name": clean_name,
            "value": val_str,
            "description": str(
                data.get("description") or data.get("intent") or f"{action_type} {clean_name or val_str or ''}".strip()
            ).strip(),
            "completion_reason": str(data.get("completion_reason") or data.get("reason") or "").strip(),
        }

    if name == "PersonaSuiteResult":
        if isinstance(parsed, list):
            data = {"personas": parsed}
        elif isinstance(parsed, dict):
            data = _unwrap_payload_dict(parsed, ("persona_suite", "suite", "result", "data", "output", "response"))
            if "personas" not in data:
                for alt_key in ("user_personas", "users", "profiles"):
                    if alt_key in data:
                        data["personas"] = data[alt_key]
                        break
        else:
            raise ValueError("PersonaSuiteResult payload must be a JSON object or array.")

        raw_personas = data.get("personas")
        if not isinstance(raw_personas, list) or not raw_personas:
            raise ValueError("LLM output did not include any personas.")

        norm_personas: list[dict[str, Any]] = []
        for idx, raw_p in enumerate(raw_personas, start=1):
            if not isinstance(raw_p, dict):
                continue
            synth = raw_p.get("synthetic_data")
            norm_personas.append(
                {
                    "persona_id": str(raw_p.get("persona_id") or raw_p.get("id") or f"PRS-{idx:02d}").strip() or f"PRS-{idx:02d}",
                    "name": str(raw_p.get("name") or f"Persona {idx}").strip(),
                    "role": str(raw_p.get("role") or "QA Explorer").strip(),
                    "behavior_profile": str(raw_p.get("behavior_profile") or raw_p.get("behavior") or "methodical").strip(),
                    "goals": _to_str_list(raw_p.get("goals")),
                    "synthetic_data": dict(synth) if isinstance(synth, dict) else {},
                    "target_scenarios": _to_str_list(raw_p.get("target_scenarios") or raw_p.get("scenarios")),
                    "risk_focus": str(raw_p.get("risk_focus") or "").strip(),
                }
            )
        if not norm_personas:
            raise ValueError("LLM output did not contain any valid personas.")

        return {
            "target_url": str(data.get("target_url") or data.get("url") or "").strip(),
            "personas": norm_personas,
            "generated_by": str(data.get("generated_by") or "langchain"),
        }

    if name == "TriageDiagnosis" and isinstance(parsed, dict):
        data = _unwrap_payload_dict(parsed, ("diagnosis", "triage_diagnosis", "triage", "result", "data", "output"))
        raw_cat = str(data.get("category") or "ASSERTION_FAILURE").strip().upper().replace(" ", "_")
        allowed_cats = {
            "NAVIGATION_MISSING",
            "OVERLAY_BLOCKING",
            "SELECTOR_DRIFT",
            "BACKEND_API_ERROR",
            "NETWORK_TIMEOUT",
            "ASSERTION_FAILURE",
            "CONSOLE_EXCEPTION",
        }
        if raw_cat not in allowed_cats:
            if "NAVIGATION" in raw_cat or "BLANK" in raw_cat:
                raw_cat = "NAVIGATION_MISSING"
            elif "OVERLAY" in raw_cat or "MODAL" in raw_cat:
                raw_cat = "OVERLAY_BLOCKING"
            elif "SELECTOR" in raw_cat or "LOCATOR" in raw_cat:
                raw_cat = "SELECTOR_DRIFT"
            elif "TIMEOUT" in raw_cat or "NETWORK" in raw_cat:
                raw_cat = "NETWORK_TIMEOUT"
            elif "CONSOLE" in raw_cat or "JS" in raw_cat:
                raw_cat = "CONSOLE_EXCEPTION"
            elif "API" in raw_cat or "BACKEND" in raw_cat or "500" in raw_cat:
                raw_cat = "BACKEND_API_ERROR"
            else:
                raw_cat = "ASSERTION_FAILURE"
        needs_healing = bool(data.get("needs_healing", raw_cat == "SELECTOR_DRIFT"))
        raw_next = str(data.get("recommended_next_agent") or ("healing_agent" if needs_healing else "human_review")).strip().lower()
        if raw_next not in ("healing_agent", "human_review", "retry_verification"):
            raw_next = "healing_agent" if needs_healing else "human_review"
        try:
            conf = float(data.get("confidence", 0.7))
        except Exception:
            conf = 0.7
        return {
            "failure_id": str(data.get("failure_id") or "FAIL-001").strip(),
            "category": raw_cat,
            "severity": _normalize_priority(data.get("severity"), default="medium"),
            "confidence": max(0.0, min(1.0, conf)),
            "root_cause_summary": str(data.get("root_cause_summary") or data.get("summary") or "Failure diagnosed by TriageAgent.").strip(),
            "verdict": str(data.get("verdict") or raw_cat).strip(),
            "needs_healing": needs_healing,
            "recommended_next_agent": raw_next,
            "evidence_highlights": _to_str_list(data.get("evidence_highlights")),
        }

    if name == "HealingSuggestion" and isinstance(parsed, dict):
        data = _unwrap_payload_dict(parsed, ("healing_suggestion", "healing", "suggestion", "result", "data", "output"))
        primary = str(data.get("primary_healed_locator") or data.get("healed_selector") or "getByRole('button')").strip()
        try:
            conf = float(data.get("confidence", 0.8))
        except Exception:
            conf = 0.8

        def _coerce_fw(fw_name: str, default_code: str) -> dict[str, str]:
            raw_fw = data.get(fw_name)
            if isinstance(raw_fw, dict):
                return {
                    "framework": fw_name,
                    "healed_selector": str(raw_fw.get("healed_selector") or primary).strip(),
                    "code_snippet": str(raw_fw.get("code_snippet") or default_code).strip(),
                    "explanation": str(raw_fw.get("explanation") or f"Resilient {fw_name} locator.").strip(),
                }
            return {
                "framework": fw_name,
                "healed_selector": primary,
                "code_snippet": default_code,
                "explanation": f"Resilient {fw_name} locator.",
            }

        return {
            "failure_id": str(data.get("failure_id") or "FAIL-001").strip(),
            "failed_selector": str(data.get("failed_selector") or "").strip() or "unknown",
            "primary_healed_locator": primary,
            "locator_strategy": str(data.get("locator_strategy") or "aria_role").strip(),
            "confidence": max(0.0, min(1.0, conf)),
            "rationale": str(data.get("rationale") or "Resilient accessible selector synthesized from DOM context.").strip(),
            "playwright": _coerce_fw("playwright", f"await page.{primary}.click();"),
            "cypress": _coerce_fw("cypress", f"cy.findByRole('button').click();"),
            "selenium": _coerce_fw("selenium", "driver.find_element(By.CSS_SELECTOR, 'button').click()"),
        }

    return parsed


class _OpenRouterChatModel(BaseChatModel):
    """httpx-powered LangChain chat model for OpenRouter with timeout & transient retry support."""

    model_name: str
    api_key: str
    temperature: float = 0.2
    base_url: str = OPENROUTER_BASE_URL
    timeout_s: float = 35.0
    max_attempts: int = 2

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

    @staticmethod
    def _convert_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for msg in messages:
            role = "user"
            msg_type = getattr(msg, "type", "")
            if msg_type == "system":
                role = "system"
            elif msg_type == "ai":
                role = "assistant"
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            out.append({"role": role, "content": content})
        return out

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        import httpx

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                resp = httpx.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "AgentSwarm QA",
                    },
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": 2048,
                    },
                    timeout=self.timeout_s,
                )
                if resp.status_code in (429, 502, 503, 504) and attempt + 1 < self.max_attempts:
                    time.sleep(1.5)
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"OpenRouter HTTP {resp.status_code} for '{self.model_name}': {resp.text[:400]}")
                data = resp.json()
                if "error" in data and data["error"]:
                    raise RuntimeError(f"OpenRouter error for '{self.model_name}': {data['error']}")
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"OpenRouter returned empty choices for '{self.model_name}': {str(data)[:300]}")
                msg_obj = choices[0].get("message") or {}
                content = msg_obj.get("content") or ""
                if isinstance(content, list):
                    content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
                text_out = str(content).strip()
                if not text_out:
                    reasoning = str(msg_obj.get("reasoning") or msg_obj.get("reasoning_content") or "").strip()
                    if reasoning and ("{" in reasoning or "[" in reasoning):
                        return reasoning
                    if attempt + 1 < self.max_attempts:
                        time.sleep(1.5)
                        continue
                    finish_reason = choices[0].get("finish_reason")
                    raise RuntimeError(
                        f"OpenRouter returned empty content for '{self.model_name}' (finish_reason={finish_reason})"
                    )
                return str(content)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(1.5)
                    continue
                raise RuntimeError(f"OpenRouter timeout/transport error for '{self.model_name}': {exc}") from exc

        raise RuntimeError(f"OpenRouter request failed for '{self.model_name}': {last_exc}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._call_api(self._convert_messages(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        schema_hint = _build_compact_schema_hint(schema)

        def _invoke_structured(prompt_val: Any) -> Any:
            if hasattr(prompt_val, "to_messages"):
                lc_msgs = prompt_val.to_messages()
            elif isinstance(prompt_val, list):
                lc_msgs = prompt_val
            else:
                lc_msgs = [HumanMessage(content=str(prompt_val))]

            api_msgs = self._convert_messages(lc_msgs)
            instruction = (
                "\n\nIMPORTANT OUTPUT REQUIREMENT:\n"
                "Return ONLY a single valid JSON object (no markdown fences, no commentary, no extra text) "
                "matching this JSON structure:\n"
                f"{schema_hint}"
            )
            if api_msgs and api_msgs[-1]["role"] == "user":
                api_msgs[-1] = {**api_msgs[-1], "content": api_msgs[-1]["content"] + instruction}
            else:
                api_msgs.append({"role": "user", "content": instruction})

            raw_text = self._call_api(api_msgs)
            parsed = _extract_json_payload(raw_text)
            coerced = _coerce_structured_payload(schema, parsed)
            return schema.model_validate(coerced)

        return RunnableLambda(_invoke_structured)


def resolve_chat_model(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> Optional[BaseChatModel]:
    """Create a LangChain BaseChatModel from explicit arguments or backend/.env."""
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH, override=True)

    parsed_provider, parsed_model = parse_provider_and_model(provider, model_name)
    if parsed_provider in ("deterministic", "local", "offline") or parsed_model == "deterministic-fallback":
        return None

    raw_provider = (parsed_provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    explicit_key = (api_key or "").strip()

    if not raw_provider:
        if explicit_key.startswith("sk-or-"):
            raw_provider = "openrouter"
        elif explicit_key.startswith("AIza"):
            raw_provider = "google_genai"
        elif explicit_key.startswith("sk-"):
            raw_provider = "openai"
        elif os.getenv("OPENROUTER_API_KEY"):
            raw_provider = "openrouter"
        elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            raw_provider = "google_genai"
        elif os.getenv("OPENAI_API_KEY"):
            raw_provider = "openai"

    if not raw_provider or raw_provider in ("deterministic", "local", "offline"):
        return None

    try:
        if raw_provider == "openrouter":
            resolved_key = (explicit_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
            if not resolved_key:
                return None
            resolved_model = (
                parsed_model or os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or "openrouter/free"
            ).strip()
            if resolved_model in _UNAVAILABLE_MODELS and _WORKING_OPENROUTER_MODEL:
                resolved_model = _WORKING_OPENROUTER_MODEL
            return _OpenRouterChatModel(model_name=resolved_model, api_key=resolved_key, temperature=temperature)

        if raw_provider in ("google", "gemini", "google_genai"):
            resolved_key = (explicit_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
            if not resolved_key:
                return None
            resolved_model = (parsed_model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash").strip()
            if resolved_model in _UNAVAILABLE_MODELS and _WORKING_GOOGLE_MODEL:
                resolved_model = _WORKING_GOOGLE_MODEL
            return init_chat_model(
                resolved_model,
                model_provider="google_genai",
                api_key=resolved_key,
                temperature=temperature,
                max_retries=1,
                timeout=18,
            )

        if raw_provider == "openai":
            resolved_key = (explicit_key or os.getenv("OPENAI_API_KEY") or "").strip()
            if not resolved_key:
                return None
            resolved_model = (parsed_model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
            return init_chat_model(
                resolved_model,
                model_provider="openai",
                api_key=resolved_key,
                temperature=temperature,
                max_retries=1,
                timeout=18,
            )

        resolved_model = (parsed_model or os.getenv("LLM_MODEL") or "").strip()
        if not resolved_model:
            return None
        kwargs: dict[str, Any] = {"model_provider": raw_provider, "temperature": temperature}
        if explicit_key:
            kwargs["api_key"] = explicit_key
        return init_chat_model(resolved_model, **kwargs)
    except Exception as exc:
        logger.warning("[AgentSwarm LLM] Failed to initialize provider '%s': %s", raw_provider, exc)
        return None


def invoke_with_model_fallback(
    prompt: ChatPromptTemplate,
    schema: type[BaseModel],
    prompt_input: dict[str, Any],
    *,
    primary_chain: Optional[Any],
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    allow_fallback: bool = True,
) -> tuple[Any, Optional[str]]:
    """Invoke a structured-output chain and recover cleanly across 404/503/429/timeout/JSON errors."""
    global _WORKING_GOOGLE_MODEL, _WORKING_OPENROUTER_MODEL
    if primary_chain is None:
        raise RuntimeError("LangChain structured output chain is not configured.")

    eff_provider, eff_model = parse_provider_and_model(provider, model_name)
    if not eff_provider:
        if _ENV_PATH.exists():
            load_dotenv(dotenv_path=_ENV_PATH, override=True)
        if os.getenv("OPENROUTER_API_KEY"):
            eff_provider = "openrouter"
        elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            eff_provider = "google_genai"
        elif os.getenv("OPENAI_API_KEY"):
            eff_provider = "openai"

    failed_model = (
        eff_model
        or (
            _WORKING_OPENROUTER_MODEL or "openrouter/free"
            if eff_provider == "openrouter"
            else (_WORKING_GOOGLE_MODEL or "gemini-3.8-flash")
        )
    ).strip()

    if eff_provider == "openrouter" and failed_model in _UNAVAILABLE_MODELS and _WORKING_OPENROUTER_MODEL and _WORKING_OPENROUTER_MODEL not in _UNAVAILABLE_MODELS:
        alt_llm = resolve_chat_model(model_name=_WORKING_OPENROUTER_MODEL, provider="openrouter", api_key=api_key, temperature=temperature)
        if alt_llm:
            primary_chain = prompt | alt_llm.with_structured_output(schema)
            failed_model = _WORKING_OPENROUTER_MODEL
    elif eff_provider == "google_genai" and failed_model in _UNAVAILABLE_MODELS and _WORKING_GOOGLE_MODEL and _WORKING_GOOGLE_MODEL not in _UNAVAILABLE_MODELS:
        alt_llm = resolve_chat_model(model_name=_WORKING_GOOGLE_MODEL, provider="google_genai", api_key=api_key, temperature=temperature)
        if alt_llm:
            primary_chain = prompt | alt_llm.with_structured_output(schema)
            failed_model = _WORKING_GOOGLE_MODEL

    try:
        return primary_chain.invoke(prompt_input), failed_model
    except Exception as exc:
        err_str = str(exc)
        err_low = err_str.lower()
        is_recoverable = isinstance(exc, (ValueError, ValidationError)) or any(
            tok in err_low
            for tok in (
                "404", "not_found", "no longer available", "is not found", "no endpoints found",
                "provider returned error", "500", "502", "503", "504", "bad gateway", "unavailable",
                "429", "resource_exhausted", "rate limit", "overloaded", "high demand",
                "timed out", "timeout", "readtimeout", "connect", "remoteprotocolerror",
                "could not extract valid json", "validation error", "field required", "invalid json", "empty choices", "empty content",
            )
        )
        if not allow_fallback or not is_recoverable:
            raise

        is_not_found = any(tok in err_low for tok in ("404", "not_found", "no longer available", "is not found", "no endpoints found"))
        if failed_model != "openrouter/free" or is_not_found:
            _UNAVAILABLE_MODELS.add(failed_model)

        if eff_provider == "openrouter":
            pool = [
                *_discover_openrouter_free_models(),
                "google/gemma-3-27b-it:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3-0324:free",
                "openrouter/free",
            ]
        elif eff_provider == "google_genai":
            suggested = re.findall(r"models/([a-zA-Z0-9._-]+)", err_str)
            pool = [*suggested, "gemini-2.0-flash", "gemini-2.5-pro", "gemini-3.8-pro", "gemini-3.8-flash", *_discover_google_models(api_key)]
        else:
            raise

        candidates: list[str] = []
        for cand in pool:
            clean = cand.strip()
            if clean and clean != failed_model and clean not in _UNAVAILABLE_MODELS and clean not in candidates:
                candidates.append(clean)

        tried_models = [failed_model]
        last_exc: Exception = exc
        for cand in candidates[:5]:
            alt_llm = resolve_chat_model(model_name=cand, provider=eff_provider, api_key=api_key, temperature=temperature)
            if not alt_llm:
                continue
            tried_models.append(cand)
            try:
                res = (prompt | alt_llm.with_structured_output(schema)).invoke(prompt_input)
                if eff_provider == "google_genai":
                    _WORKING_GOOGLE_MODEL = cand
                elif eff_provider == "openrouter":
                    _WORKING_OPENROUTER_MODEL = cand
                logger.info("[AgentSwarm LLM] Recovered from '%s' error -> using '%s'", failed_model, cand)
                return res, cand
            except Exception as cand_exc:
                cand_low = str(cand_exc).lower()
                if cand != "openrouter/free" or any(t in cand_low for t in ("404", "not_found", "no endpoints found")):
                    _UNAVAILABLE_MODELS.add(cand)
                last_exc = cand_exc
        raise RuntimeError(f"All AI models failed ({', '.join(tried_models)}): {last_exc}") from last_exc


class TestGoal(BaseModel):
    """Directional high-level QA goal generated without pre-written selectors or fixed scripts."""

    goal_id: str = Field(..., description="Unique goal identifier (e.g., GOAL-01)")
    title: str = Field(..., description="Concise directional goal title")
    objective: str = Field(..., description="Direction only describing what capability/area to exercise")
    category: str = Field("e2e", description="Category such as navigation, theme, interactive, form, smoke")
    priority: Literal["critical", "high", "medium", "low"] = Field("high")
    success_criteria: str = Field("", description="Observable outcome indicating this goal is satisfied")


class TestStepPlan(BaseModel):
    """Lightweight step record model kept for schema compatibility (empty on goal-only plans)."""

    step_number: int = Field(1, ge=1)
    action: str = "assert_visible"
    description: str = ""
    target_hint: Optional[str] = None
    input_value: Optional[str] = None
    expected_outcome: str = ""


class TestScenario(BaseModel):
    """Compatibility mirror of TestGoal with empty steps (goals drive live exploration)."""

    scenario_id: str
    title: str
    category: str = "e2e"
    priority: Literal["critical", "high", "medium", "low"] = "high"
    description: str
    preconditions: list[str] = Field(default_factory=list)
    steps: list[TestStepPlan] = Field(default_factory=list)
    expected_result: str = ""


class TestPlanResult(BaseModel):
    """High-level goal-oriented QA plan for a target web application."""

    target_url: str = Field(..., description="Normalized URL of the application under test")
    summary: str = Field(..., description="High-level summary of the goal-oriented QA strategy")
    goals: list[TestGoal] = Field(default_factory=list, description="Directional QA goals (no fixed steps)")
    scenarios: list[TestScenario] = Field(default_factory=list, description="Goal mirror for UI compatibility")
    generated_by: str = Field("langchain", description="Origin engine identifier")


class PlannerGraphState(TypedDict, total=False):
    target_url: str
    normalized_url: str
    dom_summary: Optional[str]
    objectives: list[str]
    prompt_input: dict[str, str]
    test_plan: TestPlanResult
    planner_error: Optional[str]


class PlannerAgent:
    """
    Autonomous goal-setting QA Planner powered by LangChain & LangGraph.

    Outputs directional high-level goals only (no CSS selectors, no scripted steps).
    NavigatorAgent explores the live DOM step-by-step to satisfy each goal.
    """

    SYSTEM_PROMPT = (
        "You are the Lead QA Goal Planner for AgentSwarm. Inspect the live page summary and output "
        "a concise list of 3 to 4 high-level directional QA goals tailored to this specific website.\n\n"
        "Strict Rules:\n"
        "1. Goals are DIRECTION ONLY — do NOT include CSS selectors, locators, or fixed step-by-step scripts.\n"
        "2. Base goals strictly on capabilities visibly present in the page summary (navigation links/sections, "
        "theme/mode toggle, tabs/cards/modals, forms/contact inputs, social links, etc.).\n"
        "3. Do NOT invent login/checkout goals when no login/cart controls exist.\n"
        "4. Include clear success_criteria for when an autonomous browser agent should consider the goal met."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._provider = provider
        self._custom_llm_injected = llm is not None
        self._llm: Optional[BaseChatModel] = llm or resolve_chat_model(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            temperature=0.2,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    "Target URL: {target_url}\n\n"
                    "Explicit Testing Objectives (include if given):\n{objectives_text}\n\n"
                    "Post-Preflight Live Page Summary:\n{dom_summary}\n\n"
                    "Generate 3 to 4 concise high-level QA goals tailored to this site (direction only, zero selectors, zero scripts).",
                ),
            ]
        )
        self._chain = (self._prompt | self._llm.with_structured_output(TestPlanResult)) if self._llm else None
        self._graph = self.build_graph()

    @staticmethod
    def _normalize_url(target_url: str) -> str:
        url = (target_url or "").strip()
        if not url:
            return "https://example.com"
        return url if url.startswith(("http://", "https://")) else f"https://{url}"

    def _invoke_chain(self, prompt_input: dict[str, str]) -> TestPlanResult:
        result, resolved_model = invoke_with_model_fallback(
            self._prompt,
            TestPlanResult,
            prompt_input,
            primary_chain=self._chain,
            provider=self._provider,
            model_name=self._model_name,
            api_key=self._api_key,
            temperature=0.2,
            allow_fallback=not self._custom_llm_injected,
        )
        if not isinstance(result, TestPlanResult):
            result = TestPlanResult.model_validate(result)
        if not result.goals:
            raise ValueError("LLM returned zero QA goals.")
        result.generated_by = f"langchain:{resolved_model}" if resolved_model else "langchain"
        return result

    def _node_prepare_input(self, state: PlannerGraphState) -> dict[str, Any]:
        normalized_url = self._normalize_url(state.get("target_url", ""))
        objectives = state.get("objectives") or []
        objectives_text = "\n".join(f"- {obj}" for obj in objectives) if objectives else "- Autonomously test primary features visible on the page"
        dom_summary = (state.get("dom_summary") or "No live summary provided")[:4500]
        return {
            "normalized_url": normalized_url,
            "prompt_input": {
                "target_url": normalized_url,
                "objectives_text": objectives_text,
                "dom_summary": dom_summary,
            },
        }

    def _node_generate_plan(self, state: PlannerGraphState) -> dict[str, Any]:
        if not self._chain:
            raise RuntimeError(
                "No active AI LLM configured. Set OPENROUTER_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY "
                "to generate a website-specific QA goal plan."
            )
        try:
            plan = self._invoke_chain(state["prompt_input"])
            return {"test_plan": plan, "planner_error": None}
        except Exception as exc:
            logger.error("[PlannerAgent] AI goal planning failed across models: %s", exc)
            raise RuntimeError(f"AI goal planning failed (no fallback plan generated): {exc}") from exc

    @staticmethod
    def _node_finalize_plan(state: PlannerGraphState) -> dict[str, Any]:
        plan = state["test_plan"]
        plan.target_url = state["normalized_url"]
        if not plan.goals and plan.scenarios:
            plan.goals = [
                TestGoal(
                    goal_id=sc.scenario_id,
                    title=sc.title,
                    objective=sc.description,
                    category=sc.category,
                    priority=sc.priority,
                    success_criteria=sc.expected_result,
                )
                for sc in plan.scenarios
            ]
        # Mirror goals as empty-step scenarios so UI/API schema consumers work cleanly without pre-written scripts
        plan.scenarios = [
            TestScenario(
                scenario_id=g.goal_id,
                title=g.title,
                category=g.category,
                priority=g.priority,
                description=g.objective,
                preconditions=["Single continuous browser session ready on target URL"],
                steps=[],
                expected_result=g.success_criteria,
            )
            for g in plan.goals
        ]
        return {"test_plan": plan}

    def build_graph(self):
        builder = StateGraph(PlannerGraphState)
        builder.add_node("prepare_input", self._node_prepare_input)
        builder.add_node("generate_plan", self._node_generate_plan)
        builder.add_node("finalize_plan", self._node_finalize_plan)
        builder.add_edge(START, "prepare_input")
        builder.add_edge("prepare_input", "generate_plan")
        builder.add_edge("generate_plan", "finalize_plan")
        builder.add_edge("finalize_plan", END)
        return builder.compile()

    def create_test_plan(
        self,
        target_url: str,
        dom_summary: Optional[str] = None,
        objectives: Optional[list[str]] = None,
    ) -> TestPlanResult:
        """Execute the compiled LangGraph planner workflow to produce a goal-oriented TestPlanResult."""
        result_state = self._graph.invoke(
            {
                "target_url": target_url,
                "dom_summary": dom_summary,
                "objectives": objectives or [],
            }
        )
        return result_state["test_plan"]

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        target_url = state.get("target_url", "https://example.com")
        dom_summary = state.get("dom_summary") or state.get("dom_snapshot")
        objectives = state.get("objectives")
        result_state = self._graph.invoke(
            {
                "target_url": target_url,
                "dom_summary": dom_summary,
                "objectives": objectives or [],
            }
        )
        plan = result_state["test_plan"]
        return {
            "target_url": plan.target_url,
            "test_plan": plan.model_dump(),
            "planner_error": result_state.get("planner_error"),
            "current_agent": "planner",
        }


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    return PlannerAgent().run_node(state)
