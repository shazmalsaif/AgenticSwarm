from __future__ import annotations

import json
import logging
import os
import re
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

# Load environment variables from backend/.env if present
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
    """
    Parse provider and model_name without breaking model suffixes like ':free'.

    Examples:
    - ('openrouter', 'baidu/ernie-5.5-flash-preview:free') -> ('openrouter', 'baidu/ernie-5.5-flash-preview:free')
    - (None, 'openrouter:baidu/ernie-5.5-flash-preview:free') -> ('openrouter', 'baidu/ernie-5.5-flash-preview:free')
    - (None, 'baidu/ernie-5.5-flash-preview:free') -> ('openrouter', 'baidu/ernie-5.5-flash-preview:free')
    """
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
        "label": "Gemini 2.5 Flash — Fast & Recommended",
        "label": "Gemini 2.5 Flash",
    },
    {
        "id": "google_genai:gemini-2.5-pro",
        "provider": "google_genai",
        "model_name": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro — Deep Reasoning",
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
    {
        "id": "deterministic:deterministic-fallback",
        "provider": "deterministic",
        "model_name": "deterministic-fallback",
        "label": "Deterministic DOM Engine (Offline / Instant)",
    },
]

_DISCOVERED_GOOGLE_MODELS_CACHE: Optional[list[str]] = None
_UNAVAILABLE_MODELS: set[str] = set()
_WORKING_GOOGLE_MODEL: Optional[str] = None
_WORKING_OPENROUTER_MODEL: Optional[str] = None


def _discover_google_models(api_key: Optional[str] = None) -> list[str]:
    """Query Google GenAI API once to discover Gemini model IDs supported by the current key."""
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
        "deterministic": True,
    }


def list_available_models() -> dict[str, Any]:
    """Return catalog of selectable LLM providers/models and local API key readiness."""
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
        default_provider = "deterministic"

    discovered_google: list[str] = []
    if configured_providers["google_genai"]:
        discovered_google = _discover_google_models()

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
        default_model = "deterministic-fallback"

    models_out = [
        {
            **entry,
            "configured": bool(configured_providers.get(entry["provider"], False)),
        }
        for entry in AVAILABLE_MODELS
    ]

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
    """Extract and parse a JSON object or array from raw LLM output (handling <think> tags and fences)."""
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
    """Return a concise, concrete JSON skeleton example without verbose $defs/$ref noise."""
    name = getattr(schema, "__name__", "")
    if name == "TestPlanResult":
        return json.dumps(
            {
                "target_url": "https://example.com",
                "summary": "Concise summary of tailored QA strategy",
                "scenarios": [
                    {
                        "scenario_id": "SCN-001",
                        "title": "Descriptive site-specific scenario title",
                        "category": "smoke",
                        "priority": "critical | high | medium | low",
                        "description": "What this scenario validates on this exact site",
                        "preconditions": ["Site is reachable"],
                        "steps": [
                            {
                                "step_number": 1,
                                "action": "navigate | click | fill | wait | assert_visible | assert_url",
                                "description": "Human-readable intent",
                                "target_hint": "#exact-selector-from-dom-or-url",
                                "input_value": None,
                                "expected_outcome": "Expected observable result",
                            }
                        ],
                        "expected_result": "Final acceptance condition",
                    }
                ],
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
                        "target_scenarios": ["SCN-001"],
                        "risk_focus": "Key edge case or failure mode",
                    }
                ],
            },
            indent=2,
        )
    if name == "_BrowserActionPlan":
        return json.dumps(
            {
                "actions": [
                    {
                        "step_number": 1,
                        "action_type": "goto | click | fill | wait_for_selector | assert_visible | assert_url",
                        "selector": "#css-or-aria-selector-or-null",
                        "fallback_selectors": [],
                        "value": "URL or text input or null",
                        "description": "Human-readable action intent",
                        "timeout_ms": 5000,
                    }
                ]
            },
            indent=2,
        )
    if name == "TriageDiagnosis":
        return json.dumps(
            {
                "failure_id": "FAIL-001",
                "category": "SELECTOR_DRIFT | BACKEND_API_ERROR | NETWORK_TIMEOUT | ASSERTION_FAILURE | CONSOLE_EXCEPTION",
                "severity": "critical | high | medium | low",
                "confidence": 0.9,
                "root_cause_summary": "Concise root cause",
                "verdict": "Short verdict",
                "needs_healing": True,
                "recommended_next_agent": "healing_agent | human_review | retry_verification",
                "evidence_highlights": ["Supporting log or DOM clue"],
            },
            indent=2,
        )
    if name == "HealingSuggestion":
        framework_patch = {
            "framework": "playwright | cypress | selenium",
            "healed_selector": "Resilient selector",
            "code_snippet": "Framework code snippet",
            "explanation": "Why this selector works",
        }
        return json.dumps(
            {
                "failure_id": "FAIL-001",
                "failed_selector": "#broken-selector",
                "primary_healed_locator": "getByRole('button', { name: 'Submit' })",
                "locator_strategy": "aria_role",
                "confidence": 0.9,
                "rationale": "Why this locator resists DOM drift",
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


def _to_positive_int(raw_val: Any, default: int, minimum: int = 1) -> int:
    if isinstance(raw_val, bool):
        return max(minimum, default)
    if isinstance(raw_val, (int, float)):
        return max(minimum, int(raw_val))
    if isinstance(raw_val, str):
        digits = re.findall(r"-?\d+", raw_val)
        if digits:
            return max(minimum, int(digits[0]))
    return max(minimum, default)


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


def _coerce_structured_payload(schema: type[BaseModel], parsed: Any) -> Any:
    """Normalize minor free-model JSON formatting variations before strict Pydantic validation."""
    name = getattr(schema, "__name__", "")

    if name == "TestPlanResult":
        if isinstance(parsed, list):
            data: dict[str, Any] = {"scenarios": parsed}
        elif isinstance(parsed, dict):
            data = _unwrap_payload_dict(
                parsed,
                ("test_plan", "plan", "testPlan", "result", "data", "output", "response"),
            )
            if "scenarios" not in data:
                for alt_key in ("test_scenarios", "testScenarios", "test_cases", "testCases", "cases", "tests"):
                    if alt_key in data:
                        data["scenarios"] = data[alt_key]
                        break
                else:
                    if isinstance(parsed.get("scenarios"), list):
                        data["scenarios"] = parsed["scenarios"]
                    elif "steps" in data and ("title" in data or "name" in data):
                        data = {"scenarios": [data]}
        else:
            raise ValueError("TestPlanResult payload must be a JSON object or array.")

        raw_scenarios = data.get("scenarios")
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ValueError("LLM output did not include any test scenarios.")

        norm_scenarios: list[dict[str, Any]] = []
        for idx, raw_sc in enumerate(raw_scenarios, start=1):
            if isinstance(raw_sc, str):
                sc: dict[str, Any] = {"title": raw_sc}
            elif isinstance(raw_sc, dict):
                sc = dict(raw_sc)
            else:
                continue

            title = str(sc.get("title") or sc.get("name") or sc.get("scenario_name") or f"Scenario {idx}").strip()
            raw_steps = sc.get("steps") or sc.get("actions") or []
            if isinstance(raw_steps, dict):
                raw_steps = [raw_steps]
            elif isinstance(raw_steps, str):
                raw_steps = [raw_steps]
            elif not isinstance(raw_steps, list):
                raw_steps = []

            norm_steps: list[dict[str, Any]] = []
            for s_idx, raw_st in enumerate(raw_steps, start=1):
                if isinstance(raw_st, str):
                    st_text = raw_st.strip()
                    if not st_text:
                        continue
                    norm_steps.append(
                        {
                            "step_number": s_idx,
                            "action": "assert_visible",
                            "description": st_text,
                            "target_hint": None,
                            "input_value": None,
                            "expected_outcome": "",
                        }
                    )
                    continue
                if not isinstance(raw_st, dict):
                    continue
                action = (
                    str(
                        raw_st.get("action")
                        or raw_st.get("action_type")
                        or raw_st.get("type")
                        or raw_st.get("verb")
                        or "click"
                    )
                    .strip()
                    .lower()
                    or "click"
                )
                raw_target = (
                    raw_st.get("target_hint")
                    or raw_st.get("selector")
                    or raw_st.get("locator")
                    or raw_st.get("target")
                    or raw_st.get("element")
                )
                target_hint = str(raw_target).strip() if raw_target is not None and str(raw_target).strip() else None
                raw_input = (
                    raw_st.get("input_value")
                    if raw_st.get("input_value") is not None
                    else (raw_st.get("value") if raw_st.get("value") is not None else raw_st.get("input"))
                )
                input_value = str(raw_input) if raw_input is not None else None
                desc = str(
                    raw_st.get("description")
                    or raw_st.get("intent")
                    or raw_st.get("instruction")
                    or raw_st.get("name")
                    or f"{action} {target_hint or ''}".strip()
                ).strip() or f"Step {s_idx}"
                raw_outcome = (
                    raw_st.get("expected_outcome")
                    or raw_st.get("expected_result")
                    or raw_st.get("expected")
                    or ""
                )
                norm_steps.append(
                    {
                        "step_number": _to_positive_int(
                            raw_st.get("step_number") or raw_st.get("step") or raw_st.get("order") or raw_st.get("index"),
                            default=s_idx,
                        ),
                        "action": action,
                        "description": desc,
                        "target_hint": target_hint,
                        "input_value": input_value,
                        "expected_outcome": str(raw_outcome).strip(),
                    }
                )

            norm_scenarios.append(
                {
                    "scenario_id": str(sc.get("scenario_id") or sc.get("id") or f"SCN-{idx:03d}").strip() or f"SCN-{idx:03d}",
                    "title": title,
                    "category": str(sc.get("category") or sc.get("type") or "e2e").strip() or "e2e",
                    "priority": _normalize_priority(sc.get("priority"), default="high" if idx == 1 else "medium"),
                    "description": str(sc.get("description") or sc.get("objective") or sc.get("scope") or title).strip() or title,
                    "preconditions": _to_str_list(sc.get("preconditions")),
                    "steps": norm_steps,
                    "expected_result": str(
                        sc.get("expected_result")
                        or sc.get("expected_outcome")
                        or sc.get("acceptance_criteria")
                        or sc.get("expected")
                        or "Scenario completes without console or navigation errors."
                    ).strip(),
                }
            )

        if not norm_scenarios:
            raise ValueError("LLM output did not contain any valid test scenarios.")

        summary = str(data.get("summary") or "").strip()
        if not summary:
            summary = "Tailored test plan covering: " + ", ".join(s["title"] for s in norm_scenarios[:4])

        return {
            "target_url": str(data.get("target_url") or data.get("url") or "").strip(),
            "summary": summary,
            "scenarios": norm_scenarios,
            "generated_by": str(data.get("generated_by") or "langchain"),
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
                    if isinstance(parsed.get("personas"), list):
                        data["personas"] = parsed["personas"]
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

    if name == "_BrowserActionPlan":
        if isinstance(parsed, list):
            data = {"actions": parsed}
        elif isinstance(parsed, dict):
            data = _unwrap_payload_dict(parsed, ("action_plan", "browser_action_plan", "plan", "result", "data", "output", "response"))
            if "actions" not in data:
                for alt_key in ("steps", "browser_actions"):
                    if alt_key in data:
                        data["actions"] = data[alt_key]
                        break
                else:
                    if isinstance(parsed.get("actions"), list):
                        data["actions"] = parsed["actions"]
        else:
            raise ValueError("_BrowserActionPlan payload must be a JSON object or array.")

        raw_actions = data.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("LLM output did not include any browser actions.")

        action_map = {
            "goto": "goto",
            "navigate": "goto",
            "open": "goto",
            "visit": "goto",
            "go": "goto",
            "load": "goto",
            "click": "click",
            "press": "click",
            "tap": "click",
            "submit": "click",
            "hover": "click",
            "toggle": "click",
            "fill": "fill",
            "type": "fill",
            "input": "fill",
            "enter": "fill",
            "select": "fill",
            "wait_for_selector": "wait_for_selector",
            "wait": "wait_for_selector",
            "wait_for": "wait_for_selector",
            "sleep": "wait_for_selector",
            "assert_url": "assert_url",
            "check_url": "assert_url",
            "verify_url": "assert_url",
            "url": "assert_url",
            "assert_visible": "assert_visible",
            "verify": "assert_visible",
            "check": "assert_visible",
            "assert": "assert_visible",
            "expect_visible": "assert_visible",
            "inspect": "assert_visible",
        }

        norm_actions: list[dict[str, Any]] = []
        for idx, raw_act in enumerate(raw_actions, start=1):
            if not isinstance(raw_act, dict):
                continue
            raw_type = (
                str(
                    raw_act.get("action_type")
                    or raw_act.get("action")
                    or raw_act.get("type")
                    or raw_act.get("verb")
                    or "assert_visible"
                )
                .strip()
                .lower()
            )
            mapped_type = action_map.get(raw_type, "assert_visible")
            raw_sel = (
                raw_act.get("selector")
                or raw_act.get("target_hint")
                or raw_act.get("locator")
                or raw_act.get("target")
                or raw_act.get("element")
            )
            selector = str(raw_sel).strip() if raw_sel is not None and str(raw_sel).strip() else None
            raw_val = (
                raw_act.get("value")
                if raw_act.get("value") is not None
                else (raw_act.get("input_value") if raw_act.get("input_value") is not None else raw_act.get("input"))
            )
            val_str = str(raw_val) if raw_val is not None else None
            norm_actions.append(
                {
                    "step_number": _to_positive_int(
                        raw_act.get("step_number") or raw_act.get("step") or raw_act.get("order") or raw_act.get("index"),
                        default=idx,
                    ),
                    "action_type": mapped_type,
                    "selector": selector,
                    "fallback_selectors": _to_str_list(raw_act.get("fallback_selectors")),
                    "value": val_str,
                    "description": str(
                        raw_act.get("description") or raw_act.get("intent") or f"{mapped_type} {selector or val_str or ''}".strip()
                    ).strip(),
                    "timeout_ms": _to_positive_int(raw_act.get("timeout_ms"), default=5000, minimum=100),
                }
            )
        if not norm_actions:
            raise ValueError("LLM output did not contain any valid browser actions.")
        return {"actions": norm_actions}

    if name == "TriageDiagnosis" and isinstance(parsed, dict):
        data = _unwrap_payload_dict(parsed, ("diagnosis", "triage_diagnosis", "triage", "result", "data", "output"))
        raw_cat = str(data.get("category") or "ASSERTION_FAILURE").strip().upper().replace(" ", "_")
        allowed_cats = {"SELECTOR_DRIFT", "BACKEND_API_ERROR", "NETWORK_TIMEOUT", "ASSERTION_FAILURE", "CONSOLE_EXCEPTION"}
        if raw_cat not in allowed_cats:
            if "SELECTOR" in raw_cat or "LOCATOR" in raw_cat:
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
        primary = str(data.get("primary_healed_locator") or data.get("healed_selector") or "button").strip()
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
            "playwright": _coerce_fw("playwright", f"await page.locator({json.dumps(primary)}).click();"),
            "cypress": _coerce_fw("cypress", f"cy.get({json.dumps(primary)}).click();"),
            "selenium": _coerce_fw("selenium", f"driver.find_element(By.CSS_SELECTOR, {json.dumps(primary)}).click()"),
        }

    return parsed


class _OpenRouterChatModel(BaseChatModel):
    """Lightweight httpx-powered LangChain chat model for OpenRouter (supporting :free models)."""

    model_name: str
    api_key: str
    temperature: float = 0.2
    base_url: str = OPENROUTER_BASE_URL
    timeout_s: float = 25.0

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
            },
            timeout=self.timeout_s,
        )
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
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raw_messages = self._convert_messages(messages)
        text = self._call_api(raw_messages)
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
                "matching this exact JSON structure:\n"
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
    """
    Create a model-agnostic LangChain BaseChatModel from explicit arguments or environment variables.

    Supports:
    - openrouter (OPENROUTER_API_KEY via built-in _OpenRouterChatModel supporting ':free' models)
    - google_genai (GOOGLE_API_KEY or GEMINI_API_KEY)
    - openai (OPENAI_API_KEY)
    - deterministic (returns None immediately for offline rule-based execution)
    Returns None if no valid API key is configured so agents can fall back cleanly.
    """
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
                parsed_model
                or os.getenv("LLM_MODEL")
                or os.getenv("OPENROUTER_MODEL")
                or "openrouter/free"
            ).strip()
            if resolved_model in _UNAVAILABLE_MODELS and _WORKING_OPENROUTER_MODEL:
                resolved_model = _WORKING_OPENROUTER_MODEL
            return _OpenRouterChatModel(
                model_name=resolved_model,
                api_key=resolved_key,
                temperature=temperature,
            )

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
                timeout=12,
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
                timeout=12,
            )

        # Generic provider pass-through supported by LangChain init_chat_model
        resolved_model = (parsed_model or os.getenv("LLM_MODEL") or "").strip()
        if not resolved_model:
            return None
        kwargs: dict[str, Any] = {"model_provider": raw_provider, "temperature": temperature}
        if explicit_key:
            kwargs["api_key"] = explicit_key
        return init_chat_model(resolved_model, **kwargs)
    except ImportError as exc:
        logger.warning("[AgentSwarm LLM] Missing LangChain provider package for '%s': %s", raw_provider, exc)
        return None
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
    """
    Invoke a LangChain structured-output chain and recover rapidly from 404/503/429/timeout/provider errors.

    Returns `(validated_result, resolved_model_name)`.
    """
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
        alt_llm = resolve_chat_model(
            model_name=_WORKING_OPENROUTER_MODEL,
            provider="openrouter",
            api_key=api_key,
            temperature=temperature,
        )
        if alt_llm:
            primary_chain = prompt | alt_llm.with_structured_output(schema)
            failed_model = _WORKING_OPENROUTER_MODEL
    elif eff_provider == "google_genai" and failed_model in _UNAVAILABLE_MODELS and _WORKING_GOOGLE_MODEL and _WORKING_GOOGLE_MODEL not in _UNAVAILABLE_MODELS:
        alt_llm = resolve_chat_model(
            model_name=_WORKING_GOOGLE_MODEL,
            provider="google_genai",
            api_key=api_key,
            temperature=temperature,
        )
        if alt_llm:
            primary_chain = prompt | alt_llm.with_structured_output(schema)
            failed_model = _WORKING_GOOGLE_MODEL

    try:
        res = primary_chain.invoke(prompt_input)
        return res, failed_model
    except Exception as exc:
        err_str = str(exc)
        err_low = err_str.lower()
        is_recoverable = isinstance(exc, (ValueError, ValidationError)) or any(
            tok in err_low
            for tok in (
                "404",
                "not_found",
                "no longer available",
                "is not found",
                "no endpoints found",
                "provider returned error",
                "500",
                "502",
                "503",
                "504",
                "bad gateway",
                "unavailable",
                "429",
                "resource_exhausted",
                "rate limit",
                "overloaded",
                "high demand",
                "timed out",
                "timeout",
                "readtimeout",
                "connect",
                "remoteprotocolerror",
                "could not extract valid json",
                "validation error",
                "field required",
                "invalid json",
                "empty choices",
            )
        )
        if not allow_fallback or not is_recoverable:
            raise

        is_not_found = any(
            tok in err_low
            for tok in ("404", "not_found", "no longer available", "is not found", "no endpoints found")
        )
        if failed_model != "openrouter/free" or is_not_found:
            _UNAVAILABLE_MODELS.add(failed_model)

        if eff_provider == "openrouter":
            pool = [
                "openrouter/free",
                "google/gemma-3-27b-it:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3-0324:free",
            ]
        elif eff_provider == "google_genai":
            suggested = re.findall(r"models/([a-zA-Z0-9._-]+)", err_str)
            pool = [
                *suggested,
                "gemini-2.0-flash",
                "gemini-2.5-pro",
                "gemini-3.8-pro",
                "gemini-3.8-flash",
                *_discover_google_models(api_key),
            ]
        else:
            raise

        candidates: list[str] = []
        for cand in pool:
            clean = cand.strip()
            if clean and clean != failed_model and clean not in _UNAVAILABLE_MODELS and clean not in candidates:
                candidates.append(clean)

        last_exc: Exception = exc
        for cand in candidates[:4]:
            alt_llm = resolve_chat_model(
                model_name=cand,
                provider=eff_provider,
                api_key=api_key,
                temperature=temperature,
            )
            if not alt_llm:
                continue
            try:
                alt_chain = prompt | alt_llm.with_structured_output(schema)
                res = alt_chain.invoke(prompt_input)
                if eff_provider == "google_genai":
                    _WORKING_GOOGLE_MODEL = cand
                elif eff_provider == "openrouter":
                    _WORKING_OPENROUTER_MODEL = cand
                logger.info("[AgentSwarm LLM] Recovered from '%s' error -> using '%s'", failed_model, cand)
                return res, cand
            except Exception as cand_exc:
                cand_err_low = str(cand_exc).lower()
                cand_not_found = any(
                    tok in cand_err_low
                    for tok in ("404", "not_found", "no longer available", "is not found", "no endpoints found")
                )
                if cand != "openrouter/free" or cand_not_found:
                    _UNAVAILABLE_MODELS.add(cand)
                last_exc = cand_exc
                continue
        raise last_exc


class TestStepPlan(BaseModel):
    """Single executable step inside an autonomous QA scenario."""

    step_number: int = Field(..., ge=1, description="1-based order of the step")
    action: str = Field(..., description="Browser action verb (navigate, click, fill, wait, assert_visible, assert_url)")
    description: str = Field(..., description="Human-readable intent of the step")
    target_hint: Optional[str] = Field(None, description="Suggested selector or accessible role/label hint")
    input_value: Optional[str] = Field(None, description="Input payload for fill/select actions")
    expected_outcome: str = Field("", description="Expected observable result after this step")


class TestScenario(BaseModel):
    """Structured end-to-end QA test scenario created by the PlannerAgent."""

    scenario_id: str = Field(..., description="Unique scenario identifier (e.g., SCN-001)")
    title: str = Field(..., description="Concise title of the test scenario")
    category: str = Field("e2e", description="Category such as smoke, auth, checkout, navigation, edge_case")
    priority: Literal["critical", "high", "medium", "low"] = Field("high", description="Execution priority")
    description: str = Field(..., description="Detailed scope of what this scenario validates")
    preconditions: list[str] = Field(default_factory=list, description="Required setup before running steps")
    steps: list[TestStepPlan] = Field(default_factory=list, description="Ordered steps in the scenario")
    expected_result: str = Field(..., description="Final acceptance condition for the scenario")


class TestPlanResult(BaseModel):
    """Complete multi-scenario test plan for a target web application."""

    target_url: str = Field(..., description="Normalized URL of the application under test")
    summary: str = Field(..., description="High-level summary of the generated test strategy")
    scenarios: list[TestScenario] = Field(default_factory=list, description="Prioritized test scenarios")
    generated_by: str = Field("langchain", description="Origin engine identifier")


class PlannerGraphState(TypedDict, total=False):
    """LangGraph state contract for PlannerAgent workflow."""

    target_url: str
    normalized_url: str
    dom_summary: Optional[str]
    objectives: list[str]
    prompt_input: dict[str, str]
    test_plan: TestPlanResult
    planner_error: Optional[str]


class PlannerAgent:
    """
    Model-agnostic autonomous QA test planning agent powered by LangChain & LangGraph.

    Uses `ChatPromptTemplate | llm.with_structured_output(TestPlanResult)` inside a
    compiled LangGraph `StateGraph` (`prepare_input -> generate_plan -> finalize_plan`).
    Works with Gemini, OpenAI, OpenRouter, or any injected LangChain BaseChatModel.
    """

    SYSTEM_PROMPT = (
        "You are the Lead QA Planner Agent for AgentSwarm. Inspect the live DOM / Route Summary carefully "
        "to determine what kind of web application this actually is (for example: portfolio, SaaS product, "
        "documentation site, blog, e-commerce store, or interactive web app), and autonomously design concise, "
        "non-destructive end-to-end browser test scenarios tailored specifically to its REAL features and DOM elements.\n\n"
        "Strict Planning Rules:\n"
        "1. NEVER invent login/authentication, checkout/cart, or form submission scenarios if no corresponding "
        "login, cart, or form controls appear in the DOM summary.\n"
        "2. Use exact selectors or locators shown in the DOM / Route Summary for every step's target_hint.\n"
        "3. If an entry gate, splash screen, or intro overlay button (such as skip/enter intro) appears in the DOM, "
        "include dismissing/clicking it BEFORE interacting with underlying navigation links or sections.\n"
        "4. Give each scenario a specific, meaningful title reflecting the actual website functionality under test."
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
        self._last_llm_error: Optional[str] = None
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
                    "Testing Objectives:\n{objectives_text}\n\n"
                    "DOM / Route Summary:\n{dom_summary}\n\n"
                    "First identify what this specific application does based on the headings, landmarks, links, "
                    "buttons, overlays, and inputs in the DOM summary above. Then design 2 to 4 distinct, realistic, "
                    "non-redundant end-to-end test scenarios tailored specifically to the real interactive elements and "
                    "sections present on this page (plus any explicit Testing Objectives above). Use exact selectors "
                    "from the DOM summary as target_hint.",
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
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        return url

    def _invoke_chain(self, prompt_input: dict[str, str]) -> TestPlanResult:
        if not self._chain:
            raise RuntimeError("LangChain structured output chain is not configured.")
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
        if not result.scenarios:
            raise ValueError("LLM returned zero test scenarios.")

        result.generated_by = f"langchain:{resolved_model}" if resolved_model else "langchain"
        return result

    def _node_prepare_input(self, state: PlannerGraphState) -> dict[str, Any]:
        normalized_url = self._normalize_url(state.get("target_url", ""))
        objectives = state.get("objectives") or []
        objectives_text = "\n".join(f"- {obj}" for obj in objectives) if objectives else "- Autonomously test the primary user journeys visible in the live DOM"
        dom_summary = (state.get("dom_summary") or "No DOM summary provided")[:6000]
        return {
            "normalized_url": normalized_url,
            "prompt_input": {
                "target_url": normalized_url,
                "objectives_text": objectives_text,
                "dom_summary": dom_summary,
            },
        }

    def _node_generate_plan(self, state: PlannerGraphState) -> dict[str, Any]:
        normalized_url = state["normalized_url"]
        planner_error: Optional[str] = None
        self._last_llm_error = None
        if self._chain:
            try:
                plan = self._invoke_chain(state["prompt_input"])
                return {"test_plan": plan, "planner_error": None}
            except Exception as exc:
                planner_error = str(exc).strip()[:220]
                self._last_llm_error = planner_error
                logger.warning("[PlannerAgent] LangChain invocation failed, using deterministic fallback: %s", exc)
        else:
            planner_error = "Deterministic mode selected (no LLM configured)."
            self._last_llm_error = planner_error

        plan = self._build_fallback_plan(
            target_url=normalized_url,
            dom_summary=state.get("dom_summary"),
            objectives=state.get("objectives"),
        )
        return {"test_plan": plan, "planner_error": planner_error}

    @staticmethod
    def _node_finalize_plan(state: PlannerGraphState) -> dict[str, Any]:
        plan = state["test_plan"]
        plan.target_url = state["normalized_url"]
        return {"test_plan": plan}

    def build_graph(self):
        """Build and compile the internal LangGraph workflow for test planning."""
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
        """Execute the compiled LangGraph planner workflow to produce a structured TestPlanResult."""
        result_state = self._graph.invoke(
            {
                "target_url": target_url,
                "dom_summary": dom_summary,
                "objectives": objectives or [],
            }
        )
        return result_state["test_plan"]

    @staticmethod
    def _extract_dom_candidates(dom_summary: Optional[str]) -> list[dict[str, str]]:
        """Extract interactive element hints from BrowserToolkit summary lines or raw HTML."""
        if not dom_summary:
            return []
        candidates: list[dict[str, str]] = []
        pattern = re.compile(
            r"^-\s*<([a-z0-9]+)>\s+role=([a-z0-9_-]+)(?:\s+name='([^']*)')?(?:\s+id=#([^\s]+))?\s*->\s*(.+)$",
            re.I,
        )
        for line in dom_summary.splitlines():
            m = pattern.match(line.strip())
            if m:
                candidates.append(
                    {
                        "tag": m.group(1).lower(),
                        "role": m.group(2).lower(),
                        "name": (m.group(3) or "").strip(),
                        "id": (m.group(4) or "").strip(),
                        "locator": m.group(5).strip(),
                    }
                )
        if not candidates and "</" in dom_summary:
            try:
                from ..browser.browser_tools import BrowserToolkit

                for el in BrowserToolkit.extract_interactive_elements(dom_summary):
                    candidates.append(
                        {
                            "tag": el.tag,
                            "role": el.role,
                            "name": el.accessible_name or el.text or el.placeholder or el.element_id,
                            "locator": el.recommended_locator or (f"#{el.element_id}" if el.element_id else el.tag),
                            "id": el.element_id,
                        }
                    )
            except Exception:
                pass
        return candidates

    @staticmethod
    def _extract_headings(dom_summary: Optional[str]) -> list[str]:
        if not dom_summary:
            return []
        for line in dom_summary.splitlines():
            if line.startswith("Landmarks & Headings:"):
                raw = line.split(":", 1)[1]
                headings = []
                for part in raw.split("|"):
                    txt = part.split(":", 1)[1].strip() if ":" in part else part.strip()
                    if txt and txt not in headings:
                        headings.append(txt)
                return headings
        return []

    @classmethod
    def _build_fallback_plan(
        cls,
        target_url: str,
        dom_summary: Optional[str],
        objectives: Optional[list[str]],
    ) -> TestPlanResult:
        candidates = cls._extract_dom_candidates(dom_summary)
        headings = cls._extract_headings(dom_summary)
        site_label = headings[0][:50] if headings else target_url

        buttons = [c for c in candidates if c["role"] == "button" or c["tag"] == "button"]
        links = [c for c in candidates if c["role"] == "link" or c["tag"] == "a"]
        inputs = [c for c in candidates if c["role"] in ("textbox", "combobox") or c["tag"] in ("input", "textarea", "select")]
        intro_btns = [
            b for b in buttons
            if any(tok in (b["name"] + " " + b["id"]).lower() for tok in ("skip", "enter", "intro"))
        ]

        scenarios: list[TestScenario] = []

        # Purely dynamic DOM-driven exploration from real discovered elements (no fixed templates)
        steps: list[TestStepPlan] = [
            TestStepPlan(
                step_number=1,
                action="navigate",
                description=f"Open {site_label} ({target_url})",
                target_hint=target_url,
                expected_outcome="HTTP 2xx response and page document loads",
            )
        ]
        if intro_btns:
            ib = intro_btns[0]
            ib_label = ib["name"] or ib["id"] or "Skip / Enter"
            steps.append(
                TestStepPlan(
                    step_number=len(steps) + 1,
                    action="click",
                    description=f"Dismiss intro overlay via '{ib_label[:45]}'",
                    target_hint=ib["locator"],
                    expected_outcome="Underlying page content becomes accessible",
                )
            )

        steps.append(
            TestStepPlan(
                step_number=len(steps) + 1,
                action="assert_visible",
                description=f"Verify primary landmarks render visibly on {site_label}",
                target_hint="main, body, h1",
                expected_outcome="Primary content renders without layout failure",
            )
        )

        # Exercise up to 2 real discovered interactive links/buttons
        interactive_pool: list[dict[str, str]] = []
        for lk in links:
            if lk not in interactive_pool and "skip" not in (lk["name"] + " " + lk["id"]).lower():
                interactive_pool.append(lk)
        for btn in buttons:
            if btn not in intro_btns and btn not in interactive_pool:
                interactive_pool.append(btn)

        for item in interactive_pool[:2]:
            label = item["name"] or item["id"] or item["tag"]
            steps.append(
                TestStepPlan(
                    step_number=len(steps) + 1,
                    action="click",
                    description=f"Click discovered {item['role'] or item['tag']} '{label[:45]}'",
                    target_hint=item["locator"],
                    expected_outcome="Control interacts cleanly without runtime or console errors",
                )
            )

        # Only populate real discovered form inputs if inputs actually exist in the DOM
        for inp in inputs[:2]:
            inp_label = inp["name"] or inp["id"] or "input"
            sample_val = (
                "qa.synthetic@agentswarm.dev"
                if "email" in inp_label.lower()
                else ("SyntheticPass!2026" if "pass" in inp_label.lower() else "AgentSwarm QA Test")
            )
            steps.append(
                TestStepPlan(
                    step_number=len(steps) + 1,
                    action="fill",
                    description=f"Fill discovered input '{inp_label[:40]}' with synthetic QA data",
                    target_hint=inp["locator"],
                    input_value=sample_val,
                    expected_outcome=f"Input '{inp_label[:40]}' accepts text cleanly",
                )
            )

        scenarios.append(
            TestScenario(
                scenario_id="SCN-001",
                title=f"{site_label}: DOM Exploration",
                category="smoke",
                priority="high",
                description=f"Deterministic DOM-driven fallback exploration of real discovered controls on {site_label}.",
                preconditions=["Application URL is reachable"],
                steps=steps,
                expected_result="Discovered page landmarks and interactive controls function cleanly.",
            )
        )

        if objectives:
            for objective in objectives[:3]:
                next_idx = len(scenarios) + 1
                obj_hint = buttons[0]["locator"] if buttons else (links[0]["locator"] if links else "main, body")
                scenarios.append(
                    TestScenario(
                        scenario_id=f"SCN-{next_idx:03d}",
                        title=f"Objective Check: {objective[:72]}",
                        category="objective",
                        priority="high",
                        description=f"Verify user-requested QA objective: {objective}",
                        preconditions=["Application loaded"],
                        steps=[
                            TestStepPlan(
                                step_number=1,
                                action="navigate",
                                description=f"Ensure application context at {target_url}",
                                target_hint=target_url,
                                expected_outcome="Application is active",
                            ),
                            TestStepPlan(
                                step_number=2,
                                action="assert_visible",
                                description=f"Validate elements related to objective: {objective}",
                                target_hint=obj_hint,
                                expected_outcome="Objective criteria satisfied without errors",
                            ),
                        ],
                        expected_result=f"Objective verified: {objective}",
                    )
                )

        return TestPlanResult(
            target_url=target_url,
            summary=f"Deterministic DOM-driven fallback generated {len(scenarios)} scenario(s) from discovered controls on {target_url}.",
            scenarios=scenarios,
            generated_by="deterministic_fallback",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph parent-graph node entrypoint for test planning."""
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
    """Functional LangGraph node wrapper around PlannerAgent."""
    return PlannerAgent().run_node(state)
