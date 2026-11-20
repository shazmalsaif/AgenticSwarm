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
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

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


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from raw LLM output (handling <think> tags and markdown fences)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.S | re.I).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S | re.I):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                decoder = json.JSONDecoder()
                parsed, _ = decoder.raw_decode(cleaned[first_brace:])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    raise ValueError(f"Could not extract valid JSON object from response: {cleaned[:240]}")


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
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

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
                "conforming strictly to this JSON Schema:\n"
                f"{schema_json}"
            )
            if api_msgs and api_msgs[-1]["role"] == "user":
                api_msgs[-1] = {**api_msgs[-1], "content": api_msgs[-1]["content"] + instruction}
            else:
                api_msgs.append({"role": "user", "content": instruction})

            raw_text = self._call_api(api_msgs)
            parsed = _extract_json_payload(raw_text)
            return schema.model_validate(parsed)

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
        is_recoverable = any(
            tok in err_low
            for tok in (
                "404",
                "not_found",
                "no longer available",
                "is not found",
                "no endpoints found",
                "provider returned error",
                "503",
                "unavailable",
                "429",
                "resource_exhausted",
                "rate limit",
                "overloaded",
                "high demand",
                "timed out",
                "timeout",
                "could not extract valid json",
            )
        )
        if not allow_fallback or not is_recoverable:
            raise

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
        for cand in candidates[:2]:
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


class PlannerAgent:
    """
    Model-agnostic autonomous QA test planning agent powered by LangChain & LangGraph.

    Uses `ChatPromptTemplate | llm.with_structured_output(TestPlanResult)` inside a
    compiled LangGraph `StateGraph` (`prepare_input -> generate_plan -> finalize_plan`).
    Works with Gemini, OpenAI, OpenRouter, or any injected LangChain BaseChatModel.
    """

    SYSTEM_PROMPT = (
        "You are the Lead QA Planner Agent for AgentSwarm. Design concise, realistic, "
        "non-destructive end-to-end browser test plans using semantic accessibility hints. "
        "Only target authorized/staging workflows and never invent destructive actions."
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
                    "Testing Objectives:\n{objectives_text}\n\n"
                    "DOM / Route Summary:\n{dom_summary}\n\n"
                    "Generate 3 to 5 prioritized end-to-end test scenarios covering smoke navigation, "
                    "primary user workflow, authentication/session state, and negative edge cases.",
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _invoke_chain(self, prompt_input: dict[str, str]) -> TestPlanResult:
        if not self._chain:
            raise RuntimeError("LangChain structured output chain is not configured.")
        result = self._chain.invoke(prompt_input)
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
        
        result.generated_by = f"langchain:{resolved_model}" if resolved_model else "langchain"
        return result

    def _node_prepare_input(self, state: PlannerGraphState) -> dict[str, Any]:
        normalized_url = self._normalize_url(state.get("target_url", ""))
        objectives = state.get("objectives") or []
        objectives_text = "\n".join(f"- {obj}" for obj in objectives) if objectives else "- Core smoke and user workflows"
        dom_summary = (state.get("dom_summary") or "No DOM summary provided")[:4000]
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
        if self._chain:
            try:
                plan = self._invoke_chain(state["prompt_input"])
                return {"test_plan": plan}
            except Exception as exc:
                logger.warning("[PlannerAgent] LangChain invocation failed, using deterministic fallback: %s", exc)

        plan = self._build_fallback_plan(
            target_url=normalized_url,
            dom_summary=state.get("dom_summary"),
            objectives=state.get("objectives"),
        )
        return {"test_plan": plan}

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
        dom_lower = (dom_summary or "").lower()
        candidates = cls._extract_dom_candidates(dom_summary)
        headings = cls._extract_headings(dom_summary)
        site_label = headings[0][:40] if headings else target_url

        buttons = [c for c in candidates if c["role"] == "button" or c["tag"] == "button"]
        links = [c for c in candidates if c["role"] == "link" or c["tag"] == "a"]
        inputs = [c for c in candidates if c["role"] in ("textbox", "combobox") or c["tag"] in ("input", "textarea", "select")]
        intro_btns = [
            b for b in buttons
            if any(tok in (b["name"] + " " + b["id"]).lower() for tok in ("skip", "enter", "intro"))
        ]

        scenarios: list[TestScenario] = []

        # Scenario 1: Landing & Primary Element Smoke (using real DOM targets when discovered)
        scn1_steps: list[TestStepPlan] = [
            TestStepPlan(
                step_number=1,
                action="navigate",
                description=f"Open {site_label} ({target_url})",
                target_hint=target_url,
                expected_outcome="HTTP 2xx response and main document renders",
            )
        ]
        if intro_btns:
            ib = intro_btns[0]
            ib_label = ib["name"] or ib["id"] or "Entry / Skip Intro"
            scn1_steps.append(
                TestStepPlan(
                    step_number=len(scn1_steps) + 1,
                    action="click",
                    description=f"Dismiss intro / entry overlay via '{ib_label[:40]}'",
                    target_hint=ib["locator"],
                    expected_outcome="Main page content becomes visible",
                )
            )
        scn1_steps.append(
            TestStepPlan(
                step_number=len(scn1_steps) + 1,
                action="assert_visible",
                description=f"Verify primary content landmarks on {site_label} are visible",
                target_hint="main, body, h1",
                expected_outcome="Primary application shell and heading render visibly",
            )
        )
        if links:
            first_link = next(
                (lk for lk in links if "skip" not in (lk["name"] + lk["id"]).lower()),
                links[0],
            )
            label = first_link["name"] or "primary link"
            scn1_steps.append(
                TestStepPlan(
                    step_number=len(scn1_steps) + 1,
                    action="click",
                    description=f"Click discovered link '{label[:45]}'",
                    target_hint=first_link["locator"],
                    expected_outcome="Navigation or interaction executes without console or network errors",
                )
            )
        elif buttons:
            non_intro = [b for b in buttons if b not in intro_btns]
            first_btn = non_intro[0] if non_intro else buttons[0]
            label = first_btn["name"] or "primary button"
            scn1_steps.append(
                TestStepPlan(
                    step_number=len(scn1_steps) + 1,
                    action="click",
                    description=f"Click discovered button '{label[:45]}'",
                    target_hint=first_btn["locator"],
                    expected_outcome="Button responds cleanly without runtime exception",
                )
            )

        scenarios.append(
            TestScenario(
                scenario_id="SCN-001",
                title=f"{site_label}: Landing & Discovered Navigation Smoke",
                category="smoke",
                priority="critical",
                description=f"Verify initial render and real DOM interactive controls on {site_label}.",
                preconditions=["Application URL is reachable", "Clean browser context"],
                steps=scn1_steps,
                expected_result="Application loads cleanly with accessible interactive controls.",
            )
        )

        # Scenario 2: Discovered Forms / Inputs OR Authentication Flow
        if inputs:
            form_steps: list[TestStepPlan] = [
                TestStepPlan(
                    step_number=1,
                    action="navigate",
                    description=f"Ensure target page {target_url} is loaded",
                    target_hint=target_url,
                    expected_outcome="Page form inputs are ready",
                )
            ]
            step_num = 2
            for inp in inputs[:3]:
                inp_label = inp["name"] or inp["id"] or "input field"
                sample_val = (
                    "qa.synthetic@agentswarm.dev"
                    if "email" in inp_label.lower()
                    else ("SyntheticPass!2026" if "pass" in inp_label.lower() else "AgentSwarm QA Test")
                )
                form_steps.append(
                    TestStepPlan(
                        step_number=step_num,
                        action="fill",
                        description=f"Populate '{inp_label[:40]}' input with synthetic QA value",
                        target_hint=inp["locator"],
                        input_value=sample_val,
                        expected_outcome=f"Input '{inp_label[:40]}' accepts synthetic text cleanly",
                    )
                )
                step_num += 1
            if buttons:
                submit_btn = buttons[0]
                btn_label = submit_btn["name"] or "Submit"
                form_steps.append(
                    TestStepPlan(
                        step_number=step_num,
                        action="click",
                        description=f"Click '{btn_label[:40]}' button to submit form",
                        target_hint=submit_btn["locator"],
                        expected_outcome="Submission triggers expected state change without 5xx error",
                    )
                )
            scenarios.append(
                TestScenario(
                    scenario_id="SCN-002",
                    title="Discovered Form Inputs & Submission Journey",
                    category="e2e",
                    priority="high",
                    description="Populate real discovered form inputs with synthetic QA data and verify interaction.",
                    preconditions=["Form elements present in live DOM"],
                    steps=form_steps,
                    expected_result="All discovered form controls accept synthetic inputs and submit cleanly.",
                )
            )
        elif len(buttons) >= 2 or "sign in" in dom_lower or "login" in dom_lower:
            btn_target = buttons[0]["locator"] if buttons else "button"
            btn_name = (buttons[0]["name"] or "Action") if buttons else "Action"
            scenarios.append(
                TestScenario(
                    scenario_id="SCN-002",
                    title="Interactive Controls & CTA Verification",
                    category="auth" if ("sign in" in dom_lower or "login" in dom_lower) else "e2e",
                    priority="high",
                    description="Exercise discovered interactive buttons and verify visual state transitions.",
                    preconditions=["Target page rendered"],
                    steps=[
                        TestStepPlan(
                            step_number=1,
                            action="navigate",
                            description=f"Open {target_url}",
                            target_hint=target_url,
                            expected_outcome="Page rendered",
                        ),
                        TestStepPlan(
                            step_number=2,
                            action="assert_visible",
                            description=f"Verify button '{btn_name[:40]}' is visible",
                            target_hint=btn_target,
                            expected_outcome="Interactive button is visible",
                        ),
                        TestStepPlan(
                            step_number=3,
                            action="click",
                            description=f"Click '{btn_name[:40]}' control",
                            target_hint=btn_target,
                            expected_outcome="UI transitions cleanly without console error",
                        ),
                    ],
                    expected_result="Discovered buttons respond cleanly.",
                )
            )

        # Scenario 3: Commerce / Core Interactive Workflow Verification
        is_commerce = any(token in dom_lower for token in ("checkout", "cart", "product", "price", "order"))
        if is_commerce or not scenarios or len(scenarios) < 2:
            cta_hint = (
                buttons[-1]["locator"]
                if buttons
                else "button[type='submit'], .cta-primary, getByRole('button', { name: /proceed|deploy|start/i })"
            )
            cta_label = (buttons[-1]["name"] or "primary CTA") if buttons else "primary CTA"
            scenarios.append(
                TestScenario(
                    scenario_id=f"SCN-{len(scenarios) + 1:03d}",
                    title="Multi-Stage Checkout Journey" if is_commerce else "Core Interactive Workflow Verification",
                    category="checkout" if is_commerce else "e2e",
                    priority="high",
                    description=(
                        "Execute end-to-end cart addition and checkout CTA progression while monitoring selector resilience."
                        if is_commerce
                        else "Validate primary user workflow controls, state transitions, and submission feedback."
                    ),
                    preconditions=["Main view loaded"],
                    steps=[
                        TestStepPlan(
                            step_number=1,
                            action="navigate",
                            description=f"Ensure application context at {target_url}",
                            target_hint=target_url,
                            expected_outcome="Application active",
                        ),
                        TestStepPlan(
                            step_number=2,
                            action="click",
                            description=f"Trigger '{cta_label[:40]}' action button",
                            target_hint=cta_hint,
                            expected_outcome="Workflow transitions to active processing or next step",
                        ),
                        TestStepPlan(
                            step_number=3,
                            action="assert_visible",
                            description="Confirm workflow status output updates visibly",
                            target_hint="main, body, [role='status']",
                            expected_outcome="Updated application state renders without selector timeout",
                        ),
                    ],
                    expected_result="Core interactive workflow advances and updates DOM state reliably.",
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
            summary=f"Generated {len(scenarios)} DOM-aware QA test scenarios for {target_url}.",
            scenarios=scenarios,
            generated_by="deterministic_fallback",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph parent-graph node entrypoint for test planning."""
        target_url = state.get("target_url", "https://example.com")
        dom_summary = state.get("dom_summary") or state.get("dom_snapshot")
        objectives = state.get("objectives")
        plan = self.create_test_plan(target_url=target_url, dom_summary=dom_summary, objectives=objectives)
        return {
            "target_url": plan.target_url,
            "test_plan": plan.model_dump(),
            "current_agent": "planner",
        }


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Functional LangGraph node wrapper around PlannerAgent."""
    return PlannerAgent().run_node(state)
