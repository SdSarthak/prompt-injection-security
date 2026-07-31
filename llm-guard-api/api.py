"""FastAPI service exposing the guard as an HTTP middleware.

Run with:
    uvicorn api:app --reload
or:
    python api.py
"""

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

import config
from app import LLMGuard
from guard import Decision

logger = logging.getLogger(__name__)

# Populated on startup so model loading happens once, not per request.
_state: Dict[str, Any] = {"guard": None}

MAX_BATCH_SIZE = 64


def get_guard() -> LLMGuard:
    """Return the process-wide guard instance."""
    guard = _state.get("guard")
    if guard is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guard is not initialised",
        )
    return guard


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Enforce the X-API-Key header when API_KEYS is configured.

    With no keys configured the service is open, which is the right default for
    local development but must not be used on a public interface.
    """
    if not config.API_KEYS:
        return
    # Compared digest-wise so the response time does not leak how many leading
    # characters of a guessed key were correct.
    supplied = x_api_key or ""
    if not any(hmac.compare_digest(supplied, known) for known in config.API_KEYS):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the classifier once at startup."""
    logger.info("Loading guard pipeline...")
    _state["guard"] = LLMGuard()
    logger.info("Guard ready")
    yield
    _state["guard"] = None


app = FastAPI(
    title="LLM Prompt-Injection Guard",
    description=(
        "Layered defense in front of an LLM: regex heuristics, an intent "
        "classifier, a rule-based decision engine and a prompt sanitizer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------- schemas


MAX_PROMPT_CHARS = 100_000

# The guard truncates to config.MAX_PROMPT_LENGTH anyway; this cap exists so a
# multi-megabyte body is rejected before it is parsed and copied.
PromptText = Annotated[str, Field(max_length=MAX_PROMPT_CHARS)]


class AnalyzeRequest(BaseModel):
    """A single prompt to inspect."""

    prompt: PromptText = Field(..., description="Raw user input to inspect")


class BatchAnalyzeRequest(BaseModel):
    """Several prompts to inspect in one round trip."""

    # ``max_length`` on the list bounds the item count; the item type bounds
    # each string. Without the latter a 64-item batch is unbounded in bytes.
    prompts: List[PromptText] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class GuardRequest(AnalyzeRequest):
    """A prompt to inspect and, unless blocked, forward to Gemini."""

    call_llm: bool = Field(True, description="Forward the safe prompt to Gemini")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0, le=8192)


class RegexAnalysis(BaseModel):
    flag: bool
    matched_patterns: List[str]
    risk_score: float


class IntentAnalysis(BaseModel):
    intent: str
    confidence: float
    class_scores: Dict[str, float]
    backend: str


class DecisionReasoning(BaseModel):
    reasoning: str
    confidence: float
    rule_matched: str
    combined_score: float


class Sanitization(BaseModel):
    original_length: int
    sanitized_length: int
    changes: str
    sanitized_prompt: str


class AnalyzeResponse(BaseModel):
    timestamp: str
    decision: str
    action: Optional[str] = None
    safe_prompt: Optional[str] = None
    input_truncated: bool = False
    latency_ms: float = 0.0
    regex_analysis: RegexAnalysis
    intent_analysis: IntentAnalysis
    decision_reasoning: DecisionReasoning
    sanitization: Optional[Sanitization] = None


class GuardResponse(AnalyzeResponse):
    response: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    classifier_backend: str
    sanitization_level: str
    llm_available: bool
    llm_model: str
    auth_required: bool


# ------------------------------------------------------------------ conversion


def _to_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the guard's nested metadata into the API response shape."""
    meta = result["metadata"]
    payload = {
        "timestamp": result["timestamp"],
        "decision": result["decision"],
        "action": meta.get("action"),
        "safe_prompt": result.get("safe_prompt"),
        "input_truncated": result.get("input_truncated", False),
        "latency_ms": meta.get("latency_ms", 0.0),
        "regex_analysis": meta["regex_analysis"],
        "intent_analysis": meta["intent_analysis"],
        "decision_reasoning": meta["decision_reasoning"],
        "sanitization": meta.get("sanitization"),
    }
    if "response" in result:
        payload["response"] = result.get("response")
    if result.get("error"):
        payload["error"] = result["error"]
    return payload


# --------------------------------------------------------------------- routes


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe plus the configuration the process actually loaded."""
    guard = get_guard()
    return HealthResponse(
        status="ok",
        classifier_backend=guard.classifier_backend,
        sanitization_level=guard.sanitizer.level.value,
        llm_available=guard.llm_client is not None,
        llm_model=config.GEMINI_MODEL,
        auth_required=bool(config.API_KEYS),
    )


@app.post(
    "/v1/analyze",
    response_model=AnalyzeResponse,
    tags=["guard"],
    dependencies=[Depends(require_api_key)],
)
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """Classify a prompt without calling the LLM.

    Use this when your application owns the LLM call and only wants a verdict.

    Declared ``def`` rather than ``async def`` deliberately: the guard is
    CPU-bound (regex plus a scikit-learn forward pass), and running it directly
    on the event loop would stall every other in-flight request, including
    ``/health``, for the duration. FastAPI runs sync handlers in a threadpool.
    """
    return _to_response(get_guard().analyze(request.prompt))


@app.post(
    "/v1/analyze/batch",
    response_model=List[AnalyzeResponse],
    tags=["guard"],
    dependencies=[Depends(require_api_key)],
)
def analyze_batch(request: BatchAnalyzeRequest) -> List[Dict[str, Any]]:
    """Classify up to 64 prompts in a single request.

    The whole batch goes through the classifier in one vectorised pass, which
    is most of the reason to use this endpoint over 64 calls to ``/v1/analyze``.
    """
    guard = get_guard()
    return [_to_response(result) for result in guard.analyze_batch(request.prompts)]


@app.post(
    "/v1/guard",
    response_model=GuardResponse,
    tags=["guard"],
    dependencies=[Depends(require_api_key)],
)
def guard_prompt(request: GuardRequest) -> Dict[str, Any]:
    """Inspect a prompt and, unless it is blocked, answer it with Gemini."""
    guard = get_guard()

    llm_kwargs: Dict[str, Any] = {}
    if request.temperature is not None:
        llm_kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        llm_kwargs["max_tokens"] = request.max_tokens

    result = guard.guard(request.prompt, call_llm=request.call_llm, **llm_kwargs)

    if result["decision"] != Decision.BLOCK.value and result.get("error"):
        # The verdict is valid but the upstream call failed - say so explicitly
        # rather than returning a 200 with a null answer and no explanation.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result["error"],
        )

    return _to_response(result)


def main() -> None:
    """Run the service with uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not config.API_KEYS and config.API_HOST not in ("127.0.0.1", "localhost"):
        logger.warning(
            "Binding %s with no API_KEYS set - the guard endpoint is unauthenticated",
            config.API_HOST,
        )
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


if __name__ == "__main__":
    main()
