"""Main application: LLM Guard orchestrator combining all defense layers."""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

import config
from guard import (
    Decision,
    DecisionEngine,
    PromptSanitizer,
    RegexFilter,
    SanitizationLevel,
    build_classifier,
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLMGuard:
    """Complete prompt injection guard pipeline.

    The pipeline splits cleanly in two:

    * :meth:`analyze` runs the four local defense layers and returns a verdict.
      It never touches the network, so it is cheap enough for evaluation loops,
      unit tests and dry-run traffic replay.
    * :meth:`guard` calls :meth:`analyze` and then forwards allowed or sanitized
      prompts to Gemini.
    """

    def __init__(
        self,
        classifier_model_path: Optional[str] = None,
        sanitization_level: Optional[SanitizationLevel] = None,
        classifier_backend: Optional[str] = None,
        enable_llm: bool = True,
    ):
        """
        Initialize the guard with all defense layers.

        Args:
            classifier_model_path: Override the classifier artifact path.
            sanitization_level: How aggressively to sanitize prompts. Defaults
                to ``config.SANITIZATION_LEVEL``.
            classifier_backend: "baseline" or "transformer". Defaults to
                ``config.CLASSIFIER_BACKEND``.
            enable_llm: Set False to build an analysis-only guard that never
                constructs a Gemini client.
        """
        logger.info("Initializing LLM Guard...")

        # Layer 1: Fast regex filter
        self.regex_filter = RegexFilter()
        logger.info("Regex filter initialized")

        # Layer 2: intent classifier (baseline by default; falls back to it when
        # the transformer backend has no fine-tuned checkpoint)
        self.classifier = build_classifier(
            backend=classifier_backend, model_path=classifier_model_path
        )
        self.classifier_backend = type(self.classifier).__name__
        logger.info("Intent classifier initialized (%s)", self.classifier_backend)

        # Layer 3: Decision engine
        self.decision_engine = DecisionEngine()
        logger.info("Decision engine initialized")

        # Layer 4: Sanitizer
        if sanitization_level is None:
            self.sanitizer = PromptSanitizer.from_name(config.SANITIZATION_LEVEL)
        else:
            self.sanitizer = PromptSanitizer(level=sanitization_level)
        logger.info("Sanitizer initialized (level: %s)", self.sanitizer.level.value)

        # Layer 5: Gemini API client (optional - the guard is useful without it)
        self.llm_client = None
        self.llm_error: Optional[str] = None
        if enable_llm:
            try:
                from llm import GeminiClient

                self.llm_client = GeminiClient()
                logger.info("Gemini API client initialized (%s)", config.GEMINI_MODEL)
            except Exception as exc:
                self.llm_error = str(exc)
                logger.warning(
                    "Gemini client unavailable (%s). The guard will still classify "
                    "prompts but cannot generate responses.",
                    exc,
                )

    # ------------------------------------------------------------------ layers

    def analyze(self, user_prompt: str) -> Dict[str, Any]:
        """
        Run the local defense layers and return a verdict. No network calls.

        Args:
            user_prompt: Raw user input

        Returns:
            Dict with the decision, the prompt that should be forwarded to the
            LLM (``safe_prompt``, None when blocked) and full layer metadata.
        """
        started = time.perf_counter()
        user_prompt = "" if user_prompt is None else str(user_prompt)
        timestamp = datetime.now(timezone.utc).isoformat()

        truncated = False
        if config.MAX_PROMPT_LENGTH > 0 and len(user_prompt) > config.MAX_PROMPT_LENGTH:
            # Cap before any regex work: unbounded input is itself a DoS vector.
            user_prompt = user_prompt[: config.MAX_PROMPT_LENGTH]
            truncated = True

        # Step 1: Regex filter (fast first gate)
        regex_result = self.regex_filter.check(user_prompt)
        logger.debug("Regex flag=%s score=%s", regex_result.flag, regex_result.score)

        # Step 2: Intent classification
        intent_result = self.classifier.classify(user_prompt)
        logger.debug("Intent=%s confidence=%s", intent_result.intent, intent_result.confidence)

        # Step 3: Decision
        decision_result = self.decision_engine.decide(
            regex_flag=regex_result.flag,
            regex_score=regex_result.score,
            intent=intent_result.intent,
            intent_score=intent_result.confidence,
        )

        result: Dict[str, Any] = {
            "timestamp": timestamp,
            "user_prompt": user_prompt,
            "decision": decision_result.decision.value,
            "safe_prompt": None,
            "input_truncated": truncated,
            "metadata": {
                "regex_analysis": {
                    "flag": regex_result.flag,
                    "matched_patterns": regex_result.matched_patterns,
                    "risk_score": regex_result.score,
                },
                "intent_analysis": {
                    "intent": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "class_scores": intent_result.class_scores,
                    "backend": getattr(intent_result, "backend", "unknown"),
                },
                "decision_reasoning": {
                    "reasoning": decision_result.reasoning,
                    "confidence": decision_result.confidence,
                    "rule_matched": decision_result.rule_matched,
                    "combined_score": decision_result.combined_score,
                },
                "sanitization": None,
                "action": None,
            },
        }

        # Step 4: Act on the decision
        if decision_result.decision == Decision.BLOCK:
            logger.warning("Prompt BLOCKED (%s)", decision_result.rule_matched)
            result["metadata"]["action"] = "blocked"

        elif decision_result.decision == Decision.SANITIZE:
            sanitized, summary = self.sanitizer.sanitize(user_prompt)
            result["metadata"]["sanitization"] = {
                "original_length": len(user_prompt),
                "sanitized_length": len(sanitized),
                "changes": summary,
                "sanitized_prompt": sanitized,
            }
            result["metadata"]["action"] = "sanitized"
            result["safe_prompt"] = self.sanitizer.wrap_safely(sanitized)
            logger.info("Prompt SANITIZED: %s", summary)

        else:
            result["metadata"]["action"] = "allowed"
            result["safe_prompt"] = user_prompt
            logger.info("Prompt ALLOWED")

        result["metadata"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result

    def guard(self, user_prompt: str, call_llm: bool = True, **llm_kwargs) -> Dict[str, Any]:
        """
        Run the complete guard pipeline and, unless blocked, call Gemini.

        Args:
            user_prompt: Raw user input
            call_llm: Set False to skip the Gemini call and only return the verdict
            **llm_kwargs: Forwarded to :meth:`llm.GeminiClient.call`

        Returns:
            The :meth:`analyze` payload plus a ``response`` field.
        """
        result = self.analyze(user_prompt)
        result["response"] = None

        if result["decision"] == Decision.BLOCK.value:
            result["response"] = self.decision_engine.get_safe_response()
            return result

        if not call_llm:
            return result

        if self.llm_client is None:
            result["error"] = self.llm_error or "LLM client not configured (set GEMINI_API_KEY)"
            return result

        try:
            result["response"] = self.llm_client.call(result["safe_prompt"], **llm_kwargs)
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            result["error"] = f"LLM call failed: {exc}"

        return result

    # -------------------------------------------------------------- evaluation

    def evaluate_on_test_set(
        self, test_prompts: Sequence[str], true_labels: Sequence[str]
    ) -> Dict[str, Any]:
        """
        Evaluate the guard's decisions against ground truth.

        Uses :meth:`analyze`, so evaluating a set costs nothing in API credits.

        Args:
            test_prompts: Prompts to evaluate
            true_labels: Ground truth decisions ("allow", "sanitize", "block")

        Returns:
            Accuracy plus a per-label precision/recall/F1 breakdown and the
            confusion matrix, keyed by decision.
        """
        if len(test_prompts) != len(true_labels):
            raise ValueError("test_prompts and true_labels must be the same length")
        if not test_prompts:
            raise ValueError("test_prompts must not be empty")

        predictions = [self.analyze(prompt)["decision"] for prompt in test_prompts]
        labels = [Decision.ALLOW.value, Decision.SANITIZE.value, Decision.BLOCK.value]

        correct = sum(1 for pred, truth in zip(predictions, true_labels) if pred == truth)
        confusion = {truth: {pred: 0 for pred in labels} for truth in labels}
        for pred, truth in zip(predictions, true_labels):
            if truth in confusion and pred in confusion[truth]:
                confusion[truth][pred] += 1

        per_label = {}
        for label in labels:
            true_pos = sum(1 for p, t in zip(predictions, true_labels) if p == label and t == label)
            pred_pos = sum(1 for p in predictions if p == label)
            actual_pos = sum(1 for t in true_labels if t == label)
            precision = true_pos / pred_pos if pred_pos else 0.0
            recall = true_pos / actual_pos if actual_pos else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": actual_pos,
            }

        results = {
            "accuracy": round(correct / len(test_prompts), 4),
            "total_tests": len(test_prompts),
            "correct": correct,
            "per_label": per_label,
            "confusion_matrix": confusion,
            "predictions": predictions,
            "true_labels": list(true_labels),
        }

        logger.info("Evaluation: accuracy = %.2f%%", 100 * results["accuracy"])
        return results


def _print_result(result: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return

    reasoning = result["metadata"]["decision_reasoning"]
    intent = result["metadata"]["intent_analysis"]
    regex = result["metadata"]["regex_analysis"]

    print("\n" + "-" * 60)
    print(f"Decision   : {result['decision'].upper()}")
    print(f"Confidence : {reasoning['confidence']:.2%}")
    print(f"Rule       : {reasoning['rule_matched']}")
    print(f"Reasoning  : {reasoning['reasoning']}")
    print(f"Intent     : {intent['intent']} ({intent['confidence']:.2%}) via {intent['backend']}")
    if regex["matched_patterns"]:
        print(f"Regex hits : {', '.join(regex['matched_patterns'])}")
    if result["metadata"].get("sanitization"):
        print(f"Sanitized  : {result['metadata']['sanitization']['changes']}")
    print(f"Latency    : {result['metadata']['latency_ms']} ms")
    if result.get("error"):
        print(f"Error      : {result['error']}")
    if result.get("response"):
        print(f"\nResponse:\n{result['response']}")
    print("-" * 60 + "\n")


def main() -> int:
    """CLI interface for the guard."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM prompt-injection guard")
    parser.add_argument(
        "prompt", nargs="*", help="Prompt to check. Omit for an interactive session."
    )
    parser.add_argument(
        "--backend", choices=["baseline", "transformer"], help="Classifier backend to use"
    )
    parser.add_argument(
        "--level", choices=["low", "medium", "high"], help="Sanitization aggressiveness"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Only classify; never call the Gemini API"
    )
    parser.add_argument("--json", action="store_true", help="Emit raw JSON results")
    args = parser.parse_args()

    guard = LLMGuard(
        classifier_backend=args.backend,
        sanitization_level=SanitizationLevel(args.level) if args.level else None,
        enable_llm=not args.no_llm,
    )

    if args.prompt:
        _print_result(guard.guard(" ".join(args.prompt), call_llm=not args.no_llm), args.json)
        return 0

    print("\n" + "=" * 60)
    print("LLM Prompt-Injection Guard")
    print("=" * 60)
    print("Enter prompts to test. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return 0

        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting...")
            return 0
        if not user_input:
            continue

        try:
            _print_result(guard.guard(user_input, call_llm=not args.no_llm), args.json)
        except Exception as exc:
            logger.error("Error: %s", exc)
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
