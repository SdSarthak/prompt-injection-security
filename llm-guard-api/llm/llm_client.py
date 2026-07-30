"""Gemini API client for LLM integration with error handling and safety settings."""

import logging
import time
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import config

logger = logging.getLogger(__name__)

# Failures that will never succeed on retry - retrying just wastes wall clock.
_NON_RETRYABLE = ("api key", "permission", "unauthenticated", "invalid argument", "not found")


class GeminiClient:
    """Thin, pluggable Gemini API wrapper with safety settings and retry logic."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (uses config.GEMINI_API_KEY if not provided)
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Copy .env.example to .env and set your key, "
                "or export GEMINI_API_KEY."
            )

        genai.configure(api_key=self.api_key)
        self.model_name = config.GEMINI_MODEL
        self.model = genai.GenerativeModel(self.model_name)

    def call(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        top_k: int = 40,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Call Gemini API with exponential backoff retry logic.
        
        Args:
            prompt: Prompt to send to Gemini
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_retries: Maximum number of retry attempts
            retry_delay: Initial retry delay in seconds (exponential backoff)
            
        Returns:
            Model response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        safety_settings = self._get_safety_settings()

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": top_p,
                        "top_k": top_k,
                    },
                    safety_settings=safety_settings,
                )

                return self._extract_text(response)

            except Exception as exc:
                message = str(exc).lower()
                if any(marker in message for marker in _NON_RETRYABLE):
                    raise
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        "Gemini call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        exc,
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API call failed after {max_retries} attempts: {exc}") from exc

        raise RuntimeError("API call failed: retry loop exhausted")

    @staticmethod
    def _extract_text(response) -> str:
        """Pull the text out of a response, tolerating filtered/empty candidates.

        `response.text` raises when the model returns no usable candidate (a
        safety block, a recitation stop, or an empty completion), so those cases
        are converted into an explicit marker string instead of an exception.
        """
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            return f"[BLOCKED] Prompt blocked by the Gemini safety filter: {block_reason}"

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "[EMPTY] Gemini returned no candidates"

        try:
            return response.text
        except Exception:
            finish_reason = getattr(candidates[0], "finish_reason", "unknown")
            return f"[BLOCKED] Gemini returned no usable text (finish_reason={finish_reason})"

    def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        top_k: int = 40,
    ):
        """
        Call Gemini API with streaming response.
        
        Args:
            prompt: Prompt to send to Gemini
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Yields:
            Streamed response chunks
        """
        safety_settings = self._get_safety_settings()

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
            },
            safety_settings=safety_settings,
            stream=True,
        )

        for chunk in response:
            # A filtered chunk raises on `.text` rather than returning empty.
            try:
                text = chunk.text
            except Exception:
                continue
            if text:
                yield text

    def _get_safety_settings(self):
        """
        Get safety settings for Gemini API.
        
        Returns:
            List of safety settings tuples
        """
        return [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
        ]

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        model = genai.get_model(self.model_name)
        return {
            "name": model.name,
            "display_name": model.display_name,
            "input_token_limit": model.input_token_limit,
            "output_token_limit": model.output_token_limit,
        }
