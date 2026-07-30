"""Gemini API client for LLM integration with error handling and safety settings."""

import os
import time
from typing import Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import config


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
            raise ValueError("GEMINI_API_KEY not found in environment or config")

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

                # Check for safety filtering
                if response.prompt_feedback.block_reason:
                    return (
                        "[BLOCKED] Response blocked by Gemini safety filter: "
                        f"{response.prompt_feedback.block_reason}"
                    )

                return response.text

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"API call failed (attempt {attempt + 1}). Retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")

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
            if chunk.text:
                yield chunk.text

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
