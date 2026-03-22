#!/usr/bin/env python3
"""
Template-based update_services.py for Hugging Face.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import any_llm

from unitysvc_services import ModelDataFetcher, ModelDataLookup, populate_from_iterator

# Provider Configuration
PROVIDER_NAME = "huggingface"
PROVIDER_DISPLAY_NAME = "Hugging Face"
API_BASE_URL = "https://api-inference.huggingface.co/models/"
ENV_API_KEY_NAME = "HF_TOKEN"

SCRIPT_DIR = Path(__file__).parent


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            models = any_llm.list_models(PROVIDER_NAME, api_key=self.api_key)
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        for i, model in enumerate(models, 1):
            model_info = json.loads(model.to_json())
            model_id = model_info.get("id", str(model))
            print(f"[{i}/{len(models)}] {model_id}")

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

    def _determine_example_suffix(self, model_id: str) -> str:
        name = model_id.lower()

        if "guard" in name or "moderation" in name:
            return "-guard"

        if any(k in name for k in ["imagetoimage", "image-edit", "outpaint", "inpaint"]):
            return "-imagetoimage"

        if any(k in name for k in ["diffusion", "stable-diffusion", "sdxl", "flux", "image"]):
            return "-image"

        if any(k in name for k in ["whisper", "asr", "transcription", "speech"]):
            return "-prerecordedtranscription"

        if any(k in name for k in ["sentence", "embedding", "minilm", "bge", "e5"]):
            return "-sentencetransformers"

        if any(k in name for k in ["tts", "text-to-speech", "kokoro", "bark"]):
            return "-tts"

        if any(k in name for k in ["video", "ltx", "text-to-video", "t2v"]):
            return "-ttv"

        return ""
    
    # Map HuggingFace pipeline_tag to our service types
    PIPELINE_TAG_MAP: dict[str, str] = {
        "text-generation": "llm",
        "text2text-generation": "llm",
        "conversational": "llm",
        "feature-extraction": "embedding",
        "sentence-similarity": "embedding",
        "fill-mask": "llm",
        "text-classification": "llm",
        "token-classification": "llm",
        "question-answering": "llm",
        "summarization": "llm",
        "translation": "prerecorded_translation",
        "text-to-image": "text_to_image",
        "image-to-image": "image_generation",
        "image-to-text": "vision_language_model",
        "visual-question-answering": "vision_language_model",
        "image-text-to-text": "vision_language_model",
        "image-classification": "vision_language_model",
        "object-detection": "vision_language_model",
        "image-segmentation": "vision_language_model",
        "automatic-speech-recognition": "speech_to_text",
        "text-to-speech": "text_to_speech",
        "text-to-audio": "text_to_speech",
        "audio-classification": "speech_to_text",
        "text-to-video": "video_generation",
        "image-to-video": "video_generation",
        "text-to-3d": "text_to_3d",
        "video-text-to-text": "vision_language_model",
    }

    # Map pipeline_tag to code example template suffix
    PIPELINE_EXAMPLE_MAP: dict[str, str] = {
        "text-generation": "",
        "text2text-generation": "",
        "conversational": "",
        "feature-extraction": "-sentencetransformers",
        "sentence-similarity": "-sentencetransformers",
        "text-to-image": "-image",
        "image-to-image": "-imagetoimage",
        "image-to-text": "",
        "image-text-to-text": "",
        "visual-question-answering": "",
        "automatic-speech-recognition": "-prerecordedtranscription",
        "text-to-speech": "-tts",
        "text-to-audio": "-tts",
        "text-to-video": "-ttv",
        "image-to-video": "-ttv",
    }

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict:
        """Build template variables for a model."""
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Fetch HuggingFace model details for pipeline_tag and tags
        hf_details = self.data_fetcher.fetch_huggingface_model_details(model_id, quiet=True)
        pipeline_tag = hf_details.get("pipeline_tag") if hf_details else None
        hf_tags = hf_details.get("tags", []) if hf_details else []

        # Determine service type from pipeline_tag (preferred) or name heuristics (fallback)
        if pipeline_tag and pipeline_tag in self.PIPELINE_TAG_MAP:
            service_type = self.PIPELINE_TAG_MAP[pipeline_tag]
        else:
            service_type = self._determine_service_type(model_id)

        # Determine example suffix from pipeline_tag (preferred) or name heuristics (fallback)
        if pipeline_tag and pipeline_tag in self.PIPELINE_EXAMPLE_MAP:
            example_suffix = self.PIPELINE_EXAMPLE_MAP[pipeline_tag]
        else:
            example_suffix = self._determine_example_suffix(model_id)

        # Build details from LiteLLM data and model info
        details: dict[str, Any] = {}
        if pipeline_tag:
            details["pipeline_tag"] = pipeline_tag
        if hf_tags:
            details["hf_tags"] = [t for t in hf_tags if not t.startswith("base_model:") and not t.startswith("region:")]

        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "max_input_tokens" in model_data:
                details["contextLength"] = model_data["max_input_tokens"]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Extract pricing
        pricing = None
        if model_data:
            if "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
                input_price = float(
                    model_data["input_cost_per_token"]) * 1_000_000
                output_price = float(
                    model_data["output_cost_per_token"]) * 1_000_000
                pricing = {
                    "type": "one_million_tokens",
                    "input": self._format_price(input_price),
                    "output": self._format_price(output_price),
                    "description": "Pricing Per 1M Tokens Input/Output",
                    "reference": None,
                }

        return {
            # Directory name uses -byok suffix (used by populate_from_iterator)
            "name": f"{model_id}-byok",
            # Offering name is the model_id (without -byok suffix)
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": service_type,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Listing fields
            "list_price": pricing,
            # Provider config (for templates)
            "example_suffix": example_suffix,
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": "https://router.huggingface.co/v1",
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    def _determine_service_type(self, model_id: str) -> str:
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in ["embed", "embedding"]):
            return "embedding"
        if any(kw in model_lower for kw in ["rerank"]):
            return "rerank"
        if any(kw in model_lower for kw in ["vision"]):
            return "vision_language_model"
        return "llm"

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    populate_from_iterator(
        iterator=source.iter_models(),
        templates_dir=SCRIPT_DIR.parent / "templates",
        output_dir=SCRIPT_DIR.parent / "services",
    )


if __name__ == "__main__":
    main()
