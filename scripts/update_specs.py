#!/usr/bin/env python3
"""
Template-based update_services.py for Hugging Face.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.template_populate import populate_from_iterator

# Provider Configuration
PROVIDER_NAME = "huggingface"
PROVIDER_DISPLAY_NAME = "Hugging Face"
ROUTER_API_URL = "https://router.huggingface.co/v1"
ENV_API_KEY_NAME = "HF_TOKEN"

SCRIPT_DIR = Path(__file__).parent


def _hf_canonical_id(raw: str) -> str:
    """huggingface directory naming uses 'org_model' instead of 'org/model';
    swap the FIRST underscore so canonical helpers hit the HF API correctly."""
    if "_" in raw and "/" not in raw:
        return raw.replace("_", "/", 1)
    return raw


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

        # Fetch available models directly from HF Inference Providers API
        # This returns only models actually available for inference
        print(f"Fetching available models from {PROVIDER_DISPLAY_NAME} Inference API...")
        try:
            r = httpx.get(
                f"{ROUTER_API_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            print(f"Found {len(models)} available models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("id", "")
            if not model_id:
                continue
            print(f"[{i}/{len(models)}] {model_id}", end="")

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict:
        """Build template variables for a model."""
        service_type = self._determine_service_type(model_id)
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Get capabilities from HuggingFace pipeline_tag
        capabilities, _ = ModelDataLookup.get_capabilities_from_hf(
            model_id, self.data_fetcher
        )

        # Get cleaned HF tags for details
        hf_tags = ModelDataLookup.get_hf_tags(model_id, self.data_fetcher)

        # Build details from LiteLLM data and model info
        details: dict[str, Any] = {}
        if capabilities and capabilities != ["llm"]:
            details["pipeline_tag"] = capabilities[0]
        if hf_tags:
            details["hf_tags"] = hf_tags

        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  metadata_sources records provenance so reviewers
        # can triage stale-value reports.  Note: HF directory naming uses
        # 'org_model' rather than 'org/model'; normalize before lookup so
        # the canonical helper's HF API calls resolve correctly.
        canonical = ModelDataLookup.get_canonical_metadata(
            _hf_canonical_id(model_id),
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

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
            "capabilities": capabilities,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Listing fields
            "list_price": pricing,
            # Provider config (for templates)
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
            return "embedding"
        # vision_language_model → llm (vision is a capability, not a service type)
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
