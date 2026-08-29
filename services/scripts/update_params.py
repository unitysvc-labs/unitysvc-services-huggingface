#!/usr/bin/env python3
"""
Template-based update_services.py for Hugging Face.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

# Provider Configuration
PROVIDER_NAME = "huggingface"
PROVIDER_DISPLAY_NAME = "Hugging Face"
ROUTER_API_URL = "https://router.huggingface.co/v1"
ENV_API_KEY_NAME = "HF_TOKEN"

SCRIPT_DIR = Path(__file__).parent
SPECS_DIR = SCRIPT_DIR.parent / "specs"


def committed_parameters(service_name: str) -> dict:
    """The parameters already committed for ``service_name`` ({} if it is new).

    unitysvc-sellers >= 0.3.1 keeps a committed value when the iterator yields
    ``None`` for it: from inside the writer, a lookup that failed and a lookup
    that legitimately found nothing are the same event. That is right for
    enrichment, but it means a price we FAILED to derive gets re-shipped as
    though it were this run's answer. Reading the previous value here is what
    separates the two cases — see the price guard in ``_build_template_vars``.
    """
    path = SPECS_DIR / f"{service_name}.json"
    if not path.is_file():
        return {}
    try:
        return (json.loads(path.read_text()) or {}).get("parameters") or {}
    except (OSError, ValueError):
        return {}


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
        if not self.litellm_data:
            print(
                "Error: LiteLLM model data came back empty. Every price lookup "
                "would fail, and unitysvc-sellers >= 0.3.1 would preserve the "
                "committed prices instead — re-shipping stale rate cards as "
                "though they were current."
            )
            sys.exit(1)

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
            # Not `return`. An empty iterator is indistinguishable from "the
            # upstream retired its whole catalog": with deprecate_missing the
            # writer would mark every committed service deprecated, and exiting
            # 0 would make a failed fetch look like a clean no-change run.
            sys.exit(1)

        if not models:
            print(
                "Error: upstream enumerated zero models — refusing to treat an "
                "empty enumeration as a retired catalog."
            )
            sys.exit(1)

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

    #: HuggingFace ``pipeline_tag`` -> platform capability vocabulary
    #: (unitysvc ``docs/capabilities.yml``). Anything not listed is a
    #: text-in/text-out task and maps to ``chat`` — including
    #: ``image-text-to-text`` and the other vision tags, since an image in
    #: the request is an attribute of a chat call, not a capability.
    _PIPELINE_CAPABILITY = {
        "feature-extraction": "embed",
        "sentence-similarity": "embed",
        "text-to-image": "image-generate",
        "unconditional-image-generation": "image-generate",
        "image-to-image": "image-edit",
        "automatic-speech-recognition": "speech-to-text",
        "text-to-speech": "text-to-speech",
        "text-to-audio": "text-to-speech",
        "text-to-video": "video-generate",
        "image-to-video": "video-generate",
    }

    @classmethod
    def _capability_for(cls, pipeline_tags: list[str]) -> str:
        """The platform capability implied by a HuggingFace pipeline tag."""
        tag = pipeline_tags[0] if pipeline_tags else ""
        return cls._PIPELINE_CAPABILITY.get(tag, "chat")

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict:
        """Build template variables for a model."""
        service_name = f"{PROVIDER_NAME}/{model_id}"
        service_type = self._determine_service_type(model_id)
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # HuggingFace's task taxonomy for this model. This is an UPSTREAM
        # fact, stored in details.pipeline_tag, and it is what both templates
        # branch on to pick the right code-example / connectivity presets.
        pipeline_tags, _ = ModelDataLookup.get_capabilities_from_hf(
            model_id, self.data_fetcher
        )
        # The platform capability is a separate axis: what the caller GETS
        # (unitysvc docs/capabilities.yml). Derived from the pipeline tag,
        # never equal to it.
        capabilities = [self._capability_for(pipeline_tags)]

        # Build details from LiteLLM data and model info
        details: dict[str, Any] = {}
        if pipeline_tags and pipeline_tags != ["llm"]:
            details["pipeline_tag"] = pipeline_tags[0]

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

        # Gate for the function-calling code example: only attach it when
        # LiteLLM positively records tool support. Models without the flag
        # (or absent from LiteLLM) fail the fc example against providers
        # that reject `tools` (TOOL_USE_NOT_SUPPORTED / UNSUPPORTED_OPENAI_PARAMS).
        # Corrections for LiteLLM optimism live in the per-model
        # <name>.override.json companions (merged at render time), so this
        # script never needs to change for one.
        supports_tools = bool(model_data and model_data.get("supports_function_calling"))

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

        # BYOK: the customer's own key pays the provider directly, so the service
        # is free through the gateway. Keep the price cell short ("Free (BYOK)");
        # the provider's reference rates go into pricing_note, which the template
        # renders as the closing paragraph of the offering description.
        pricing = {"type": "constant", "price": "0", "description": "Free (BYOK)"}
        pricing_note = None
        if model_data and "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
            # Per-token costs arrive as JSON floats (e.g. 5e-08). Scaling them
            # in binary float gives 0.049999999999999996 rather than 0.05, and
            # that string was written straight into pricing_note. Decimal(str(x))
            # recovers the intended decimal exactly — str() gives the shortest
            # round-tripping repr — and the arithmetic stays in Decimal.
            input_price = Decimal(str(model_data["input_cost_per_token"])) * 1_000_000
            output_price = Decimal(str(model_data["output_cost_per_token"])) * 1_000_000
            pricing_note = (
                f"${self._format_price(input_price)} / "
                f"${self._format_price(output_price)} "
                f"per 1M input/output tokens"
            )

        # `pricing_note` is the only field derived from the upstream rate card
        # here (the price itself is the constant "Free (BYOK)"). It is nullable,
        # it is a template param rather than a schema field, and so a failed
        # lookup is rejected by nothing downstream. Since unitysvc-sellers 0.3.1
        # preserves committed values against a yielded None, that failure now
        # SHIPS THE PREVIOUS RATE CARD as though it were this run's answer. A
        # model that has never appeared in the LiteLLM data has no committed
        # value and nothing to silently ship; a model that had one and can no
        # longer derive it is the regression, and it is fatal.
        if pricing_note is None and committed_parameters(service_name).get("pricing_note") is not None:
            print(
                f"  FATAL: {model_id} has a committed pricing_note but no "
                "input_cost_per_token/output_cost_per_token in this run's "
                "LiteLLM data. Refusing to re-ship the previous rate card."
            )
            sys.exit(1)

        return {
            # The service's name IS its path under specs/ (flat layout, #1263).
            # unitysvc-sellers >= 0.3.1 requires this key verbatim: `name_field`
            # is gone and there is no fallback for a dict that omits it.
            "service_name": service_name,
            # Offering name is the bare upstream model_id
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": service_type,
            "capabilities": capabilities,
            "supports_tools": supports_tools,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Listing fields
            "list_price": pricing,
            # Reference rates for the BYOK pricing paragraph (template-rendered)
            "pricing_note": pricing_note,
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

    def _format_price(self, price: Decimal) -> str:
        """Render a Decimal price as a plain string.

        normalize() strips trailing zeros ("2.50" -> "2.5", "2.0" -> "2") but can
        emit scientific notation for round values ("100" -> "1E+2"), so format
        with "f" to force plain digits.
        """
        return format(price.normalize(), "f")


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    write_params_from_iterator(
        iterator=source.iter_models(),
        output_dir=SPECS_DIR,
    )


if __name__ == "__main__":
    main()
