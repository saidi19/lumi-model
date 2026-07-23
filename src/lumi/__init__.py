"""Public Lumi API with optional lazy loading of the model-training stack."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

_PUBLIC_ATTRIBUTES = {
    "DistributedContext": (".distributed_training", "DistributedContext"),
    "EpochPermutationPlanner": (".distributed_training", "EpochPermutationPlanner"),
    "FinalCorpusTokenDataset": (".production_training", "FinalCorpusTokenDataset"),
    "LumiConfig": (".configuration_lumi", "LumiConfig"),
    "LumiForCausalLM": (".modeling_lumi", "LumiForCausalLM"),
    "LumiModel": (".modeling_lumi", "LumiModel"),
    "PackedDocumentMetadata": (".modeling_lumi", "PackedDocumentMetadata"),
    "PackedTokenDataset": (".packed_data", "PackedTokenDataset"),
    "ParameterBreakdown": (".parameter_count", "ParameterBreakdown"),
    "build_document_causal_mask": (".modeling_lumi", "build_document_causal_mask"),
    "build_account_segment_plan": (".production_training", "build_account_segment_plan"),
    "build_heldout_evaluation": (".evaluation_data", "build_heldout_evaluation"),
    "build_packed_dataset": (".packed_data", "build_packed_dataset"),
    "build_packed_document_metadata": (".modeling_lumi", "build_packed_document_metadata"),
    "build_production_training_preflight_plan": (
        ".production_training",
        "build_production_training_preflight_plan",
    ),
    "build_scaling_packed_dataset": (".scaling_data", "build_scaling_packed_dataset"),
    "build_scaling_source_index": (".scaling_data", "build_scaling_source_index"),
    "create_scaling_data_contract": (".scaling_data", "create_scaling_data_contract"),
    "evaluate_checkpoint": (".checkpoint_evaluation", "evaluate_checkpoint"),
    "parameter_breakdown": (".parameter_count", "parameter_breakdown"),
    "plan_4b_80b_production": (
        ".production_pretraining",
        "plan_4b_80b_production",
    ),
    "train_distributed_packed": (".distributed_training", "train_distributed_packed"),
    "train_packed_pilot": (".pilot_training", "train_packed_pilot"),
    "validate_final_corpus_training_manifest": (
        ".production_training",
        "validate_final_corpus_training_manifest",
    ),
    "validate_heldout_manifest": (".evaluation_data", "validate_heldout_manifest"),
    "validate_packed_manifest": (".packed_data", "validate_packed_manifest"),
    "validate_scaling_packed_manifest": (
        ".scaling_data",
        "validate_scaling_packed_manifest",
    ),
    "validate_scaling_source_index": (".scaling_data", "validate_scaling_source_index"),
    "verify_distributed_checkpoint": (
        ".distributed_training",
        "verify_distributed_checkpoint",
    ),
}


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRIBUTES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def register_lumi() -> None:
    """Register local and exported-checkpoint auto classes; safe to call repeatedly."""

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from .configuration_lumi import LumiConfig
    from .modeling_lumi import LumiForCausalLM, LumiModel

    AutoConfig.register(LumiConfig.model_type, LumiConfig, exist_ok=True)
    AutoModel.register(LumiConfig, LumiModel, exist_ok=True)
    AutoModelForCausalLM.register(LumiConfig, LumiForCausalLM, exist_ok=True)
    LumiConfig.register_for_auto_class("AutoConfig")
    LumiModel.register_for_auto_class("AutoModel")
    LumiForCausalLM.register_for_auto_class("AutoModelForCausalLM")


# Preserve the historical auto-registration behavior in full model environments,
# while allowing CPU-only data workers to import ``lumi.production_corpus`` without
# installing PyTorch or Transformers.
if importlib.util.find_spec("transformers") is not None:
    register_lumi()


__all__ = [*_PUBLIC_ATTRIBUTES, "register_lumi"]
