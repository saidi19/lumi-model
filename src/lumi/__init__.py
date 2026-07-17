"""Public Lumi model API and Hugging Face auto-class registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .checkpoint_evaluation import evaluate_checkpoint
from .configuration_lumi import LumiConfig
from .evaluation_data import build_heldout_evaluation, validate_heldout_manifest
from .modeling_lumi import (
    LumiForCausalLM,
    LumiModel,
    PackedDocumentMetadata,
    build_document_causal_mask,
    build_packed_document_metadata,
)
from .packed_data import PackedTokenDataset, build_packed_dataset, validate_packed_manifest
from .parameter_count import ParameterBreakdown, parameter_breakdown
from .pilot_training import train_packed_pilot
from .scaling_data import (
    build_scaling_packed_dataset,
    build_scaling_source_index,
    create_scaling_data_contract,
    validate_scaling_packed_manifest,
    validate_scaling_source_index,
)


def register_lumi() -> None:
    """Register local and exported-checkpoint auto classes; safe to call repeatedly."""

    AutoConfig.register(LumiConfig.model_type, LumiConfig, exist_ok=True)
    AutoModel.register(LumiConfig, LumiModel, exist_ok=True)
    AutoModelForCausalLM.register(LumiConfig, LumiForCausalLM, exist_ok=True)
    LumiConfig.register_for_auto_class("AutoConfig")
    LumiModel.register_for_auto_class("AutoModel")
    LumiForCausalLM.register_for_auto_class("AutoModelForCausalLM")


register_lumi()

__all__ = [
    "LumiConfig",
    "LumiForCausalLM",
    "LumiModel",
    "PackedTokenDataset",
    "PackedDocumentMetadata",
    "ParameterBreakdown",
    "build_packed_dataset",
    "build_document_causal_mask",
    "build_packed_document_metadata",
    "build_scaling_packed_dataset",
    "build_scaling_source_index",
    "build_heldout_evaluation",
    "evaluate_checkpoint",
    "parameter_breakdown",
    "create_scaling_data_contract",
    "register_lumi",
    "train_packed_pilot",
    "validate_heldout_manifest",
    "validate_packed_manifest",
    "validate_scaling_packed_manifest",
    "validate_scaling_source_index",
]
