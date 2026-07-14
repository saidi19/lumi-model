"""Public Lumi model API and Hugging Face auto-class registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_lumi import LumiConfig
from .modeling_lumi import LumiForCausalLM, LumiModel
from .parameter_count import ParameterBreakdown, parameter_breakdown


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
    "ParameterBreakdown",
    "parameter_breakdown",
    "register_lumi",
]
