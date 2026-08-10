"""
__init__.py

for Tiresias model in Logion app

elide need for trust_remote_code
"""
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from .configuration_tiresias import TiresiasConfig
from .modeling_tiresias import TiresiasModel, TiresiasForMaskedLM

AutoConfig.register("tiresias", TiresiasConfig, exist_ok=True)
AutoModel.register(TiresiasConfig, TiresiasModel, exist_ok=True)
AutoModelForMaskedLM.register(TiresiasConfig, TiresiasForMaskedLM, exist_ok=True)