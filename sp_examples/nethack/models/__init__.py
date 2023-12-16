from sp_examples.nethack.models.chaotic_dwarf import ChaoticDwarvenGPT5
from sp_examples.nethack.models.scaled import ScaledNet

MODELS = [
    ChaoticDwarvenGPT5,
    ScaledNet,
]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}
