from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "ldm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
