from .encoders.modules import GeneralConditioner

__all__ = [
    # `sgm.models.GeneralConditioner` is referenced in model configurations, etc.,
    # so it must be re-exported from this module.
    "GeneralConditioner",
    "UNCONDITIONAL_CONFIG",
]

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
