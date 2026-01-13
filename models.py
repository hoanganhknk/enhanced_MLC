# models.py
import torch
from resnet import resnet32

def _load_ssl_weights(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    cleaned = {}
    for k, v in sd.items():
        # handle common SSL prefixes
        for pref in ("encoder.module.", "encoder.", "module."):
            if k.startswith(pref):
                k = k[len(pref):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected

def build_resnet32(num_classes: int, ssl_path: str = ""):
    model = resnet32(num_classes=num_classes)
    if ssl_path:
        _load_ssl_weights(model, ssl_path)
    return model
