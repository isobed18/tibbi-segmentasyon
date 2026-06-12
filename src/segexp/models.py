from __future__ import annotations

from typing import Any

import torch
from monai.networks.nets import AttentionUnet, SwinUNETR, UNet


def build_model(name: str, in_channels: int, image_size: int) -> torch.nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
        )
    if name == "attention_unet":
        return AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
        )
    if name == "swin_unetr":
        # A compact 2D SwinUNETR keeps the Transformer comparison feasible on repeated runs.
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=1,
            feature_size=24,
            spatial_dims=2,
            use_checkpoint=True,
        )
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_model(model: torch.nn.Module, in_channels: int, image_size: int, device: torch.device) -> dict[str, Any]:
    model = model.to(device).eval()
    dummy = torch.randn(1, in_channels, image_size, image_size, device=device)
    params = count_parameters(model)
    flops = None
    try:
        from thop import profile

        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception as exc:  # FLOPs are helpful, not mandatory for training.
        flops = None
        flop_error = str(exc)
    else:
        flop_error = None
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = 0
    return {
        "parameters": params,
        "flops": flops,
        "flops_g": None if flops is None else round(flops / 1e9, 4),
        "flop_error": flop_error,
        "profile_peak_memory_mb": round(peak / 1024**2, 2),
    }
