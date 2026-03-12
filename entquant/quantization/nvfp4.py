"""NVFP4 quantization helpers.

This is a minimal self-contained implementation of the NVFP4 weight format:
- FP4 E2M1 codebook
- 16-weight block scaling
- FP8 E4M3 scale encoding
- packed 2-nibble-per-byte storage
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

BLOCK_SIZE = 16
FP4_MAX_SCALE_VALUE = 448.0
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_EPS = 1e-6


@dataclass(frozen=True)
class NVFP4Tensor:
    packed: Tensor
    dequantized: Tensor
    codes: Tensor
    actual_scales: Tensor
    encoded_scale: Tensor
    encoded_global_scale: Tensor


def _codebook(device: torch.device) -> Tensor:
    return FP4_VALUES.to(device=device)


def pack_nvfp4_codes(codes: Tensor) -> Tensor:
    """Pack uint8 nibble codes into bytes."""
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {codes.shape[-1]}")
    return (codes[..., 1::2] << 4) | codes[..., 0::2]


def unpack_nvfp4_codes(packed: Tensor) -> Tensor:
    """Unpack uint8 bytes into NVFP4 nibbles."""
    unpacked_shape = list(packed.shape)
    unpacked_shape[-1] *= 2
    unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=packed.device)
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = packed >> 4
    return unpacked


def fp4_codes_to_values(codes: Tensor) -> Tensor:
    """Map NVFP4 code ids to FP4 E2M1 values."""
    return _codebook(codes.device)[codes.long()]


def quantize_to_fp4_codes(weight: Tensor) -> Tensor:
    """Quantize scaled weights to NVFP4 code ids."""
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinals = torch.zeros_like(weight_abs, dtype=torch.uint8)
    ordinals[(weight_abs > 0.25) & (weight_abs < 0.75)] = 1
    ordinals[(weight_abs >= 0.75) & (weight_abs <= 1.25)] = 2
    ordinals[(weight_abs > 1.25) & (weight_abs < 1.75)] = 3
    ordinals[(weight_abs >= 1.75) & (weight_abs <= 2.5)] = 4
    ordinals[(weight_abs > 2.5) & (weight_abs < 3.5)] = 5
    ordinals[(weight_abs >= 3.5) & (weight_abs <= 5.0)] = 6
    ordinals[weight_abs > 5.0] = 7
    return (sign_bit << 3) + ordinals


def absmax_actual_scales(weight: Tensor) -> Tensor:
    """Per-block absmax initialization."""
    weight = weight.to(torch.float32)
    if weight.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(f"Weight last dimension must be divisible by {BLOCK_SIZE}, got {tuple(weight.shape)}")
    blocks = weight.view(*weight.shape[:-1], -1, BLOCK_SIZE)
    return torch.clamp(blocks.abs().amax(dim=-1) / 6.0, min=_EPS)


def encode_actual_scales(scales: Tensor) -> tuple[Tensor, Tensor]:
    """Encode effective block scales to (fp8_scale, global_scale)."""
    scales = scales.to(torch.float32)
    max_scale = torch.max(scales)
    if float(max_scale) == 0.0:
        global_scale = torch.tensor([1.0], dtype=torch.float32, device=scales.device)
        fp8_scale = torch.zeros_like(scales, dtype=torch.float8_e4m3fn)
        return fp8_scale, global_scale

    global_scale = (FP4_MAX_SCALE_VALUE / max_scale).to(torch.float32).reshape(1)
    normalized = (scales * global_scale).clamp(min=0.0, max=FP4_MAX_SCALE_VALUE)
    fp8_scale = normalized.to(torch.float8_e4m3fn)
    return fp8_scale, global_scale


def quantize_to_nvfp4(weight: Tensor, actual_scales: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize to packed NVFP4 bytes using provided actual per-block scales."""
    weight = weight.to(torch.float32)
    if weight.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(f"Weight last dimension must be divisible by {BLOCK_SIZE}, got {tuple(weight.shape)}")
    scaled = weight.view(*weight.shape[:-1], -1, BLOCK_SIZE) / actual_scales.unsqueeze(-1)
    codes = quantize_to_fp4_codes(scaled.reshape_as(weight))
    packed = pack_nvfp4_codes(codes)
    dequantized = fp4_codes_to_values(codes).view(*weight.shape[:-1], -1, BLOCK_SIZE)
    dequantized = dequantized * actual_scales.unsqueeze(-1)
    return packed, dequantized.reshape_as(weight), codes


def quantize_weight_absmax_nvfp4(weight: Tensor) -> NVFP4Tensor:
    """Direct absmax NVFP4 quantization."""
    scales = absmax_actual_scales(weight)
    encoded_scale, encoded_global_scale = encode_actual_scales(scales)
    actual_scales = encoded_scale.to(torch.float32) / encoded_global_scale
    packed, dequantized, codes = quantize_to_nvfp4(weight, actual_scales)
    return NVFP4Tensor(
        packed=packed,
        dequantized=dequantized,
        codes=codes,
        actual_scales=actual_scales,
        encoded_scale=encoded_scale,
        encoded_global_scale=encoded_global_scale,
    )
