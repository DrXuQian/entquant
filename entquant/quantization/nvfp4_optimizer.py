"""EntQuant-style offline scale optimization for NVFP4 tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .nvfp4 import (
    BLOCK_SIZE,
    NVFP4Tensor,
    absmax_actual_scales,
    encode_actual_scales,
    fp4_codes_to_values,
    quantize_to_fp4_codes,
    quantize_to_nvfp4,
    quantize_weight_absmax_nvfp4,
)
from .utils import LpNormDistance

_EPS = 1e-6
_FP4_VALUES_F32 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_CODE_SIGNS = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1])


@dataclass(frozen=True)
class NVFP4EntQuantConfig:
    variant: Literal["entquant_exact", "entquant_soft"] = "entquant_exact"
    norm_type: Literal["absolute", "relative", "relative_entrywise", "mean"] = "relative"
    norm_p: float = 1.0
    reg_param: float = 3.9
    soft_param: float = 0.0
    temperature: float = 0.20
    lr: float = 1.0
    max_iters: int = 80
    block_chunk_size: int = 8192
    device: str = "cuda"
    verbose: bool = False


@dataclass(frozen=True)
class NVFP4EntQuantOptimizationReport:
    variant: str
    norm_type: str
    norm_p: float
    reg_param: float
    soft_param: float
    temperature: float
    lr: float
    max_iters: int
    block_chunk_size: int
    num_blocks: int
    initial_scale_mean: float
    initial_scale_std: float
    optimized_scale_mean: float
    optimized_scale_std: float
    optimized_scale_min: float
    optimized_scale_max: float


@dataclass(frozen=True)
class NVFP4EntQuantResult:
    quantized: NVFP4Tensor
    report: NVFP4EntQuantOptimizationReport


class _NVFP4ScaleQuantizer(nn.Module):
    def __init__(self, initial_scales: Tensor):
        super().__init__()
        self.log2_scale = nn.Parameter(torch.log2(initial_scales.float()))

    @property
    def scale(self) -> Tensor:
        return torch.exp2(self.log2_scale)

    def forward(self, weight_blocks: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scale.unsqueeze(-1)
        scaled = weight_blocks * torch.exp2(-self.log2_scale).unsqueeze(-1)
        hard_codes = quantize_to_fp4_codes(scaled.reshape(-1)).view_as(weight_blocks)
        hard_values = fp4_codes_to_values(hard_codes).to(torch.float32).view_as(weight_blocks)
        ste_values = scaled + (hard_values - scaled).detach()
        dequantized = ste_values * scale
        return dequantized, ste_values, hard_codes, scaled


def _soft_code_logits(scaled: Tensor, temperature: float) -> Tensor:
    codebook = _FP4_VALUES_F32.to(device=scaled.device)
    code_signs = _CODE_SIGNS.to(device=scaled.device, dtype=torch.float32)
    scaled_expanded = scaled.unsqueeze(-1)
    logits = -(scaled_expanded - codebook).abs() / max(temperature, _EPS)
    sign = torch.sign(scaled_expanded)
    mismatch = (sign * code_signs < 0).to(logits.dtype)
    logits = logits - 0.25 * mismatch
    logits[..., 0] = logits[..., 0] - 0.10 * (scaled < 0).to(logits.dtype)
    logits[..., 8] = logits[..., 8] - 0.10 * (scaled >= 0).to(logits.dtype)
    return logits


def soft_code_entropy_bits(scaled: Tensor, temperature: float = 0.20) -> Tensor:
    probs = torch.softmax(_soft_code_logits(scaled, temperature), dim=-1)
    hist = probs.mean(dim=tuple(range(probs.ndim - 1)))
    return -(hist * torch.log2(hist.clamp_min(_EPS))).sum()


def entquant_l1_regularizer(quantized_values: Tensor) -> Tensor:
    return (torch.abs(quantized_values) * 4.0).mean()


def _optimize_chunk(weight_blocks: Tensor, initial_scales: Tensor, config: NVFP4EntQuantConfig, lr: float | None = None) -> Tensor:
    lr = config.lr if lr is None else lr
    quantizer = _NVFP4ScaleQuantizer(initial_scales).to(weight_blocks.device)
    dist_fun = LpNormDistance(norm_type=config.norm_type, p=config.norm_p).to(weight_blocks.device)
    optimizer = torch.optim.LBFGS(
        quantizer.parameters(),
        lr=lr,
        max_iter=config.max_iters,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        dequantized, ste_values, _, scaled = quantizer(weight_blocks)
        rec_loss = dist_fun(dequantized - weight_blocks, weight_blocks)
        reg_loss = config.reg_param * entquant_l1_regularizer(ste_values)
        if config.variant == "entquant_soft":
            reg_loss = reg_loss + config.soft_param * soft_code_entropy_bits(scaled, temperature=config.temperature)
        loss = rec_loss + reg_loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NVFP4 EntQuant loss became non-finite.")
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except (RuntimeError, ValueError):
        if lr <= 1e-3:
            return initial_scales
        return _optimize_chunk(weight_blocks, initial_scales, config, lr=lr * 0.5)

    with torch.no_grad():
        optimized = torch.clamp(quantizer.scale, min=_EPS)
    return optimized.to(initial_scales.device)


def optimize_nvfp4_tensor_entquant(weight: Tensor, config: NVFP4EntQuantConfig | None = None) -> NVFP4EntQuantResult:
    """Optimize per-block NVFP4 scales against a full-precision tensor."""
    config = config or NVFP4EntQuantConfig()
    weight = weight.to(torch.float32)
    if weight.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(f"Weight last dimension must be divisible by {BLOCK_SIZE}, got {tuple(weight.shape)}")

    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    initial = quantize_weight_absmax_nvfp4(weight)
    initial_scales = initial.actual_scales.to(torch.float32).reshape(-1)
    weight_blocks = weight.reshape(-1, BLOCK_SIZE)
    optimized_scales = torch.empty_like(initial_scales)

    for start in range(0, weight_blocks.shape[0], config.block_chunk_size):
        end = min(start + config.block_chunk_size, weight_blocks.shape[0])
        optimized_scales[start:end] = _optimize_chunk(
            weight_blocks[start:end].to(device),
            initial_scales[start:end].to(device),
            config,
        ).cpu()

    optimized_scales = optimized_scales.view_as(initial.actual_scales)
    encoded_scale, encoded_global_scale = encode_actual_scales(optimized_scales)
    actual_scales = encoded_scale.to(torch.float32) / encoded_global_scale
    packed, dequantized, codes = quantize_to_nvfp4(weight, actual_scales)
    quantized = NVFP4Tensor(
        packed=packed,
        dequantized=dequantized,
        codes=codes,
        actual_scales=actual_scales,
        encoded_scale=encoded_scale,
        encoded_global_scale=encoded_global_scale,
    )
    report = NVFP4EntQuantOptimizationReport(
        variant=config.variant,
        norm_type=config.norm_type,
        norm_p=config.norm_p,
        reg_param=config.reg_param,
        soft_param=config.soft_param,
        temperature=config.temperature,
        lr=config.lr,
        max_iters=config.max_iters,
        block_chunk_size=config.block_chunk_size,
        num_blocks=int(actual_scales.numel()),
        initial_scale_mean=float(initial_scales.mean().item()),
        initial_scale_std=float(initial_scales.std().item()),
        optimized_scale_mean=float(actual_scales.mean().item()),
        optimized_scale_std=float(actual_scales.std().item()),
        optimized_scale_min=float(actual_scales.min().item()),
        optimized_scale_max=float(actual_scales.max().item()),
    )
    return NVFP4EntQuantResult(quantized=quantized, report=report)
