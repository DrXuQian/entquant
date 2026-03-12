"""Minimal NVFP4 EntQuant prototype walkthrough."""

from __future__ import annotations

import torch

from entquant.quantization.nvfp4 import quantize_weight_absmax_nvfp4
from entquant.quantization.nvfp4_optimizer import NVFP4EntQuantConfig, optimize_nvfp4_tensor_entquant


def main() -> None:
    torch.manual_seed(0)
    weight = torch.randn(256, 256, dtype=torch.float32)

    baseline = quantize_weight_absmax_nvfp4(weight)
    optimized = optimize_nvfp4_tensor_entquant(
        weight,
        NVFP4EntQuantConfig(
            variant="entquant_exact",
            reg_param=0.05,
            lr=1.0,
            max_iters=50,
            block_chunk_size=2048,
            device="cpu",
        ),
    )

    baseline_mse = torch.mean((baseline.dequantized - weight) ** 2).item()
    optimized_mse = torch.mean((optimized.quantized.dequantized - weight) ** 2).item()
    print("Baseline MSE:", baseline_mse)
    print("Optimized MSE:", optimized_mse)
    print("Report:", optimized.report)


if __name__ == "__main__":
    main()
