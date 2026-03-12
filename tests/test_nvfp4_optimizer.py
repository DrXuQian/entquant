import torch

from entquant.quantization.nvfp4 import (
    BLOCK_SIZE,
    fp4_codes_to_values,
    pack_nvfp4_codes,
    quantize_to_fp4_codes,
    quantize_weight_absmax_nvfp4,
    unpack_nvfp4_codes,
)
from entquant.quantization.nvfp4_optimizer import (
    NVFP4EntQuantConfig,
    optimize_nvfp4_tensor_entquant,
    soft_code_entropy_bits,
)


def test_pack_unpack_roundtrip():
    codes = torch.arange(0, 16, dtype=torch.uint8).repeat(2, 1)
    packed = pack_nvfp4_codes(codes)
    unpacked = unpack_nvfp4_codes(packed)
    assert torch.equal(codes, unpacked)


def test_absmax_quantization_shapes():
    weight = torch.randn(32, 64)
    result = quantize_weight_absmax_nvfp4(weight)
    assert result.packed.shape == (32, 32)
    assert result.dequantized.shape == weight.shape
    assert result.codes.shape == weight.shape
    assert result.actual_scales.shape == (32, 4)


def test_soft_code_entropy_prefers_concentrated_values():
    concentrated = torch.zeros(64, BLOCK_SIZE)
    dispersed = torch.linspace(-6.0, 6.0, steps=64 * BLOCK_SIZE).reshape(64, BLOCK_SIZE)
    assert soft_code_entropy_bits(concentrated) < soft_code_entropy_bits(dispersed)


def test_nvfp4_entquant_optimizer_runs():
    torch.manual_seed(0)
    weight = torch.randn(64, 64)
    result = optimize_nvfp4_tensor_entquant(
        weight,
        NVFP4EntQuantConfig(
            variant="entquant_exact",
            reg_param=0.05,
            lr=1.0,
            max_iters=10,
            block_chunk_size=256,
            device="cpu",
        ),
    )
    assert result.quantized.dequantized.shape == weight.shape
    assert result.quantized.packed.dtype == torch.uint8
    assert result.report.num_blocks == (weight.numel() // BLOCK_SIZE)
