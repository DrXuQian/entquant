import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from entquant.quantization.nvfp4 import quantize_weight_absmax_nvfp4
from entquant.quantization.nvfp4_export import export_nvfp4_checkpoint
from entquant.quantization.nvfp4_optimizer import NVFP4EntQuantConfig


def test_export_nvfp4_checkpoint(tmp_path: Path):
    fp_dir = tmp_path / "fp"
    template_dir = tmp_path / "template"
    out_dir = tmp_path / "out"
    fp_dir.mkdir()
    template_dir.mkdir()

    weight = torch.randn(8, 16, dtype=torch.float32)
    save_file({"model.layers.0.self_attn.q_proj.weight": weight}, str(fp_dir / "model.safetensors"))

    baseline = quantize_weight_absmax_nvfp4(weight)
    template_state = {
        "model.layers.0.self_attn.q_proj.weight_packed": baseline.packed,
        "model.layers.0.self_attn.q_proj.weight_scale": baseline.encoded_scale,
        "model.layers.0.self_attn.q_proj.weight_global_scale": baseline.encoded_global_scale.reshape(1),
        "model.norm.weight": torch.ones(16, dtype=torch.float32),
    }
    save_file(template_state, str(template_dir / "model.safetensors"), metadata={"format": "pt"})
    (template_dir / "config.json").write_text(json.dumps({"architectures": ["ToyModel"]}))

    summary = export_nvfp4_checkpoint(
        full_precision_model_dir=fp_dir,
        template_nvfp4_dir=template_dir,
        output_dir=out_dir,
        config=NVFP4EntQuantConfig(reg_param=0.05, max_iters=5, block_chunk_size=64, device="cpu"),
        max_shard_size="1GB",
    )

    assert summary["num_modified_layers"] == 1
    assert (out_dir / "config.json").exists()
    assert (out_dir / "nvfp4_export_report.json").exists()
    assert any(name.endswith(".safetensors") for name in summary["shards"])

    model_file = out_dir / summary["shards"][0]
    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        assert "model.layers.0.self_attn.q_proj.weight_packed" in handle.keys()
        assert "model.layers.0.self_attn.q_proj.weight_scale" in handle.keys()
        assert "model.layers.0.self_attn.q_proj.weight_global_scale" in handle.keys()
