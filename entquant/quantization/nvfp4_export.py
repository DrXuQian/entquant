"""Standalone export of a full NVFP4 checkpoint directory.

This exporter is intentionally template-based:
- full-precision weights come from a standard HF model directory
- layout and non-weight files come from an existing NVFP4 template directory
- selected weight tensors are re-quantized with the NVFP4 EntQuant optimizer

The output is a final checkpoint directory with:
- copied config/tokenizer/template assets
- exported ``model.safetensors`` or sharded ``model-xxxxx.safetensors``
- an export JSON report
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file

from .nvfp4_optimizer import NVFP4EntQuantConfig, optimize_nvfp4_tensor_entquant

CHUNK_SIZE_BYTES = 32 * 1024


def _match_any(name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def compute_entropy_bits(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    counts = np.bincount(data.reshape(-1), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / float(data.size)
    return float(-(probs * np.log2(probs)).sum())


def compute_nvfp4_compression_rate(weight_packed: np.ndarray, chunk_size_bytes: int = CHUNK_SIZE_BYTES) -> dict[str, float]:
    flat = weight_packed.reshape(-1)
    if flat.size == 0:
        return {
            "mean_entropy_bits": 0.0,
            "std_entropy_bits": 0.0,
            "compression_rate": 0.0,
            "compressed_size_ratio": 0.0,
            "n_chunks": 0.0,
        }

    n_chunks = flat.size // chunk_size_bytes
    if n_chunks == 0:
        chunks = [flat[:chunk_size_bytes]]
    else:
        chunks = [flat[i * chunk_size_bytes : (i + 1) * chunk_size_bytes] for i in range(n_chunks)]

    entropies = []
    for chunk in chunks:
        even_bytes = chunk[0::2]
        odd_bytes = chunk[1::2]
        h_even = compute_entropy_bits(even_bytes)
        h_odd = compute_entropy_bits(odd_bytes)
        entropies.append((h_even + h_odd) / 2.0)

    mean_entropy = float(np.mean(entropies))
    std_entropy = float(np.std(entropies))
    compressed_ratio = mean_entropy / 8.0
    return {
        "mean_entropy_bits": mean_entropy,
        "std_entropy_bits": std_entropy,
        "compression_rate": float(1.0 - compressed_ratio),
        "compressed_size_ratio": float(compressed_ratio),
        "n_chunks": float(len(entropies)),
    }


def _weighted_average(reports: list[dict[str, Any]], field: str) -> float | None:
    total_weight = 0
    total = 0.0
    for report in reports:
        if field not in report:
            continue
        num_weights = int(report["num_weights"])
        total_weight += num_weights
        total += num_weights * float(report[field])
    if total_weight == 0:
        return None
    return total / total_weight


def list_nvfp4_weight_prefixes(
    template_safetensors: str | Path,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """Return weight prefixes that expose NVFP4 packed/scale tensors."""
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    template_safetensors = Path(template_safetensors)
    prefixes: list[str] = []
    with safe_open(str(template_safetensors), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    for key in sorted(keys):
        if not key.endswith(".weight_packed"):
            continue
        prefix = key[: -len(".weight_packed")]
        if prefix + ".weight_scale" not in keys or prefix + ".weight_global_scale" not in keys:
            continue
        if include_patterns and not _match_any(prefix, include_patterns):
            continue
        if exclude_patterns and _match_any(prefix, exclude_patterns):
            continue
        prefixes.append(prefix)
    return prefixes


class ShardedTensorLoader:
    """Read tensors from either a single safetensors file or an indexed HF shard set."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        index_path = model_dir / "model.safetensors.index.json"
        single_path = model_dir / "model.safetensors"
        if index_path.exists():
            with index_path.open() as handle:
                self._weight_map = json.load(handle)["weight_map"]
        elif single_path.exists():
            with safe_open(str(single_path), framework="pt", device="cpu") as handle:
                self._weight_map = {key: single_path.name for key in handle.keys()}
        else:
            raise FileNotFoundError(
                f"Could not find model.safetensors or model.safetensors.index.json in {model_dir}"
            )
        self._model_dir = model_dir
        self._handles: dict[str, Any] = {}
        self._stack: Any = None

    def __enter__(self) -> "ShardedTensorLoader":
        import contextlib

        self._stack = contextlib.ExitStack()
        for shard_name in sorted(set(self._weight_map.values())):
            self._handles[shard_name] = self._stack.enter_context(
                safe_open(str(self._model_dir / shard_name), framework="pt", device="cpu")
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None
        self._handles.clear()

    def get_tensor(self, key: str) -> torch.Tensor:
        shard_name = self._weight_map[key]
        return self._handles[shard_name].get_tensor(key)


def _copy_non_weight_files(source_dir: Path, output_dir: Path) -> None:
    ignore = shutil.ignore_patterns("model.safetensors", "model.safetensors.index.json", ".git")
    shutil.copytree(source_dir, output_dir, ignore=ignore)


def _save_sharded_state_dict(
    *,
    output_dir: Path,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
    max_shard_size: int | str,
) -> list[str]:
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )

    shard_names: list[str] = []
    for filename, tensor_names in split.filename_to_tensors.items():
        shard_state = {name: state_dict[name] for name in tensor_names}
        save_file(shard_state, str(output_dir / filename), metadata=metadata)
        shard_names.append(filename)

    if split.is_sharded:
        index = {
            "metadata": split.metadata,
            "weight_map": split.tensor_to_filename,
        }
        (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return shard_names


def export_nvfp4_checkpoint(
    *,
    full_precision_model_dir: str | Path,
    template_nvfp4_dir: str | Path,
    output_dir: str | Path,
    config: NVFP4EntQuantConfig | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_layers: int = 0,
    max_shard_size: int | str = "5GB",
) -> dict[str, Any]:
    """Export a final NVFP4 checkpoint directory using an NVFP4 template."""
    config = config or NVFP4EntQuantConfig()
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []

    full_precision_model_dir = Path(full_precision_model_dir)
    template_nvfp4_dir = Path(template_nvfp4_dir)
    output_dir = Path(output_dir)
    template_safetensors = template_nvfp4_dir / "model.safetensors"

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    _copy_non_weight_files(template_nvfp4_dir, output_dir)
    prefixes = list_nvfp4_weight_prefixes(
        template_safetensors,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    if max_layers > 0:
        prefixes = prefixes[:max_layers]

    print(
        f"Found {len(prefixes)} NVFP4 weight tensors to export from {full_precision_model_dir}",
        flush=True,
    )

    reports: list[dict[str, Any]] = []
    modified_tensors: dict[str, torch.Tensor] = {}

    with safe_open(str(template_safetensors), framework="pt", device="cpu") as template_handle:
        metadata = template_handle.metadata()
        with ShardedTensorLoader(full_precision_model_dir) as fp_loader:
            for index, prefix in enumerate(prefixes, start=1):
                start_time = time.perf_counter()
                weight = fp_loader.get_tensor(prefix + ".weight").to(torch.float32)
                print(
                    f"[{index}/{len(prefixes)}] exporting {prefix} shape={tuple(weight.shape)} "
                    f"num_weights={weight.numel()}",
                    flush=True,
                )
                optimized = optimize_nvfp4_tensor_entquant(weight, config=config)
                quantized = optimized.quantized
                modified_tensors[prefix + ".weight_packed"] = quantized.packed.cpu()
                modified_tensors[prefix + ".weight_scale"] = quantized.encoded_scale.cpu()
                modified_tensors[prefix + ".weight_global_scale"] = quantized.encoded_global_scale.reshape(1).cpu()
                mse = torch.mean((quantized.dequantized - weight) ** 2).item()
                compression_stats = compute_nvfp4_compression_rate(quantized.packed.cpu().numpy())
                reports.append(
                    {
                        "layer": prefix,
                        "shape": list(weight.shape),
                        "num_weights": int(weight.numel()),
                        "variant": config.variant,
                        "reg_param": config.reg_param,
                        "soft_param": config.soft_param,
                        "block_chunk_size": config.block_chunk_size,
                        "mse": mse,
                        "mean_entropy_bits": compression_stats["mean_entropy_bits"],
                        "compression_rate": compression_stats["compression_rate"],
                        "optimized_scale_mean": optimized.report.optimized_scale_mean,
                        "optimized_scale_std": optimized.report.optimized_scale_std,
                    }
                )
                elapsed_s = time.perf_counter() - start_time
                print(
                    f"[{index}/{len(prefixes)}] done {prefix} in {elapsed_s:.1f}s "
                    f"mse={mse:.6e} "
                    f"entropy={compression_stats['mean_entropy_bits']:.4f}bit "
                    f"comp={compression_stats['compression_rate'] * 100:.2f}% "
                    f"scale_mean={optimized.report.optimized_scale_mean:.6f} "
                    f"scale_std={optimized.report.optimized_scale_std:.6f}",
                    flush=True,
                )
                if str(config.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            state_dict: dict[str, torch.Tensor] = {}
            for key in template_handle.keys():
                state_dict[key] = modified_tensors.get(key, template_handle.get_tensor(key))

    print(f"Saving {len(state_dict)} tensors with max_shard_size={max_shard_size}", flush=True)
    shard_names = _save_sharded_state_dict(
        output_dir=output_dir,
        state_dict=state_dict,
        metadata=metadata,
        max_shard_size=max_shard_size,
    )
    for shard_name in shard_names:
        print(f"Wrote shard: {output_dir / shard_name}", flush=True)

    summary = {
        "full_precision_model_dir": str(full_precision_model_dir),
        "template_nvfp4_dir": str(template_nvfp4_dir),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "num_modified_layers": len(prefixes),
        "aggregate": {
            "weighted_mse": _weighted_average(reports, "mse"),
            "weighted_mean_entropy_bits": _weighted_average(reports, "mean_entropy_bits"),
            "weighted_compression_rate": _weighted_average(reports, "compression_rate"),
        },
        "layers": reports,
        "shards": shard_names,
        "max_shard_size": str(max_shard_size),
    }
    (output_dir / "nvfp4_export_report.json").write_text(json.dumps(summary, indent=2))
    print(f"Export complete -> {output_dir}", flush=True)
    return summary
