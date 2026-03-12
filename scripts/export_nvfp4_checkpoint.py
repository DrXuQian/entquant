"""Export a final NVFP4 checkpoint directory using the experimental NVFP4 optimizer."""

from __future__ import annotations

import argparse

from entquant.quantization.nvfp4_export import export_nvfp4_checkpoint
from entquant.quantization.nvfp4_optimizer import NVFP4EntQuantConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-precision-model-dir", required=True)
    parser.add_argument("--template-nvfp4-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--variant", choices=("entquant_exact", "entquant_soft"), default="entquant_exact")
    parser.add_argument("--reg-param", type=float, default=3.9)
    parser.add_argument("--soft-param", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--block-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm-type", default="relative")
    parser.add_argument("--norm-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_nvfp4_checkpoint(
        full_precision_model_dir=args.full_precision_model_dir,
        template_nvfp4_dir=args.template_nvfp4_dir,
        output_dir=args.output_dir,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_layers=args.max_layers,
        max_shard_size=args.max_shard_size,
        config=NVFP4EntQuantConfig(
            variant=args.variant,
            norm_type=args.norm_type,
            norm_p=args.norm_p,
            reg_param=args.reg_param,
            soft_param=args.soft_param,
            temperature=args.temperature,
            lr=args.lr,
            max_iters=args.max_iters,
            block_chunk_size=args.block_chunk_size,
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()
