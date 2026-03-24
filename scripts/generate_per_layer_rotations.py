#!/usr/bin/env python3
"""Generate per-layer Q rotation files for sensitivity analysis.

For each target layer L, creates a rotation file where:
  - Layer L has the real rotation matrix
  - All other layers have identity (no-op rotation)

This lets us measure how much each individual layer's QR contributes.
"""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-rotation-path",
        type=str,
        default="/data/shared/charlie/sglangfork/q_rotation_layer_second_moment_damp01.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/shared/charlie/sglangfork/per_layer_rotations",
    )
    args = parser.parse_args()

    state = torch.load(args.full_rotation_path, map_location="cpu")
    meta_keys = {k: v for k, v in state.items() if k != "layers"}

    os.makedirs(args.output_dir, exist_ok=True)

    for target_layer in sorted(state["layers"].keys()):
        single = dict(meta_keys)
        single["layers"] = {target_layer: state["layers"][target_layer]}
        out_path = os.path.join(args.output_dir, f"q_rotation_layer{target_layer}.pt")
        torch.save(single, out_path)
        print(f"Layer {target_layer:2d} -> {out_path}")

    print(f"\nDone. {len(state['layers'])} files in {args.output_dir}")


if __name__ == "__main__":
    main()
