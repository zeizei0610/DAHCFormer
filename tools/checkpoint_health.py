"""Check whether selected student checkpoints produce non-degenerate outputs.

This is a diagnostic for completed training runs. It does not select a benchmark
checkpoint and it does not write publication results.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import TwoBranchesFusionNet


def read_gray(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def evaluate(checkpoint, pairs, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("student", payload) if isinstance(payload, dict) else payload
    has_gate = any(".gate_net." in name for name in state)
    model = TwoBranchesFusionNet(3, 64, 1, 1, has_gate)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    rows = []
    with torch.no_grad():
        for left, right in pairs:
            a = torch.from_numpy(read_gray(left).astype(np.float32) / 255.0)[None, None].to(device)
            b = torch.from_numpy(read_gray(right).astype(np.float32) / 255.0)[None, None].to(device)
            z = model.forward_encoder(a, b)
            u = model.forward_MultiTask_branch(z, z, trainingTag=2)
            fused = model.forward_mixed_decoder(z, u)[0, 0].detach().cpu().numpy()
            rows.append({
                "min": float(np.nanmin(fused)),
                "max": float(np.nanmax(fused)),
                "mean": float(np.nanmean(fused)),
                "finite": bool(np.isfinite(fused).all()),
                "nonconstant": bool(float(np.nanmax(fused)) > float(np.nanmin(fused))),
            })
    return {
        "checkpoint": str(checkpoint),
        "gate": has_gate,
        "device": device,
        "pairs": rows,
        "all_finite": all(row["finite"] for row in rows),
        "all_nonconstant": all(row["nonconstant"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint filename within --run")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    run = args.run.resolve()
    pairs = [(args.left.resolve(), args.right.resolve())]
    results = []
    for name in args.checkpoint:
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(path)
        results.append(evaluate(path, pairs, args.device))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
