import argparse
import pathlib
import sys
import time
import warnings

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from args import Args as args
from GIFNet_model import TwoBranchesFusionNet

try:
    from thop import profile
except ImportError:  # pragma: no cover - depends on local environment
    profile = None


class FusionInferenceWrapper(torch.nn.Module):
    """Wrap the actual test-time path used by test.py."""

    def __init__(self, gate: bool):
        super().__init__()
        self.model = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride, gate)

    def forward(self, x, y):
        fea_com = self.model.forward_encoder(x, y)
        fea_fused = self.model.forward_MultiTask_branch(
            fea_com_ivif=fea_com,
            fea_com_mfif=fea_com,
            trainingTag=2,
        )
        return self.model.forward_mixed_decoder(fea_com, fea_fused)


def measure_latency(model, x, y, warmup, runs, use_cuda):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x, y)
        if use_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(runs):
                model(x, y)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / runs

        start = time.perf_counter()
        for _ in range(runs):
            model(x, y)
        return (time.perf_counter() - start) * 1000.0 / runs


def profile_model(gate, size, runs, warmup):
    model = FusionInferenceWrapper(gate)
    x = torch.randn(1, args.channel, size, size)
    y = torch.randn(1, args.channel, size, size)

    params = sum(p.numel() for p in model.parameters())
    macs = None
    if profile is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs, _ = profile(model, inputs=(x, y), verbose=False)

    cpu_ms = measure_latency(model, x, y, warmup, runs, use_cuda=False)

    gpu_ms = None
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda")
        gpu_model = FusionInferenceWrapper(gate).to(device)
        xg = torch.randn(1, args.channel, size, size, device=device)
        yg = torch.randn(1, args.channel, size, size, device=device)
        gpu_ms = measure_latency(gpu_model, xg, yg, warmup, runs, use_cuda=True)

    return {
        "gate": gate,
        "params_m": params / 1e6,
        "macs_g": None if macs is None else macs / 1e9,
        "flops_g": None if macs is None else 2 * macs / 1e9,
        "cpu_ms": cpu_ms,
        "cpu_fps": 1000.0 / cpu_ms,
        "gpu_ms": gpu_ms,
        "gpu_fps": None if gpu_ms is None else 1000.0 / gpu_ms,
        "gpu_name": gpu_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    opts = parser.parse_args()

    print("model,params(M),MACs(G),FLOPs(G),CPU ms,CPU FPS,GPU ms,GPU FPS,GPU")
    for gate in (False, True):
        row = profile_model(gate, opts.size, opts.runs, opts.warmup)
        print(
            "{model},{params:.4f},{macs},{flops},{cpu_ms:.2f},{cpu_fps:.2f},{gpu_ms},{gpu_fps},{gpu}".format(
                model="Base inference path (w/o HCFGM)" if not gate else "DAHCFormer inference path (w/ HCFGM)",
                params=row["params_m"],
                macs="NA" if row["macs_g"] is None else f"{row['macs_g']:.4f}",
                flops="NA" if row["flops_g"] is None else f"{row['flops_g']:.4f}",
                cpu_ms=row["cpu_ms"],
                cpu_fps=row["cpu_fps"],
                gpu_ms="NA" if row["gpu_ms"] is None else f"{row['gpu_ms']:.3f}",
                gpu_fps="NA" if row["gpu_fps"] is None else f"{row['gpu_fps']:.2f}",
                gpu="NA" if row["gpu_name"] is None else row["gpu_name"],
            )
        )


if __name__ == "__main__":
    main()
