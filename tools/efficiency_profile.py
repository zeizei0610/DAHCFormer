import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from args import Args as args
from GIFNet_model import TwoBranchesFusionNet, WindowAttention


class DeploymentFusionPath(nn.Module):
    def __init__(self, model: TwoBranchesFusionNet, training_tag: int = 2):
        super().__init__()
        self.model = model
        self.training_tag = training_tag

    def forward(self, x, y):
        fea_com = self.model.forward_encoder(x, y)
        fea_fused = self.model.forward_MultiTask_branch(
            fea_com_ivif=fea_com,
            fea_com_mfif=fea_com,
            trainingTag=self.training_tag,
        )
        return self.model.forward_mixed_decoder(fea_com, fea_fused)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_attention_matmuls(inputs):
    """Two streams, each with QK and AV; report multiply-accumulates."""
    qkv = inputs[0]
    windows, tokens, three_channels = qkv.shape
    channels = three_channels // 3
    return 4 * windows * tokens * tokens * channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--gate", action="store_true", help="enable HCFGM/GateNet")
    parser.add_argument("--training-tag", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args()

    torch.manual_seed(parsed.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    model = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride, parsed.gate)
    model.eval().to(device)
    deploy_model = DeploymentFusionPath(model, training_tag=parsed.training_tag).eval().to(device)

    x = torch.randn(1, 1, parsed.height, parsed.width, device=device)
    y = torch.randn(1, 1, parsed.height, parsed.width, device=device)

    params = sum(p.numel() for p in model.parameters())
    model_size_decimal_mb = params * 4 / 1e6
    model_size_mib = params * 4 / (1024 ** 2)

    rec_params = sum(p.numel() for p in model.decoder_rec.parameters())
    attention_macs = [0]

    def record_attention(module, inputs, output):
        attention_macs[0] += count_attention_matmuls(inputs)

    hooks = [m.register_forward_hook(record_attention) for m in model.modules()
             if isinstance(m, WindowAttention)]
    with torch.no_grad():
        # A THOP parent custom hook would hide the child position-bias MLP.
        # Count matmuls separately and let THOP visit all supported children.
        try:
            module_macs, _ = profile(deploy_model, inputs=(x, y), verbose=False)
        finally:
            for hook in hooks:
                hook.remove()
    macs = module_macs + attention_macs[0]

    with torch.no_grad():
        for _ in range(parsed.warmup):
            _ = deploy_model(x, y)
        synchronize(device)

        allocated_before = None
        peak_allocated = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
            _ = deploy_model(x, y)
            synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)

        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(parsed.repeat):
                _ = deploy_model(x, y)
            end_event.record()
            torch.cuda.synchronize(device)
            latency_ms = start_event.elapsed_time(end_event) / parsed.repeat
        else:
            start = time.perf_counter()
            for _ in range(parsed.repeat):
                _ = deploy_model(x, y)
            synchronize(device)
            elapsed = time.perf_counter() - start
            latency_ms = elapsed / parsed.repeat * 1000

    fps = 1000.0 / latency_ms

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"input: 1x1x{parsed.height}x{parsed.width}")
    print(f"gate_enabled: {parsed.gate}")
    print(f"training_tag: {parsed.training_tag}")
    print(f"parameters_M: {params / 1e6:.4f}")
    print(f"model_size_decimal_MB: {model_size_decimal_mb:.4f}")
    print(f"model_size_MiB: {model_size_mib:.4f}")
    print(f"Estimated GMACs (including window attention): {macs / 1e9:.4f}")
    print(f"latency_ms: {latency_ms:.4f}")
    print(f"FPS: {fps:.2f}")
    record = {
        "measurement": "static student inference graph; no quality evaluation",
        "weights": "seeded random initialization; no trained checkpoint loaded",
        "seed": parsed.seed,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_shape_each": [1, 1, parsed.height, parsed.width],
        "dtype": "float32", "gate": parsed.gate, "tag": parsed.training_tag,
        "warmup": parsed.warmup, "repetitions": parsed.repeat,
        "student_parameters": params,
        "parameters_excluding_reconstruction_decoder": params - rec_params,
        "parameter_storage_decimal_mb": model_size_decimal_mb,
        "estimated_gmacs": macs / 1e9,
        "profiled_module_gmacs": module_macs / 1e9,
        "attention_matmul_gmacs": attention_macs[0] / 1e9,
        "mac_counting": "THOP modules plus QK/AV matmuls in both attention streams; not an exact all-operation FLOP count",
        "latency_ms": latency_ms, "fps": fps,
        "allocated_before_mib": None if allocated_before is None else allocated_before / 2**20,
        "peak_allocated_mib": None if peak_allocated is None else peak_allocated / 2**20,
        "scope": "model forward only; excludes preprocessing, transfers, postprocessing and teacher training",
    }
    if parsed.output:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
