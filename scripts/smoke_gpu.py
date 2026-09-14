"""Verify real CUDA execution, gradients and optimizer state (not device detection)."""
import json, platform, subprocess, sys, time
from pathlib import Path
import torch
from fsd.runtime import atomic_json, record_stage, seed_all


def main():
    seed_all(42)
    assert torch.cuda.is_available(), 'CUDA is required'
    device = torch.device('cuda')
    conv = torch.nn.Conv2d(3, 32, 3, padding=1).to(device)
    attention = torch.nn.MultiheadAttention(32, 4, batch_first=True).to(device)
    optimizer = torch.optim.AdamW(list(conv.parameters()) + list(attention.parameters()), lr=1e-3)
    before = conv.weight.detach().clone()
    scaler = torch.amp.GradScaler('cuda')
    start = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            features = conv(torch.randn(2, 3, 16, 16, device=device)).flatten(2).transpose(1, 2)
            y, _ = attention(features, features, features, need_weights=False)
            loss = y.float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in conv.parameters())
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    delta = (conv.weight - before).abs().max().item()
    assert delta > 0 and torch.isfinite(loss)
    report = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                  cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  compute_capability=list(torch.cuda.get_device_capability()),
                  architectures=torch.cuda.get_arch_list(), parameter_delta=delta,
                  final_loss=loss.item(), elapsed_seconds=time.perf_counter()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  checks=['convolution','attention','mixed_precision','backward','optimizer_update'])
    atomic_json('artifacts/environment/gpu_smoke.json', report)
    record_stage('01_environment', 'passed', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
