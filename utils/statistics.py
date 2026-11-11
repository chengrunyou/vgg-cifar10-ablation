"""参数量与 FLOPs 统计。"""

import torch

__all__ = ["count_params", "count_flops"]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_res=32):
    orig_device = next(model.parameters()).device
    model = model.cpu()
    model.eval()
    x = torch.randn(1, 3, input_res, input_res)
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        model(x)
    flops = sum(e.flops for e in prof.events() if e.flops is not None)
    model.to(orig_device)
    return flops
