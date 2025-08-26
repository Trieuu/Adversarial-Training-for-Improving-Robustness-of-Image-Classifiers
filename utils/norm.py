import torch

def is_soft_labels(targets: torch.Tensor) -> bool:
    return targets.dtype.is_floating_point

def to_hard(targets: torch.Tensor) -> torch.Tensor:
    return targets.argmax(dim=-1) if is_soft_labels(targets) else targets
