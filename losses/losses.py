import torch
import torch.nn as nn
import torch.nn.functional as F

def is_soft(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, targets):
        # targets: probabilities
        logp = F.log_softmax(logits, dim=-1)
        return -(targets * logp).sum(dim=-1).mean()

def make_losses(mixup_enabled: bool, has_timm_soft_ce: bool):
    """
    Returns (loss_for_hard, loss_for_soft)
    """
    hard = nn.CrossEntropyLoss()
    if mixup_enabled and has_timm_soft_ce:
        try:
            from timm.loss import SoftTargetCrossEntropy as TimmSoftCE
            soft = TimmSoftCE()
        except Exception:
            soft = SoftTargetCrossEntropy()
    else:
        soft = SoftTargetCrossEntropy()
    return hard, soft
