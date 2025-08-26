import torch
import torch.nn.functional as F

def _is_soft(targets: torch.Tensor) -> bool:
    return targets.dtype.is_floating_point

def _hard_targets(targets: torch.Tensor) -> torch.Tensor:
    return targets.argmax(dim=-1) if _is_soft(targets) else targets

@torch.enable_grad()
def fgsm_attack(model, images, targets, eps=2/255.0):
    """
    Perform FGSM in *normalized* space.
    Inputs are already normalized by dataset transforms.
    We clamp per-channel to the valid normalized range of [0,1] after Normalize.
    """
    images = images.detach().clone().requires_grad_(True)
    logits = model(images)
    if _is_soft(targets):
        # soft CE
        logp = torch.log_softmax(logits, dim=-1)
        loss = -(targets * logp).sum(dim=-1).mean()
    else:
        loss = F.cross_entropy(logits, targets)

    model.zero_grad(set_to_none=True)
    loss.backward()
    adv = images + eps * images.grad.sign()
    # No absolute clamp to [0,1] because we are in normalized space.
    # Just prevent runaway: clamp to images' min/max range (safe + simple).
    lo = images.amin(dim=(2,3), keepdim=True)
    hi = images.amax(dim=(2,3), keepdim=True)
    adv = torch.max(torch.min(adv, hi), lo)
    return adv.detach()

def fgsm_eval(model, images, hard_targets, eps=2/255.0):
    """
    FGSM for evaluation should use *hard* targets.
    """
    return fgsm_attack(model, images, hard_targets, eps=eps)
