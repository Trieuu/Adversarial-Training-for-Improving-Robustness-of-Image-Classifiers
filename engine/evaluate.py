import torch
from tqdm.auto import tqdm
from .trainers import topk_accuracy
from attacks.fgsm import fgsm_eval
from utils.norm import is_soft_labels, to_hard

@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    for x, y in tqdm(loader, desc="eval/clean", leave=False):
        if is_soft_labels(y):  # guard only; test set should be hard
            y = to_hard(y)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        acc1, acc5 = topk_accuracy(logits, y, topk=(1, 5))
        bs = y.size(0)
        top1_sum += acc1 * bs
        top5_sum += acc5 * bs
        total += bs
    return {"top1": top1_sum / max(1, total), "top5": top5_sum / max(1, total)}

@torch.no_grad()
def evaluate_fgsm(model, loader, device, eps=2/255.0):
    model.eval()
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    for x, y in tqdm(loader, desc=f"eval/fgsm(eps={eps})", leave=False):
        if is_soft_labels(y):
            y = to_hard(y)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = fgsm_eval(model, x, y, eps=eps)
        logits = model(x_adv)
        acc1, acc5 = topk_accuracy(logits, y, topk=(1, 5))
        bs = y.size(0)
        top1_sum += acc1 * bs
        top5_sum += acc5 * bs
        total += bs
    return {"top1": top1_sum / max(1, total), "top5": top5_sum / max(1, total)}
