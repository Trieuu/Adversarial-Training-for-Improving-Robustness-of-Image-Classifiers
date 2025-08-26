import torch
from tqdm.auto import tqdm
from utils.norm import is_soft_labels, to_hard

def topk_accuracy(logits, targets, topk=(1,)):
    """Return list of top‑k accuracies in % for the given k values."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        res.append(100.0 * correct_k / targets.size(0))
    return res

def train_epoch(model, loader, optimizer, device, scaler, loss_hard, loss_soft):
    """Standard epoch with tqdm. Shows running loss."""
    model.train()
    total_loss, total = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            logits = model(x)
            loss = (loss_soft if is_soft_labels(y) else loss_hard)(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total += bs
        pbar.set_postfix(loss=f"{total_loss/max(1,total):.4f}")
    return total_loss / max(1, total)

def train_epoch_adv(model, loader, optimizer, device, scaler,
                    loss_hard, loss_soft, eps=2/255.0, adv_ratio=0.5):
    """
    Mix clean + FGSM per batch (shown with tqdm).
    """
    from attacks.fgsm import fgsm_attack

    model.train()
    total_loss, total = 0.0, 0
    pbar = tqdm(loader, desc="adv-train", leave=False)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Build adversarial copy using hard labels
        y_hard = to_hard(y)
        x_adv = fgsm_attack(model, x, y_hard, eps=eps)

        if adv_ratio <= 0.0:
            mix_x, mix_y = x, y
        elif adv_ratio >= 1.0:
            mix_x, mix_y = x_adv, y
        else:
            n = x.size(0)
            n_adv = int(round(n * adv_ratio))
            mix_x = torch.cat([x_adv[:n_adv], x[n_adv:]], dim=0)
            mix_y = torch.cat([y[:n_adv], y[n_adv:]], dim=0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            logits = model(mix_x)
            loss = (loss_soft if is_soft_labels(mix_y) else loss_hard)(logits, mix_y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = mix_x.size(0)
        total_loss += loss.item() * bs
        total += bs
        pbar.set_postfix(loss=f"{total_loss/max(1,total):.4f}")

    return total_loss / max(1, total)
