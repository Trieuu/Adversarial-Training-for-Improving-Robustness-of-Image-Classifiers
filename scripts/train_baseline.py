import time
from pathlib import Path
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cifar100_data import CIFAR100Config, get_cifar100_dataloaders, _HAS_TIMM
from models.mobilenetv3 import build_mobilenetv3_small
from losses.losses import make_losses
from utils.seed import set_seed
from engine.trainers import train_epoch
from engine.evaluate import evaluate_clean, evaluate_fgsm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=2/255.0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", type=str, default="./runs_mnv3")
    args = ap.parse_args()

    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = CIFAR100Config()
    train_loader, _, test_loader = get_cifar100_dataloaders(cfg)

    model = build_mobilenetv3_small(num_classes=100, pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = None if args.no_amp or device == "cpu" else torch.amp.GradScaler("cuda")
    use_mixup = (cfg.use_mixup and _HAS_TIMM)
    loss_hard, loss_soft = make_losses(use_mixup, has_timm_soft_ce=_HAS_TIMM)

    for e in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler, loss_hard, loss_soft)
        scheduler.step()
        dt = time.time() - t0
        clean = evaluate_clean(model, test_loader, device)
        adv   = evaluate_fgsm(model, test_loader, device, eps=args.eps)
        print(f"[Epoch {e+1}/{args.epochs}] loss={train_loss:.4f}  "
            f"clean@1={clean['top1']:.2f} clean@5={clean['top5']:.2f}  "
            f"fgsm@1={adv['top1']:.2f} fgsm@5={adv['top5']:.2f}")


    torch.save(model.state_dict(), out / "mobilenetv3_cifar100_baseline.pt")

if __name__ == "__main__":
    main()
