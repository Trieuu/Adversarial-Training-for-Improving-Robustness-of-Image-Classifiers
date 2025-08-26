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
from engine.trainers import train_epoch_adv
from engine.evaluate import evaluate_clean, evaluate_fgsm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_adv", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=2/255.0)
    ap.add_argument("--adv_ratio", type=float, default=0.5)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", type=str, default="./runs_mnv3")
    ap.add_argument("--init", type=str, default="imagenet",
                    choices=["imagenet", "baseline", "scratch"],
                    help="Initialization: imagenet | baseline | scratch")
    args = ap.parse_args()

    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = CIFAR100Config()
    train_loader, _, test_loader = get_cifar100_dataloaders(cfg)

    # --- Model initialization
    if args.init == "imagenet":
        model = build_mobilenetv3_small(num_classes=100, pretrained=True).to(device)
        print("Init from ImageNet-pretrained weights.")
    elif args.init == "scratch":
        model = build_mobilenetv3_small(num_classes=100, pretrained=False).to(device)
        print("Init from scratch.")
    elif args.init == "baseline":
        model = build_mobilenetv3_small(num_classes=100, pretrained=True).to(device)
        baseline_path = out / "mobilenetv3_cifar100_baseline.pt"
        if baseline_path.exists():
            state = torch.load(baseline_path, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"Loaded baseline checkpoint from {baseline_path}")
        else:
            print("Baseline not found → falling back to ImageNet-pretrained init.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_adv)

    scaler = None if args.no_amp or device == "cpu" else torch.amp.GradScaler("cuda")
    use_mixup = (cfg.use_mixup and _HAS_TIMM)
    loss_hard, loss_soft = make_losses(use_mixup, has_timm_soft_ce=_HAS_TIMM)

    # --- Training loop
    for e in range(args.epochs_adv):
        t0 = time.time()
        train_loss = train_epoch_adv(
            model, train_loader, optimizer, device, scaler,
            loss_hard=loss_hard, loss_soft=loss_soft,
            eps=args.eps, adv_ratio=args.adv_ratio
        )
        scheduler.step()
        dt = time.time() - t0
        clean = evaluate_clean(model, test_loader, device)
        adv   = evaluate_fgsm(model, test_loader, device, eps=args.eps)
        print(f"[Adv Epoch {e+1}/{args.epochs_adv}] loss={train_loss:.4f}  "
              f"clean@1={clean['top1']:.2f} clean@5={clean['top5']:.2f}  "
              f"fgsm@1={adv['top1']:.2f} fgsm@5={adv['top5']:.2f}")

    # Save
    torch.save(model.state_dict(), out / "mobilenetv3_cifar100_advtrained.pt")

    # --- Optional comparison (only if baseline exists)
    baseline_path = out / "mobilenetv3_cifar100_baseline.pt"
    if baseline_path.exists():
        base = build_mobilenetv3_small(num_classes=100, pretrained=True).to(device)
        base.load_state_dict(torch.load(baseline_path, map_location=device))
        base_clean = evaluate_clean(base, test_loader, device)
        base_adv   = evaluate_fgsm(base, test_loader, device)
        final_clean = evaluate_clean(model, test_loader, device)
        final_adv   = evaluate_fgsm(model, test_loader, device)
        print("\n=== Results (Top-1/Top-5 %) ===")
        print("Model                      Clean@1 Clean@5 FGSM@1 FGSM@5")
        print(f"MobileNetV3 Baseline       {base_clean['top1']:7.2f} {base_clean['top5']:7.2f} "
              f"{base_adv['top1']:7.2f} {base_adv['top5']:7.2f}")
        print(f"MobileNetV3 Adv.Trained    {final_clean['top1']:7.2f} {final_clean['top5']:7.2f} "
              f"{final_adv['top1']:7.2f} {final_adv['top5']:7.2f}")


if __name__ == "__main__":
    main()
