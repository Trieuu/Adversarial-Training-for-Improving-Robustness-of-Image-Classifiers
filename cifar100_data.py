from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# Optional (recommended) for MixUp/CutMix:
try:
    from timm.data import Mixup
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

# Optional: PyTorch Lightning support
try:
    import pytorch_lightning as pl
    _HAS_PL = True
except Exception:
    _HAS_PL = False


# -------------------------
# Config
# -------------------------
@dataclass
class CIFAR100Config:
    data_dir: str = "./data"
    img_size: int = 224             # Always resize to 224 as requested
    batch_size: int = 256
    num_workers: int = 2            # 2 on Colab, 4 for Kaggle P100
    val_split: float = 0.0          # set >0 to split train into train/val
    seed: int = 1337
    # Augmentations
    randaugment: bool = True
    randaugment_N: int = 2
    randaugment_M: int = 9
    random_erasing_p: float = 0.25
    # MixUp/CutMix (requires timm)
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 1.0
    # Norm: use ImageNet stats to match ImageNet-pretrained backbones
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Dataloader niceties
    pin_memory: bool = True
    persistent_workers: bool = True
    # Evaluation
    drop_last_train: bool = True


# -------------------------
# Worker seeding (repro)
# -------------------------
def _seed_worker(worker_id: int):
    # Ensures each dataloader worker is deterministically seeded
    worker_seed = torch.initial_seed() % 2**32
    import random, numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# -------------------------
# Transforms
# -------------------------
def build_transforms(cfg: CIFAR100Config):
    # Always upsample to 224; use bicubic for quality
    train_tfms = [
        transforms.Resize(cfg.img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if cfg.randaugment:
        # torchvision RandAugment works well; N=2, M=9 default is stable
        train_tfms.append(transforms.RandAugment(num_ops=cfg.randaugment_N, magnitude=cfg.randaugment_M))

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])

    # RandomErasing happens after normalization
    if cfg.random_erasing_p > 0:
        train_tfms.append(transforms.RandomErasing(p=cfg.random_erasing_p, value='random'))

    train_tfms = transforms.Compose(train_tfms)

    # Eval: strict resize + center crop (optional); CIFAR is small, so just Resize is fine.
    eval_tfms = transforms.Compose([
        transforms.Resize(cfg.img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])

    return train_tfms, eval_tfms


# -------------------------
# Collate (MixUp/CutMix)
# -------------------------
class _IdentityCollate:
    def __call__(self, batch):
        # Default collate from PyTorch
        return torch.utils.data.default_collate(batch)


def build_collate_fn(cfg: CIFAR100Config):
    if cfg.use_mixup and _HAS_TIMM:
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            mode="batch",          # batch mode = robust and simple
            label_smoothing=0.0,
            num_classes=100,
        )
        # Wrap a collate that applies mixup on the fly
        def _collate(batch):
            inputs, targets = torch.utils.data.default_collate(batch)
            inputs, targets = mixup_fn(inputs, targets)
            return inputs, targets
        return _collate
    else:
        return _IdentityCollate()


# -------------------------
# Plain PyTorch helpers
# -------------------------
def get_cifar100_dataloaders(cfg: CIFAR100Config):
    train_tfms, eval_tfms = build_transforms(cfg)

    full_train = datasets.CIFAR100(root=cfg.data_dir, train=True, download=True, transform=train_tfms)
    test_set = datasets.CIFAR100(root=cfg.data_dir, train=False, download=True, transform=eval_tfms)

    # Optional manual val split from training set
    if cfg.val_split and cfg.val_split > 0.0:
        n_total = len(full_train)  # 50_000
        n_val = int(n_total * cfg.val_split)
        n_train = n_total - n_val
        g = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = torch.utils.data.random_split(full_train, [n_train, n_val], generator=g)
    else:
        train_set = full_train
        # If no explicit val split, evaluate on test_set and/or create a small hold‑out:
        val_set = None

    collate_fn = build_collate_fn(cfg)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last_train,
        persistent_workers=cfg.persistent_workers,
        worker_init_fn=_seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            persistent_workers=cfg.persistent_workers,
            worker_init_fn=_seed_worker,
        )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=cfg.persistent_workers,
        worker_init_fn=_seed_worker,
    )

    return train_loader, val_loader, test_loader


# -------------------------
# Lightning DataModule (optional)
# -------------------------
if _HAS_PL:
    class CIFAR100DataModule(pl.LightningDataModule):
        def __init__(self, cfg: CIFAR100Config):
            super().__init__()
            self.cfg = cfg
            self.train_set = None
            self.val_set = None
            self.test_set = None
            self._collate = build_collate_fn(cfg)

        def prepare_data(self):
            # Download once
            datasets.CIFAR100(root=self.cfg.data_dir, train=True, download=True)
            datasets.CIFAR100(root=self.cfg.data_dir, train=False, download=True)

        def setup(self, stage: Optional[str] = None):
            train_tfms, eval_tfms = build_transforms(self.cfg)
            full_train = datasets.CIFAR100(root=self.cfg.data_dir, train=True, download=False, transform=train_tfms)
            self.test_set = datasets.CIFAR100(root=self.cfg.data_dir, train=False, download=False, transform=eval_tfms)

            if self.cfg.val_split and self.cfg.val_split > 0.0:
                n_total = len(full_train)
                n_val = int(n_total * self.cfg.val_split)
                n_train = n_total - n_val
                g = torch.Generator().manual_seed(self.cfg.seed)
                self.train_set, self.val_set = torch.utils.data.random_split(full_train, [n_train, n_val], generator=g)
            else:
                self.train_set = full_train
                self.val_set = None

        def train_dataloader(self):
            g = torch.Generator().manual_seed(self.cfg.seed)
            return DataLoader(
                self.train_set,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=self.cfg.drop_last_train,
                persistent_workers=self.cfg.persistent_workers,
                worker_init_fn=_seed_worker,
                generator=g,
                collate_fn=self._collate,
            )

        def val_dataloader(self):
            if self.val_set is None:
                return None
            return DataLoader(
                self.val_set,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=False,
                persistent_workers=self.cfg.persistent_workers,
                worker_init_fn=_seed_worker,
            )

        def test_dataloader(self):
            return DataLoader(
                self.test_set,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=False,
                persistent_workers=self.cfg.persistent_workers,
                worker_init_fn=_seed_worker,
            )
