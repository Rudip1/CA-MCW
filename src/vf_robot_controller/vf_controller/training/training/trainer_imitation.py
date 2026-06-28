"""
trainer_imitation.py — IMITATION training loop (behaviour cloning of MPPI).

Loss = MSE(vx) + MSE(wz)

Same dataset / split / checkpoint policy as trainer.py. Phase 8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .losses import velocity_mse_loss


@dataclass
class ImitationTrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"
    log_every: int = 50
    early_stop_patience: int = 5


@dataclass
class ImitationTrainResult:
    best_val_loss: float = float("inf")
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = field(default_factory=list)


def train_imitation(
    model: torch.nn.Module,
    train_ds: Dataset,
    val_ds: Optional[Dataset],
    cfg: ImitationTrainConfig,
) -> ImitationTrainResult:
    device = torch.device(cfg.device)
    model = model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        if val_ds is not None and len(val_ds) > 0 else None)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    result = ImitationTrainResult()
    epochs_since_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_total = 0.0
        train_n = 0
        last_components: Dict[str, float] = {}
        for batch in train_loader:
            x, y, _mask = batch
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            loss, comps = velocity_mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_total += float(loss.detach().cpu()) * x.shape[0]
            train_n += x.shape[0]
            last_components = comps

        train_loss = train_total / max(1, train_n)
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vt = 0.0
                vn = 0
                for batch in val_loader:
                    x, y, _mask = batch
                    x = x.to(device).float()
                    y = y.to(device).float()
                    pred = model(x)
                    loss, _ = velocity_mse_loss(pred, y)
                    vt += float(loss) * x.shape[0]
                    vn += x.shape[0]
                val_loss = vt / max(1, vn)
            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                result.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
        else:
            result.best_val_loss = train_loss
            result.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  **last_components}
        result.history.append(record)
        print(f"[train_imitation] epoch={epoch} train={train_loss:.5f} "
              f"val={val_loss:.5f} vx_mse={last_components.get('vx_mse', 0):.4f} "
              f"wz_mse={last_components.get('wz_mse', 0):.4f}")

        if (val_loader is not None
                and epochs_since_improve >= cfg.early_stop_patience):
            print(f"[train_imitation] early stop @ epoch {epoch}")
            break

    if result.best_state_dict is None:
        result.best_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    return result
