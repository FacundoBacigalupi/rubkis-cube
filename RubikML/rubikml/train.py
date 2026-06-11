"""
python -m rubikml.train --min-depth 1 --max-depth 3 --epochs 20
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import RubikScrambleDataset
from .model import build_model


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = _resolve_device(cfg.device)
    print(f"device: {device}  model: {cfg.model_type}  depths: {cfg.min_depth}-{cfg.max_depth}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model = build_model(cfg.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        # only restore optimizer if depths/lr match, otherwise start fresh optimizer
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            pass
        print(f"Resumed from {cfg.resume} (epoch {ckpt.get('epoch', '?')})")
    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    train_ds = RubikScrambleDataset(cfg.min_depth, cfg.max_depth, cfg.samples_per_epoch,
                                    value_norm=cfg.value_norm)
    val_ds   = RubikScrambleDataset(cfg.min_depth, cfg.max_depth, cfg.val_samples,
                                    seed=cfg.val_seed, value_norm=cfg.value_norm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    history = []
    best_val_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = total_acc = n_batches = 0

        for batch in train_loader:
            x, y, v = batch
            x, y, v = x.to(device), y.to(device), v.to(device)
            optimizer.zero_grad()

            if cfg.model_type in ("policy_value", "policy_value_v2"):
                policy_logits, value_pred = model(x)
                policy_loss = policy_criterion(policy_logits, y)
                value_loss = value_criterion(value_pred, v)
                loss = policy_loss + cfg.value_weight * value_loss
            else:
                policy_logits = model(x)
                loss = policy_criterion(policy_logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += _accuracy(policy_logits, y)
            n_batches += 1

        train_loss = total_loss / n_batches
        train_acc = total_acc / n_batches

        # validation
        model.eval()
        val_loss = val_acc = v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, v_target = batch
                x, y, v_target = x.to(device), y.to(device), v_target.to(device)
                if cfg.model_type in ("policy_value", "policy_value_v2"):
                    policy_logits, value_pred = model(x)
                    val_loss += (policy_criterion(policy_logits, y)
                                 + cfg.value_weight * value_criterion(value_pred, v_target)).item()
                else:
                    policy_logits = model(x)
                    val_loss += policy_criterion(policy_logits, y).item()
                val_acc += _accuracy(policy_logits, y)
                v_batches += 1
        val_loss /= v_batches
        val_acc /= v_batches

        elapsed = time.time() - t0
        print(
            f"epoch {epoch:3d}/{cfg.epochs}  "
            f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"({elapsed:.1f}s)"
        )

        row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc)
        history.append(row)

        # save latest
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.__dict__,
        }
        torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "latest.pt"))

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "best.pt"))

    # save run history
    os.makedirs("runs", exist_ok=True)
    run_name = f"train_{cfg.model_type}_d{cfg.min_depth}-{cfg.max_depth}_{int(time.time())}"
    with open(os.path.join("runs", run_name + ".json"), "w") as f:
        json.dump({"config": cfg.__dict__, "history": history}, f, indent=2)
    print(f"Run saved to runs/{run_name}.json")
    print(f"Best val accuracy: {best_val_acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train RubikML policy network")
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--val-samples", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-type", type=str, default="policy",
                        choices=["policy", "policy_value", "policy_value_v2"])
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--value-norm", type=int, default=26,
                        help="Fixed denominator for value targets (depth/value_norm). "
                             "Never change this between curriculum phases.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/best.pt)")
    args = parser.parse_args()

    cfg = TrainConfig(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device=args.device,
        model_type=args.model_type,
        value_weight=args.value_weight,
        value_norm=args.value_norm,
        resume=args.resume,
    )
    train(cfg)


if __name__ == "__main__":
    main()
