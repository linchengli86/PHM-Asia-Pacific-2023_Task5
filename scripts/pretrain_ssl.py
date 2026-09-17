"""SSL pretraining entry point.

Usage:
    python scripts/pretrain_ssl.py --config configs/default.yaml --method simclr
    python scripts/pretrain_ssl.py --config configs/default.yaml --method mae
    python scripts/pretrain_ssl.py --config configs/default.yaml --method dual
"""

import sys, os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import load_raw_data, normalize_series
from src.data.augment import create_training_dataset
from src.data.labels import generate_labels
from src.models.encoder import build_encoder
from src.pretrain import SimCLRPretrainer, MAEPretrainer, DualSSLPretrainer
from src.utils.config import load_config

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="simclr", choices=["simclr", "mae", "dual"])
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch (if --resume)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"=== SSL Pretraining: {args.method} ===")

    # Load all data (train + test, NO labels)
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_ids = list(range(data_cfg["train_cases"][0], data_cfg["train_cases"][1] + 1))
    test_ids = list(range(data_cfg["test_cases"][0], data_cfg["test_cases"][1] + 1))

    X_train, X_test = load_raw_data(paths["data_root"], train_ids, test_ids)
    X_train_n, X_test_n = normalize_series(X_train, X_test)

    # Use windowed training data if available (much richer for SSL)
    use_windows = cfg["pretrain"].get("use_windowed_data", False)
    if use_windows:
        from src.data.labels import generate_labels
        from src.data.augment import create_training_dataset
        y_train = generate_labels(paths["train_labels_path"])
        X_ssl, _, _, _ = create_training_dataset(X_train_n, y_train, cfg)
        # Also include test data windows (no labels)
        X_test_win = X_test_n[:, ::6, :][:, :200, :]
        X_all = np.concatenate([X_ssl, X_test_win], axis=0).astype(np.float32)
        print(f"SSL data: {X_all.shape[0]} windows (train windows + test) × {X_all.shape[1]} steps")
    else:
        if cfg["pretrain"].get("use_all_data", True):
            X_all = np.concatenate([X_train_n, X_test_n], axis=0)
        else:
            X_all = X_train_n
        X_all = X_all[:, ::6, :][:, :200, :].astype(np.float32)
        print(f"SSL data: {X_all.shape[0]} sequences × {X_all.shape[1]} steps")

    # Build encoder
    encoder = build_encoder(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build pretrainer
    if args.method == "simclr":
        trainer = SimCLRPretrainer(encoder, **{k: v for k, v in cfg["pretrain"]["simclr"].items()
                                                if k != "augment_strength"})
    elif args.method == "mae":
        trainer = MAEPretrainer(encoder, cfg["data"]["n_params"])
    elif args.method == "dual":
        trainer = DualSSLPretrainer(encoder, cfg)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    trainer.to(device)
    params = trainer.parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg["pretrain"]["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=cfg["pretrain"]["epochs"])

    dataset = TensorDataset(torch.from_numpy(X_all))
    loader = DataLoader(dataset, batch_size=cfg["pretrain"]["batch_size"],
                        shuffle=True, drop_last=True)

    best_loss = float("inf")
    start_epoch = 0
    loss_history = []

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", args.start_epoch)
        best_loss = ckpt.get("best_loss", float("inf"))
        loss_history = ckpt.get("loss_history", [])
        # Advance scheduler to correct position
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    ckpt_dir = os.path.join(paths["output_root"], "models", "ssl_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg["pretrain"]["epochs"]):
        trainer.train()
        ep_loss, n = 0, 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            loss_dict = trainer.compute_loss(x_batch)

            if isinstance(loss_dict, dict):
                loss = loss_dict["total"]
            else:
                loss = loss_dict

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = ep_loss / max(n, 1)
        loss_history.append(avg_loss)

        # Always save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(),
                       os.path.join(paths["output_root"], "models", f"ssl_{args.method}_best.pt"))

        # Checkpoint every 10 epochs + first 5 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) <= 5:
            torch.save({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "best_loss": best_loss,
                "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_history": loss_history,
            }, os.path.join(ckpt_dir, f"ssl_{args.method}_ep{epoch+1:03d}.pt"))

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            comps = ""
            if isinstance(loss_dict, dict):
                comps = " | ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            print(f"  Epoch {epoch+1}/{cfg['pretrain']['epochs']}: loss={avg_loss:.4f} | {comps}")

    # Save final + loss curve
    torch.save(encoder.state_dict(),
               os.path.join(paths["output_root"], "models", f"ssl_{args.method}_final.pt"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"SSL Pretraining Loss ({args.method})"); ax.grid(alpha=0.3)
    plt.savefig(os.path.join(paths["output_root"], "models", f"ssl_{args.method}_loss.png"), dpi=100)
    plt.close()

    print(f"\nSSL pretraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
