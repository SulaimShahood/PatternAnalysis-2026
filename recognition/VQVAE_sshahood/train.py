"""
train.py

Trains either the VQ-VAE or the plain-autoencoder baseline on HipMRI 2D
prostate slices, using one identical loop for both (selected with
--model), so the comparison in the README is apples-to-apples. Logs loss,
reconstruction error, codebook perplexity and validation SSIM to a CSV
every epoch, saves the best checkpoint by validation SSIM, and plots the
training curves at the end.

Usage:
    python train.py --model vqvae --smoke        # fast end-to-end check
    python train.py --model vqvae                # real run
    python train.py --model autoencoder          # baseline, same settings
"""

import os
import csv
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # no display on Rangpur compute nodes
import matplotlib.pyplot as plt

from modules import VQVAE, Autoencoder
from dataset import HipMRISlices
from utils import batch_ssim

MODELS = {"vqvae": VQVAE, "autoencoder": Autoencoder}


def build_argparser():
    """Defines every knob the build plan calls out: model choice, the
    --smoke fast-path, and the architecture/training hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Train the VQ-VAE or its autoencoder baseline on HipMRI 2D slices."
    )
    parser.add_argument("--model", choices=list(MODELS), default="vqvae")
    parser.add_argument("--data-dir", default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-embeddings", type=int, default=512)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--num-downsampling-layers", type=int, default=3)
    parser.add_argument("--num-residual-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true",
        help="fast run on 20 slices for 2 epochs, to catch shape/gradient "
             "errors before committing a real training run",
    )
    parser.add_argument(
        "--verify-split", action="store_true",
        help="run the patient-level leakage check across all three splits "
             "before training starts",
    )
    return parser


def run_epoch(model, loader, device, optimizer=None):
    """Runs one pass over `loader`. If `optimizer` is given, trains
    (model.train(), backward + step); otherwise evaluates (model.eval(),
    no_grad). Returns a dict of epoch-averaged metrics.

    Used for both the training and validation passes, and for both VQVAE
    and Autoencoder, since modules.VQVAE.forward and modules.Autoencoder
    .forward share the same (reconstruction, vq_loss, perplexity,
    indices) return signature.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = total_recon = total_vq = total_ssim = total_perplexity = 0.0
    n_batches = 0
    n_perplexity_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)

            recon, vq_loss, perplexity, _ = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            loss = recon_loss + vq_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ssim_mean, _, _ = batch_ssim(recon, batch)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_ssim += ssim_mean
            if perplexity is not None:
                total_perplexity += perplexity.item()
                n_perplexity_batches += 1
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "vq_loss": total_vq / n_batches,
        "ssim": total_ssim / n_batches,
        "perplexity": (total_perplexity / n_perplexity_batches) if n_perplexity_batches else None,
    }


def plot_curves(history, model_name, figures_dir):
    """Writes three PNGs to figures_dir: loss curve, SSIM curve, and (for
    the VQ-VAE only) the codebook perplexity curve -- the metric that
    reveals codebook collapse long before it's visible in reconstructions.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss (reconstruction + vq)")
    plt.title(f"{model_name}: training loss")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f"{model_name}_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["val_ssim"])
    plt.axhline(0.6, linestyle="--", label="target SSIM = 0.6")
    plt.xlabel("epoch")
    plt.ylabel("validation SSIM")
    plt.title(f"{model_name}: validation SSIM")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f"{model_name}_ssim_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if any(p is not None for p in history["val_perplexity"]):
        plt.figure()
        plt.plot(epochs, history["val_perplexity"])
        plt.xlabel("epoch")
        plt.ylabel("codebook perplexity")
        plt.title(f"{model_name}: codebook perplexity (collapses toward 1)")
        plt.savefig(os.path.join(figures_dir, f"{model_name}_perplexity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()


def main():
    """Parses args, builds the model/data/optimizer, runs the training
    loop, and writes the metrics CSV, the best checkpoint, and the
    training-curve figures."""
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 4)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = HipMRISlices(args.data_dir, split="train", early_stop=args.smoke, verify_split=args.verify_split)
    val_ds = HipMRISlices(args.data_dir, split="validate", early_stop=args.smoke)
    print(f"Train slices: {len(train_ds)}, validation slices: {len(val_ds)}")

    # drop_last=False: there is no BatchNorm in modules.py (nothing that
    # needs a consistent batch size), and dropping the final batch
    # silently zeroed out training entirely on a small/smoke dataset
    # whenever it was smaller than one batch.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MODELS[args.model](
        in_channels=1,
        hidden_channels=args.hidden_channels,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        num_downsampling_layers=args.num_downsampling_layers,
        num_residual_layers=args.num_residual_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    csv_path = os.path.join(args.output_dir, f"{args.model}_metrics.csv")
    checkpoint_path = os.path.join(args.output_dir, f"{args.model}_best.pt")
    history = {"train_loss": [], "val_loss": [], "val_ssim": [], "val_perplexity": []}
    best_val_ssim = float("-inf")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_recon_loss", "train_vq_loss",
            "val_loss", "val_ssim", "val_perplexity",
        ])

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_ssim"].append(val_metrics["ssim"])
            history["val_perplexity"].append(val_metrics["perplexity"])

            writer.writerow([
                epoch, train_metrics["loss"], train_metrics["recon_loss"], train_metrics["vq_loss"],
                val_metrics["loss"], val_metrics["ssim"], val_metrics["perplexity"],
            ])
            f.flush()

            perplexity_str = f"{val_metrics['perplexity']:.1f}" if val_metrics["perplexity"] is not None else "n/a"
            print(
                f"[{args.model}] epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_ssim={val_metrics['ssim']:.4f} "
                f"val_perplexity={perplexity_str}"
            )

            if val_metrics["perplexity"] is not None and val_metrics["perplexity"] < args.num_embeddings * 0.02:
                print(
                    f"  WARNING: codebook perplexity ({val_metrics['perplexity']:.1f}) is very low "
                    f"relative to num_embeddings ({args.num_embeddings}) -- this looks like codebook "
                    f"collapse. Consider stopping and checking the EMA update / learning rate."
                )

            if val_metrics["ssim"] > best_val_ssim:
                best_val_ssim = val_metrics["ssim"]
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  New best validation SSIM ({best_val_ssim:.4f}) -- checkpoint saved to {checkpoint_path}")

    plot_curves(history, args.model, "figures")
    print(f"Done. Best validation SSIM: {best_val_ssim:.4f}. Checkpoint: {checkpoint_path}. Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
