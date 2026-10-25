"""
predict.py

Final, held-out evaluation of both trained models (VQVAE and the
Autoencoder baseline) on the TEST split -- the one split neither model
ever saw during training, and that never influenced which checkpoint got
saved (that was 'validate''s job, during train.py). Reports the real
numbers this project is judged on (SSIM, PSNR, codebook perplexity),
writes them to outputs/test_metrics.csv for the README results table,
and saves a qualitative original-vs-reconstruction comparison figure.

Usage:
    python predict.py
"""

import os
import csv

import torch

import matplotlib
matplotlib.use("Agg")  # no display on Rangpur compute nodes
import matplotlib.pyplot as plt

from modules import VQVAE, Autoencoder
from dataset import get_dataloaders
from utils import batch_ssim, batch_psnr

DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
OUTPUT_DIR = "outputs"
FIGURES_DIR = "figures"


def load_model(model_class, weights_path, device):
    """Builds a model with the same default architecture train.py used
    (hidden_channels=128, embedding_dim=64, num_embeddings=512, ... --
    see build_argparser() in train.py) and loads its best checkpoint."""
    model = model_class().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def evaluate_model(model, loader, device):
    """Runs `model` over every batch in `loader` (no gradient needed --
    this is evaluation only) and returns dataset-wide mean SSIM, mean
    PSNR, and mean codebook perplexity (None for the Autoencoder, which
    has no codebook).

    Accumulates a running sum rather than averaging per-batch means,
    so the result is the exact mean over every test image even though
    the final batch is a different size from the rest (drop_last=False).
    """
    total_ssim = 0.0
    total_psnr = 0.0
    total_perplexity = 0.0
    n_images = 0
    n_perplexity_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, _, perplexity, _ = model(batch)

            _, _, ssim_scores = batch_ssim(recon, batch)
            _, _, psnr_scores = batch_psnr(recon, batch)
            total_ssim += float(ssim_scores.sum())
            total_psnr += float(psnr_scores.sum())
            n_images += batch.shape[0]

            if perplexity is not None:
                total_perplexity += perplexity.item()
                n_perplexity_batches += 1

    return {
        "ssim": total_ssim / n_images,
        "psnr": total_psnr / n_images,
        "perplexity": (total_perplexity / n_perplexity_batches) if n_perplexity_batches else None,
        "n_images": n_images,
    }


def save_comparison_figure(images, recons_by_model, save_path, n_examples=4):
    """Saves an (n_examples x (1 + len(recons_by_model))) grid: original
    slice in the first column, each model's reconstruction of the same
    slice in the columns after it. `recons_by_model` is an ordered dict
    of {label: reconstructed_batch}."""
    model_labels = list(recons_by_model.keys())
    n_cols = 1 + len(model_labels)

    fig, axes = plt.subplots(n_examples, n_cols, figsize=(3 * n_cols, 3 * n_examples))
    fig.suptitle("Test-set reconstructions: original vs. each model", fontsize=14)

    for row in range(n_examples):
        axes[row, 0].imshow(images[row, 0], cmap="gray")
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original")

        for col, label in enumerate(model_labels, start=1):
            axes[row, col].imshow(recons_by_model[label][row, 0], cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Evaluation loader comes from the TEST split -- see get_dataloaders()
    # in dataset.py for why not 'validate'.
    _, test_loader = get_dataloaders(DATA_DIR, batch_size=32, split_for_eval="test")
    print(f"Test slices: {len(test_loader.dataset)}")

    print("Loading checkpoints...")
    vqvae = load_model(VQVAE, os.path.join(OUTPUT_DIR, "vqvae_best.pt"), device)
    autoencoder = load_model(Autoencoder, os.path.join(OUTPUT_DIR, "autoencoder_best.pt"), device)

    print("Evaluating VQVAE on the test set...")
    vqvae_metrics = evaluate_model(vqvae, test_loader, device)
    print("Evaluating Autoencoder baseline on the test set...")
    ae_metrics = evaluate_model(autoencoder, test_loader, device)

    for name, metrics in [("vqvae", vqvae_metrics), ("autoencoder", ae_metrics)]:
        perplexity_str = f"{metrics['perplexity']:.1f}" if metrics["perplexity"] is not None else "n/a"
        print(
            f"[{name}] test_ssim={metrics['ssim']:.4f} test_psnr={metrics['psnr']:.2f}dB "
            f"test_perplexity={perplexity_str} (n={metrics['n_images']})"
        )

    # Write the headline numbers the README's Results section and the
    # dilemma-investigation benchmarking table both need.
    metrics_csv_path = os.path.join(OUTPUT_DIR, "test_metrics.csv")
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_ssim", "test_psnr_db", "test_perplexity", "n_images"])
        for name, metrics in [("vqvae", vqvae_metrics), ("autoencoder", ae_metrics)]:
            writer.writerow([
                name, metrics["ssim"], metrics["psnr"],
                metrics["perplexity"] if metrics["perplexity"] is not None else "",
                metrics["n_images"],
            ])
    print(f"Wrote {metrics_csv_path}")

    # Qualitative figure: reuse one real test batch for both models so
    # the comparison is on identical input slices.
    example_batch = next(iter(test_loader)).to(device)
    with torch.no_grad():
        vqvae_recon, _, _, _ = vqvae(example_batch)
        ae_recon, _, _, _ = autoencoder(example_batch)

    save_path = os.path.join(FIGURES_DIR, "reconstruction_comparison.png")
    save_comparison_figure(
        example_batch.cpu().numpy(),
        {
            f"Autoencoder (test SSIM {ae_metrics['ssim']:.3f})": ae_recon.cpu().numpy(),
            f"VQ-VAE (test SSIM {vqvae_metrics['ssim']:.3f})": vqvae_recon.cpu().numpy(),
        },
        save_path,
    )
    print(f"Saved comparison grid to {save_path}")


if __name__ == "__main__":
    main()
