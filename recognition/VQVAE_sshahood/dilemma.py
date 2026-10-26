"""
dilemma.py

The "Investigation of the Open Research Dilemma" evidence-gathering
script. Produces three things predict.py deliberately doesn't:

1. A memorisation audit -- does either model do suspiciously better on
   the training patients it has seen than on the held-out test patients
   it hasn't? For the VQ-VAE specifically, does the learned codebook's
   usage pattern actually transfer to unseen patients, or does test data
   need a visibly different mix of codes?
2. A latent interpolation figure -- walking between two test images in
   latent space and decoding each step. Smooth, plausible intermediate
   anatomy suggests the model has learned a generalisable representation;
   garbage in between suggests it has just memorised isolated points.
3. Failure-case autopsies -- the worst individual test reconstructions,
   saved with their SSIM scores so each one can be looked at and
   explained rather than just averaged away.

Usage:
    python dilemma.py
"""

import os
import csv

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import VQVAE, Autoencoder
from dataset import HipMRISlices
from utils import batch_ssim, batch_psnr
from predict import load_model, evaluate_model, DATA_DIR, OUTPUT_DIR

FIGURES_DIR = "figures"
SEED = 42
N_INTERP_STEPS = 7
N_FAILURE_CASES = 5


def get_train_sample_loader(data_dir, n_samples, batch_size=32, seed=SEED):
    """A random, fixed-size sample of the TRAINING split, the same size as
    the test split, so its mean SSIM/PSNR is directly comparable to the
    test-set numbers predict.py already reports -- without this, "trained
    on 11,460 slices, evaluated on 540" would make any train/test gap
    partly an artefact of sample size, not just memorisation."""
    train_ds = HipMRISlices(data_dir, split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), size=min(n_samples, len(train_ds)), replace=False)
    subset = torch.utils.data.Subset(train_ds, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)


def codebook_usage_histogram(model, loader, device, num_embeddings):
    """Counts how often each of the VQ-VAE's codebook entries gets used
    across every image in `loader`, normalised to a probability
    distribution. Comparing this between the training sample and the test
    set is a VQ-VAE-specific memorisation check: if the codebook learned
    from training patients needs a visibly different usage pattern to
    describe unseen test patients, that's a sign the codebook encodes
    something patient-specific rather than general anatomy."""
    counts = torch.zeros(num_embeddings)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, _, indices = model(batch)
            flat_indices = indices.reshape(-1).cpu()
            counts += torch.bincount(flat_indices, minlength=num_embeddings).float()
    return (counts / counts.sum()).numpy()


def total_variation_distance(p, q):
    """0 = identical distributions, 1 = disjoint support. Simple,
    interpretable summary of how different two codebook-usage
    distributions are."""
    return 0.5 * np.abs(p - q).sum()


def run_memorisation_audit(vqvae, autoencoder, device):
    print("\n=== Memorisation audit: train-sample vs. test ===")
    test_ds = HipMRISlices(DATA_DIR, split="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    train_sample_loader = get_train_sample_loader(DATA_DIR, n_samples=len(test_ds))

    rows = []
    for name, model in [("vqvae", vqvae), ("autoencoder", autoencoder)]:
        train_metrics = evaluate_model(model, train_sample_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        gap = train_metrics["ssim"] - test_metrics["ssim"]
        print(
            f"[{name}] train_ssim={train_metrics['ssim']:.4f} "
            f"test_ssim={test_metrics['ssim']:.4f} gap={gap:+.4f} "
            f"(train_psnr={train_metrics['psnr']:.2f}dB test_psnr={test_metrics['psnr']:.2f}dB)"
        )
        rows.append([name, train_metrics["ssim"], test_metrics["ssim"], gap,
                     train_metrics["psnr"], test_metrics["psnr"]])

    with open(os.path.join(OUTPUT_DIR, "memorisation_audit.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "train_sample_ssim", "test_ssim", "ssim_gap",
                          "train_sample_psnr_db", "test_psnr_db"])
        writer.writerows(rows)

    # VQ-VAE-specific: does the codebook's usage pattern transfer to
    # unseen patients?
    num_embeddings = vqvae.quantizer.num_embeddings
    train_hist = codebook_usage_histogram(vqvae, train_sample_loader, device, num_embeddings)
    test_hist = codebook_usage_histogram(vqvae, test_loader, device, num_embeddings)
    tv_distance = total_variation_distance(train_hist, test_hist)
    print(f"[vqvae] codebook usage total-variation distance (train sample vs. test): {tv_distance:.4f}")

    order = np.argsort(-train_hist)  # sort by training usage, descending
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_embeddings), train_hist[order], alpha=0.6, label="train sample", width=1.0)
    plt.bar(range(num_embeddings), test_hist[order], alpha=0.6, label="test", width=1.0)
    plt.xlabel("codebook entry (sorted by training-set usage)")
    plt.ylabel("fraction of all encoder positions")
    plt.title(f"VQ-VAE codebook usage: train sample vs. test (TV distance = {tv_distance:.3f})")
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, "codebook_usage_train_vs_test.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return rows, tv_distance


def run_latent_interpolation(vqvae, autoencoder, device):
    print("\n=== Latent interpolation ===")
    test_ds = HipMRISlices(DATA_DIR, split="test")
    img_a = test_ds[0].unsqueeze(0).to(device)
    img_b = test_ds[len(test_ds) // 2].unsqueeze(0).to(device)

    alphas = np.linspace(0.0, 1.0, N_INTERP_STEPS)
    fig, axes = plt.subplots(2, N_INTERP_STEPS, figsize=(2 * N_INTERP_STEPS, 4.2))
    fig.suptitle("Latent interpolation between two test slices (left=A, right=B)", fontsize=13)

    for row, (name, model) in enumerate([("VQ-VAE", vqvae), ("Autoencoder", autoencoder)]):
        with torch.no_grad():
            z_a = model.encoder(img_a)
            z_b = model.encoder(img_b)
            for col, alpha in enumerate(alphas):
                z = (1 - alpha) * z_a + alpha * z_b
                z_q, _, _, _ = model.quantizer(z)
                recon = model.decoder(z_q)
                axes[row, col].imshow(recon[0, 0].cpu().numpy(), cmap="gray")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"a={alpha:.2f}")
        # axis("off") above also hides a normal ylabel, so label each row
        # with fig.text instead, positioned just left of that row's axes.
        row_y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        fig.text(0.01, row_y, name, va="center", ha="left", rotation=90, fontsize=11)

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    save_path = os.path.join(FIGURES_DIR, "latent_interpolation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def run_failure_autopsies(model, model_name, device):
    print(f"\n=== Failure-case autopsies: {model_name} ===")
    test_ds = HipMRISlices(DATA_DIR, split="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    all_ssim = []
    all_images = []
    all_recons = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            recon, _, _, _ = model(batch)
            _, _, scores = batch_ssim(recon, batch)
            all_ssim.extend(scores.tolist())
            all_images.append(batch.cpu())
            all_recons.append(recon.cpu())

    all_images = torch.cat(all_images, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    all_ssim = np.array(all_ssim)

    worst_indices = np.argsort(all_ssim)[:N_FAILURE_CASES]
    print("Worst test-set reconstructions (by SSIM):")
    for idx in worst_indices:
        file_path = test_ds.files[idx]
        print(f"  index {idx} ({os.path.basename(file_path)}): ssim={all_ssim[idx]:.4f}")

    fig, axes = plt.subplots(N_FAILURE_CASES, 2, figsize=(5, 2.6 * N_FAILURE_CASES))
    fig.suptitle(f"{model_name}: {N_FAILURE_CASES} worst test-set reconstructions", fontsize=13)
    for row, idx in enumerate(worst_indices):
        axes[row, 0].imshow(all_images[idx, 0].numpy(), cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 1].imshow(all_recons[idx, 0].numpy(), cmap="gray")
        axes[row, 1].axis("off")
        axes[row, 1].set_title("Reconstruction" if row == 0 else "")
        axes[row, 1].text(
            1.02, 0.5, f"SSIM={all_ssim[idx]:.3f}\n{os.path.basename(test_ds.files[idx])}",
            transform=axes[row, 1].transAxes, va="center", fontsize=8,
        )

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{model_name}_failure_cases.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading checkpoints...")
    vqvae = load_model(VQVAE, os.path.join(OUTPUT_DIR, "vqvae_best.pt"), device)
    autoencoder = load_model(Autoencoder, os.path.join(OUTPUT_DIR, "autoencoder_best.pt"), device)

    run_memorisation_audit(vqvae, autoencoder, device)
    run_latent_interpolation(vqvae, autoencoder, device)
    run_failure_autopsies(vqvae, "vqvae", device)
    run_failure_autopsies(autoencoder, "autoencoder", device)

    print("\nDone. See outputs/memorisation_audit.csv and figures/*.png")


if __name__ == "__main__":
    main()
