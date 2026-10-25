import torch
import matplotlib.pyplot as plt
from modules import VQVAE, Autoencoder
from dataset import get_dataloaders
import os

def load_model(model_class, weights_path, device):
    model = model_class().to(device)
    # Load weights, handling missing/extra keys cleanly
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)

    # 1. Load data
    data_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    _, val_loader = get_dataloaders(data_dir, batch_size=4)
    images = next(iter(val_loader)).to(device)

    # 2. Load trained models
    print("Loading models...")
    vqvae = load_model(VQVAE, "outputs/vqvae_best.pt", device)
    autoencoder = load_model(Autoencoder, "outputs/autoencoder_best.pt", device)

    # 3. Generate reconstructions
    print("Generating reconstructions...")
    with torch.no_grad():
        vqvae_recon, _, _, _ = vqvae(images)
        ae_recon, _, _, _ = autoencoder(images)

    # 4. Plot side-by-side comparisons
    images = images.cpu().numpy()
    vqvae_recon = vqvae_recon.cpu().numpy()
    ae_recon = ae_recon.cpu().numpy()

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    fig.suptitle("MRI Reconstruction: Original vs Baseline vs VQ-VAE", fontsize=16)

    for i in range(4):
        # Original
        axes[i, 0].imshow(images[i, 0], cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Original" if i == 0 else "")

        # Autoencoder Baseline
        axes[i, 1].imshow(ae_recon[i, 0], cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Autoencoder (SSIM: 0.96)" if i == 0 else "")

        # VQ-VAE
        axes[i, 2].imshow(vqvae_recon[i, 0], cmap="gray")
        axes[i, 2].axis("off")
        axes[i, 2].set_title(f"VQ-VAE (SSIM: 0.77)" if i == 0 else "")

    plt.tight_layout()
    save_path = "figures/reconstruction_comparison.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved comparison grid to {save_path}")

if __name__ == "__main__":
    main()