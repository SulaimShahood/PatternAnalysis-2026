"""
modules.py

Model components for the VQ-VAE and its convolutional-autoencoder baseline,
built to reconstruct 2D HipMRI prostate slices (COMP3710 Pattern
Recognition, VQ-VAE project, Hard difficulty).

Everything in this file is pure PyTorch -- no NumPy or any other array
library -- per the assignment spec.

Architecture overview
----------------------
Encoder:   MRI slice (1, H, W) -> continuous feature grid (D, H/8, W/8)
Quantiser: each feature-grid vector is snapped to the nearest of K learned
           codebook vectors (an EMA-updated vector quantiser with a
           straight-through gradient estimator, following van den Oord,
           Vinyals and Kavukcuoglu, "Neural Discrete Representation
           Learning", NeurIPS 2017)
Decoder:   quantised feature grid -> reconstructed slice (1, H, W)

VQVAE and Autoencoder share the same Encoder/Decoder implementation; the
only difference between them is whether the bottleneck is quantised
(VectorQuantizerEMA) or left continuous (IdentityQuantizer). That makes the
VQ-VAE vs. baseline comparison in the report attributable to exactly one
variable: discretisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation residual block: ReLU -> 3x3 conv -> ReLU -> 1x1 conv,
    added back onto its input. Adds depth to the encoder/decoder without
    changing spatial resolution or channel count.
    """

    def __init__(self, channels: int, hidden_channels: int = None):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, C, H, W), same shape in and out."""
        return x + self.block(x)


class Encoder(nn.Module):
    """Strided-convolution encoder. Downsamples the input by a factor of 2
    per layer (default 3 layers -> factor of 8 total, matching the dataset
    audit in the README), applies a residual stack, then a 1x1 conv that
    projects onto the embedding dimension the quantiser expects.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 128,
                 embedding_dim: int = 64, num_downsampling_layers: int = 3,
                 num_residual_layers: int = 2):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_downsampling_layers):
            out_channels = hidden_channels if i > 0 else hidden_channels // 2
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels = out_channels
        for _ in range(num_residual_layers):
            layers.append(ResidualBlock(channels))
        layers.append(nn.Conv2d(channels, embedding_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) -> (B, embedding_dim, H/2^n, W/2^n)."""
        return self.net(x)


class Decoder(nn.Module):
    """Mirrors the encoder: a 1x1 conv back up to hidden_channels, a
    residual stack, then transposed convolutions that upsample by a factor
    of 2 per layer back to the original resolution. Kernel size 4 with
    stride 2 is used throughout (stride divides the kernel size evenly),
    which avoids the checkerboard artifacts a mismatched kernel/stride
    would otherwise print into every reconstruction. Ends in a sigmoid
    because inputs are min-max normalised to [0, 1] and trained with MSE.
    """

    def __init__(self, out_channels: int = 1, hidden_channels: int = 128,
                 embedding_dim: int = 64, num_downsampling_layers: int = 3,
                 num_residual_layers: int = 2):
        super().__init__()
        layers = [nn.Conv2d(embedding_dim, hidden_channels, kernel_size=1)]
        for _ in range(num_residual_layers):
            layers.append(ResidualBlock(hidden_channels))
        channels = hidden_channels
        for i in range(num_downsampling_layers):
            is_last = i == num_downsampling_layers - 1
            next_channels = out_channels if is_last else max(hidden_channels // 2, out_channels)
            layers.append(nn.ConvTranspose2d(channels, next_channels, kernel_size=4, stride=2, padding=1))
            if not is_last:
                layers.append(nn.ReLU(inplace=True))
            channels = next_channels
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, embedding_dim, H/2^n, W/2^n) -> (B, out_channels, H, W)."""
        return self.net(z)


class VectorQuantizerEMA(nn.Module):
    """Vector quantiser with an exponential-moving-average codebook update
    (van den Oord et al., 2017, Appendix A.1). Each spatial position of the
    encoder's output is replaced by its nearest neighbour in a learned
    codebook of num_embeddings vectors. The codebook is not learned by
    gradient descent -- it is nudged towards a running average of the
    encoder outputs assigned to each code, which is more stable and far
    less prone to codebook collapse than the gradient-based update used in
    the original paper's main formulation.
    """

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64,
                 commitment_cost: float = 0.25, decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embedding.clone())

    def forward(self, z_e: torch.Tensor):
        """z_e: (B, D, H, W) continuous encoder output.

        Returns:
            z_q: (B, D, H, W) quantised output. The straight-through
                estimator is applied so gradients flow back to the encoder
                as if this were the identity function.
            loss: scalar commitment-loss tensor.
            perplexity: scalar tensor, exp(entropy of code usage this
                batch). Equals num_embeddings if every code is used equally
                often, and collapses toward 1 if the model has settled on
                using only a handful of codes (codebook collapse).
            encoding_indices: (B, H, W) long tensor, the chosen code index
                per spatial position. Used by predict.py for the
                memorisation audit and would feed a later generative prior.
        """
        B, D, H, W = z_e.shape
        # (B, D, H, W) -> (B*H*W, D): one row per spatial position
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)

        # Squared Euclidean distance from every encoder vector to every
        # codebook vector: ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        encoding_indices = distances.argmin(dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type_as(flat)

        quantized = encodings @ self.embedding
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        if self.training:
            # EMA update of per-code usage counts and the running sum of
            # encoder vectors assigned to each code (no gradient involved)
            cluster_size = encodings.sum(0)
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            dw = encodings.t() @ flat
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Laplace smoothing so a code with zero hits this batch isn't
            # divided by (near) zero
            n = self.ema_cluster_size.sum()
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            self.embedding.copy_(self.ema_w / smoothed_cluster_size.unsqueeze(1))

        # Commitment loss: keeps the encoder's output close to whichever
        # code it keeps being assigned to. No separate codebook loss term
        # is needed -- the EMA update above takes its place.
        loss = self.commitment_cost * F.mse_loss(z_e, quantized.detach())

        # Straight-through estimator: z_q equals the quantised vector on
        # the forward pass, but on the backward pass this line behaves as
        # the identity, so the decoder's gradient is copied straight back
        # onto the encoder's output past the (non-differentiable)
        # nearest-neighbour lookup.
        z_q = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, loss, perplexity, encoding_indices.view(B, H, W)


class IdentityQuantizer(nn.Module):
    """Drop-in replacement for VectorQuantizerEMA that does nothing to its
    input. Used by the Autoencoder baseline so it shares exactly the same
    Encoder/Decoder code as the VQVAE, differing from it in only this one
    module -- i.e. the baseline/VQ-VAE comparison isolates discretisation
    as the single variable under test.
    """

    def forward(self, z_e: torch.Tensor):
        """z_e: (B, D, H, W) -> unchanged, with a zero loss and no
        perplexity/indices (those are only meaningful for a discrete
        codebook)."""
        zero_loss = z_e.new_zeros(())
        return z_e, zero_loss, None, None


class VQVAE(nn.Module):
    """Encoder -> EMA vector quantiser -> Decoder. The Hard-difficulty
    model this project is built around."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 128,
                 embedding_dim: int = 64, num_embeddings: int = 512,
                 num_downsampling_layers: int = 3, num_residual_layers: int = 2,
                 commitment_cost: float = 0.25, decay: float = 0.99, **_unused):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim,
                                num_downsampling_layers, num_residual_layers)
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                             commitment_cost, decay)
        self.decoder = Decoder(in_channels, hidden_channels, embedding_dim,
                                num_downsampling_layers, num_residual_layers)

    def forward(self, x: torch.Tensor):
        """x: (B, in_channels, H, W) raw slice.
        Returns (reconstruction, vq_loss, perplexity, encoding_indices)."""
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity, indices


class Autoencoder(nn.Module):
    """Plain convolutional autoencoder baseline: identical Encoder/Decoder
    to VQVAE, but the bottleneck stays continuous (IdentityQuantizer in
    place of VectorQuantizerEMA). Required by the assignment spec as the
    quantitative comparison point for the VQ-VAE, and doubles as a working
    fallback while the quantiser is being debugged.

    Accepts (and ignores, via **_unused) the quantiser-only keyword
    arguments (num_embeddings, commitment_cost, decay) so train.py can
    construct either model from one shared config dict.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 128,
                 embedding_dim: int = 64, num_downsampling_layers: int = 3,
                 num_residual_layers: int = 2, **_unused):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim,
                                num_downsampling_layers, num_residual_layers)
        self.quantizer = IdentityQuantizer()
        self.decoder = Decoder(in_channels, hidden_channels, embedding_dim,
                                num_downsampling_layers, num_residual_layers)

    def forward(self, x: torch.Tensor):
        """x: (B, in_channels, H, W) raw slice.
        Returns (reconstruction, zero_loss, None, None) -- the same
        4-tuple interface as VQVAE.forward, so train.py can train both
        models from one identical loop."""
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity, indices
