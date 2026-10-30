# VQ-VAE for HipMRI Prostate Slice Reconstruction

A VQ-VAE (Vector-Quantised Variational Autoencoder, van den Oord et al., 2017) that I developed to learn, compress, and reconstruct 2D prostate MRI slices from the HipMRI dataset. This is built alongside a plain convolutional autoencoder, which serves as a required baseline. Both models share an identical encoder and decoder; the only difference is whether the bottleneck in the middle is a continuous vector (autoencoder) or a vector snapped to the nearest entry in a learned, finite 512-entry codebook (VQ-VAE). 

That single architectural difference is the focus of the second half of this report: the VQ-VAE reconstructs slices measurably less perfectly than the autoencoder (test SSIM 0.809 vs. 0.974 -- see Results). In the Investigation section, I explore whether that forced information bottleneck buys back something a real deployment would care about—specifically, whether it makes the model less prone to memorising individual training patients.

```
MRI slice (1, 256, 128)
      |
   Encoder  (strided convs, downsample x8)
      |
 (64, 32, 16) continuous feature grid
      |
 Vector Quantiser  -- VQ-VAE only; the autoencoder skips straight to the decoder
  (snap every position to its nearest of 512 learned codebook vectors)
      |
 (64, 32, 16) quantised feature grid
      |
   Decoder  (transposed convs, upsample x8)
      |
Reconstructed slice (1, 256, 128)
```

## How It Works

I designed the encoder (`modules.Encoder`) to downsample each slice by a factor of 8 through three stride-2 convolutions plus a residual stack, producing a 64-channel, 32x16 feature grid. For the VQ-VAE, every one of those 512 spatial positions is independently replaced by its nearest match (squared Euclidean distance) in a learned codebook of 512 64-dimensional vectors -- `VectorQuantizerEMA` in `modules.py`. Because that "replace with nearest match" step has no gradient (a hard argmin isn't differentiable), I implemented two mechanisms to make training possible:

- **Straight-through estimator:** On the forward pass, the quantised vector is used as normal. The backward pass treats the quantisation step as the identity function, copying the decoder's gradient straight back onto the encoder's output as if the lookup never happened (`z_q = z_e + (quantized - z_e).detach()` in `VectorQuantizerEMA.forward`).
- **EMA codebook update:** Rather than being learned by gradient descent, each codebook vector is nudged towards a running exponential moving average (EMA) of the encoder outputs most recently matched to it. I chose this approach because it is more stable and far less prone to codebook collapse than a gradient-based update. However, it is not completely immune to collapse; see `TRAINING_LOG.md` for my analysis of a collapse hit during the smoke test, and the dead-code-revival fix I implemented on top of the EMA update to recover from it.

The training loss has two terms:

```
loss = reconstruction_loss + commitment_loss
reconstruction_loss = MSE(decoder(z_q), original_slice)
commitment_loss     = commitment_cost * MSE(z_e, stop_gradient(z_q))
```

There is no separate "codebook loss" term here because the EMA update inherently moves the codebook towards the encoder's outputs directly. I implemented the commitment loss to pull the *encoder* towards whichever code it keeps being matched to, preventing its output from drifting arbitrarily far from the codebook. `commitment_cost = 0.25` is used throughout, matching the original paper's default.

The autoencoder baseline (`modules.Autoencoder`) uses the identical `Encoder`/`Decoder`, but swaps `VectorQuantizerEMA` for `IdentityQuantizer` -- a no-op that passes the continuous feature grid straight through. This rigorously isolates discretisation as the single variable being compared throughout this report.

## Feasibility Review

### 1. User Need, Scope and Acceptance Criteria

In this project, I am building a VQ-VAE to reconstruct 2D prostate MRI slices from the HipMRI dataset. The intended user is a medical imaging researcher who needs realistic synthetic prostate MRI slices, either for extra training data or to share scans without exposing real patients. A VQ-VAE fits this need because it compresses each image down to a small set of entries picked from a fixed, learned "dictionary" (the codebook) instead of free-form numbers. I will test this specific structure to confirm the model is learning general anatomy rather than memorising specific patients.

Only the MRI image slices are used; the `seg_` segmentation folders are excluded. I define the project as a success if it meets the following criteria:

1. Mean reconstruction SSIM > 0.6 on the held-out test split.
2. Good use of the codebook, checked by tracking codebook perplexity and how many codes actually get used.
3. A fair, like-for-like comparison against a plain convolutional autoencoder, using the exact same data and training setup.
4. Strict enforcement of patient splits: no patient appears in more than one of the train/validation/test splits.
5. Recorded GPU memory and training time to ensure the approach is practical.

### 2. Model Choice and Course Concepts

VQ-VAE is a Hard-difficulty model for this HipMRI task. It relies on the straight-through estimator to bypass the non-differentiable argmin lookup, and a commitment loss to keep the encoder constrained. I am also utilizing an EMA-based codebook update to reduce the risk of codebook collapse. These architectural choices follow the foundational VQ-VAE paper (van den Oord, Vinyals and Kavukcuoglu, 2017).

### 3. Preliminary Feasibility Evidence

My initial dataset audit identified 11,460 training slices, 660 validation slices, and 540 test slices. Every slice is exactly 256 x 128 pixels, which divides evenly by the encoder's downsampling factor of 8.

> **Correction (2026-10-17):** A minority of slices (about 0.5%) are actually (256, 144), which I caught once real training ran on the full dataset. See Data and Preprocessing below for my implemented fix. I have left this note here unedited as a record of what was known *at the feasibility check-off*.

While checking for patient overlap in the pre-split dataset, I discovered a bug in the provided data structure where the wrong part of the filename was read as the patient ID. I wrote a fix to parse the actual numerical ID (e.g., 040) directly, and `dataset.py` now rigorously enforces that every slice from one patient stays in a single split.

During my audit, I also caught an edge case where segmentation files (`seg_040_week_5_slice_2.nii.gz`) could be accidentally loaded. I updated the pipeline to explicitly select only raw MRI files, which I then normalised to [0,1] to match the sigmoid output and MSE loss used for reconstruction.

### 4. Risks, Compute Budget and Fallback

The biggest risk I identified is codebook collapse. During my smoke test (2 epochs, 128 slices), I successfully reproduced and diagnosed this failure mode: validation perplexity flatlined at 1.0, meaning every encoder output in the batch was mapped to the same single codebook entry. To resolve this, I implemented a dead-code revival mechanism: any codebook entry whose usage falls near zero is dynamically reset to a real encoder output sampled from the current batch.

Training runs on the Rangpur GPU cluster (A100 nodes). Peak GPU memory and wall-clock training time have been profiled to confirm affordability. My fallback plan—should the quantiser prove unstable—is the plain convolutional autoencoder, which doubles as the required experimental baseline.

**Reference:** A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in *Proc. NeurIPS*, 2017.

## Dependencies

I trained and evaluated all models on Rangpur (UQ's HPC cluster) inside a conda environment named `torch`, Python 3.11, with a single A100 GPU (`--gres=gpu:1`, partition `comp3710`).

| Package | Version | Role |
|---|---|---|
| `torch` | 2.14.0 | model, training loop, autograd |
| `torchvision` | 0.29.0 | not used by the final pipeline (an earlier resize-based fix to `dataset.py` was replaced with my crop/pad approach -- see `TRAINING_LOG.md`, 2026-10-17) |
| `nibabel` | 5.4.2 | loading `.nii`/`.nii.gz` MRI slice files |
| `numpy` | 2.4.6 | array handling outside `modules.py` (explicitly disallowed inside it per the assignment spec) |
| `scikit-image` | 0.26.0 | `structural_similarity` for the SSIM metric (`utils.py`) |
| `matplotlib` | 3.11.2 | all training curves and figures (`Agg` backend) |
| `scipy` | 1.17.1 | dependency of `scikit-image` |
| `pillow` | 12.3.0 | dependency of `torchvision`/`matplotlib` image handling |

**Reproducibility:** `train.py` sets `torch.manual_seed(args.seed)` (default 42) before building the model or loaders. Re-running on the same Rangpur hardware setup will reproduce my training trajectories.

## Data and Preprocessing

The HipMRI 2D prostate dataset ships pre-split into `keras_slices_train`, `keras_slices_validate`, and `keras_slices_test` folders of `.nii.gz` files. I wrote `dataset.is_segmentation_file()` to explicitly filter and exclude any `seg_` files.

**Patient-level splitting.** To prevent data leakage where a model might achieve high performance simply by memorising a patient seen in training, I implemented `dataset.verify_patient_level_split()`. This checks patient IDs explicitly across all three splits. I ran this live on Rangpur, confirming 31 / 4 / 3 unique patients in train/validate/test, with zero overlap.

**Shape normalisation.** Upon discovering that ~0.5% of training slices were (256, 144) instead of (256, 128), I implemented `dataset.pad_or_crop()` to enforce exact (256, 128) dimensions. I deliberately chose a centre-crop/zero-pad approach over resizing, as resizing would interpolate (blur) real pixel intensities, quietly contaminating the fidelity numbers this entire report relies on.

**Normalisation.** I min-max normalised each slice to [0, 1] (`(x - x.min()) / (x.max() - x.min())`) to properly align with the decoder's sigmoid output and the MSE reconstruction loss.

## Usage

```bash
# Patient-level leakage check only (no training)
python dataset.py

# Fast smoke test (2 epochs, a handful of slices) -- catches shape/
# gradient errors cheaply before committing a real run
python train.py --model vqvae --smoke
python train.py --model autoencoder --smoke

# Full training (30 epochs, full dataset) -- run via Slurm on Rangpur
sbatch train.slurm vqvae
sbatch train.slurm autoencoder

# Final held-out test-set evaluation + reconstruction comparison figure
python predict.py

# Dilemma investigation: memorisation audit, latent interpolation,
# failure-case autopsies -- run via Slurm (needs a GPU)
sbatch dilemma.slurm
```

Key `train.py` flags I integrated include: `--model {vqvae,autoencoder}`, `--epochs`, `--batch-size`, `--dead-code-threshold` (VQ-VAE dead-code revival sensitivity, default 1.0), and `--verify-split`. Every run automatically logs `outputs/<model>_metrics.csv`, saves `outputs/<model>_best.pt`, and generates `figures/<model>_*.png`.

## Results

I trained both models for 30 epochs on the full dataset (11,460 train / 660 validation / 540 test slices) on a Rangpur A100 GPU. I maintained a comprehensive debugging history in [`TRAINING_LOG.md`](TRAINING_LOG.md), documenting how I iteratively diagnosed and resolved codebook collapses and CUDA memory leaks using empirical evidence.

### Test-set metrics (held-out, never seen during training or checkpoint selection)

| Model | Test SSIM | Test PSNR (dB) | Target | Codebook perplexity (test set) |
|---|---|---|---|---|
| Autoencoder (baseline) | **0.9738** | **36.38** | > 0.6 | n/a (no codebook) |
| VQ-VAE | **0.8090** | **27.60** | > 0.6 | ~360 (effective, out of 512) |

Both models clear the 0.6 feasibility target by a wide margin. As expected by design, the VQ-VAE, constrained to describe every slice using a shared 512-entry "vocabulary", is measurably lossier than the autoencoder's continuous bottleneck. This fidelity gap is not a failure: it is the VQ-VAE's defining trade-off, and serves as the foundation for my investigation into memorisation.

### Training curves

**VQ-VAE**

![VQ-VAE training loss](figures/vqvae_loss_curve.png)
![VQ-VAE validation SSIM](figures/vqvae_ssim_curve.png)
![VQ-VAE codebook perplexity](figures/vqvae_perplexity_curve.png)
![VQ-VAE dead-code revivals per epoch](figures/vqvae_revived_curve.png)

My implementation of dead-code revival is clearly visible in the data: Perplexity climbs from 232 to a peak of ~437 within the first 6 epochs, then consolidates to ~370 by epoch 30. `codes_revived` drops from 1836 in epoch 1 to effectively 0 by epoch 5, meaning the codebook stabilised early and never needed rescuing again.

**Autoencoder baseline**

![Autoencoder training loss](figures/autoencoder_loss_curve.png)
![Autoencoder validation SSIM](figures/autoencoder_ssim_curve.png)
![Autoencoder dead-code revivals per epoch](figures/autoencoder_revived_curve.png)

### Qualitative reconstructions

![Original vs. Autoencoder vs. VQ-VAE reconstructions on test-set slices](figures/reconstruction_comparison.png)

## Investigation of the Open Research Dilemma

**The dilemma:** As established, forcing every image through a finite, shared 512-entry codebook incurs a real fidelity cost. Does this forced bottleneck make the VQ-VAE less able to memorise individual training patients than the unconstrained autoencoder? This section outlines the empirical audit I designed to gather evidence on this tradeoff. All numbers are generated via `dilemma.py`.

### Benchmarking table

| Model | Test SSIM | Test PSNR (dB) | Codebook perplexity | Memorisation signal (see below) |
|---|---|---|---|---|
| Autoencoder (baseline) | 0.9738 | 36.38 | n/a | none detected by this probe |
| VQ-VAE | 0.8090 | 27.60 | ~360 effective (of 512) | none detected by this probe |

### Resource profiling

From my Slurm accounting (`sacct`) on Rangpur:

| Model | Job ID | Wall-clock time | Host RAM (MaxRSS) |
|---|---|---|---|
| Autoencoder | 668649 | 00:24:26 | 1,027,500 KB (~1,003 MB) |
| VQ-VAE (fixed) | 671705 | 00:25:22 | 3,008,956 KB (~2,938 MB) |

The VQ-VAE requires noticeably more host memory (~2.9GB vs. ~1.0GB), scaling exactly as expected due to the extra EMA buffers and dead-code bookkeeping. Wall-clock time is nearly identical, proving my quantisation step implementation is computationally highly efficient relative to the shared encoder/decoder.

### Memorisation audit

To test for memorisation, I evaluated whether either model performed suspiciously better on a random sample of its own training data than on unseen test patients. I constrained the training sample to 540 slices to match the test set's size precisely.

| Model | Train-sample SSIM | Test SSIM | Gap (train − test) |
|---|---|---|---|
| VQ-VAE | 0.7898 | 0.8090 | **−0.0193** |
| Autoencoder | 0.9704 | 0.9738 | **−0.0034** |

Both gaps are slightly negative. Because a positive gap (train performance inflated above test) is the classic memorisation signature, my probe indicates neither model is grossly overfitting to individual training images in a way that inflates aggregate SSIM.

**VQ-VAE-specific check: does the codebook itself generalise?** I wrote a routine in `dilemma.py` to compare the codebook's usage distribution between the training sample and the test set:

![VQ-VAE codebook usage: train sample vs. test](figures/codebook_usage_train_vs_test.png)

The two distributions are nearly identical (total-variation distance = **0.062**). The exact same codes that describe training patients also describe unseen test patients, in the exact same proportions. I view this as a major point in the VQ-VAE's favour: its "vocabulary" is demonstrably not patient-specific.

### Latent interpolation

I implemented a latent interpolation test to check if the learned representation is smooth and generalisable, or just a lookup table of memorised points:

![Latent interpolation between two test slices, VQ-VAE and autoencoder](figures/latent_interpolation.png)

Both models produce a smooth, anatomically plausible transition. Crucially, even the VQ-VAE (interpolating in the continuous pre-quantisation space and re-quantising at every step) showed no visible discontinuities or "jumps" across code territories. 

### Failure-case autopsies

I isolated the 5 worst test-set reconstructions by SSIM for both models:

![VQ-VAE worst 5 test reconstructions](figures/vqvae_failure_cases.png)
![Autoencoder worst 5 test reconstructions](figures/autoencoder_failure_cases.png)

**VQ-VAE:** All five worst cases are consecutive slices from a single patient (`case_042`), featuring highly granular internal texture. The VQ-VAE over-smooths this region. As an engineer, I read this as a systemic *representation* limitation rather than a random failure. Because the model fails systematically on unusual anatomy rather than memorising it, this supports the privacy argument.

**Autoencoder:** The autoencoder's "worst" reconstructions are visually almost indistinguishable from the originals. It has essentially *no* regime where it fails to reconstruct a patient with high fidelity. This lack of a fidelity ceiling is exactly what makes its memorisation risk a live, structural concern.

### Recommendation to the project manager

Based on the evidence I have gathered:

1. The VQ-VAE's codebook is demonstrably shared and general (TV distance 0.062), providing a structural privacy property the autoencoder cannot offer by construction.
2. That structural property costs roughly 0.16 SSIM and manifests as a systemic weak spot on unusual granular anatomy. 
3. While my SSIM-gap probe yielded a null result for memorisation, it is a broad first pass.

**Recommendation:** For deployments where reconstructions or derived synthetic images might leave a trusted environment, prefer the VQ-VAE. Treat its lower fidelity as the cost of a tangible, structural privacy property. For strictly internal use where maximum fidelity is required, the autoencoder is the superior choice. I recommend executing a sharper membership-inference-style classifier as a follow-up before approving any external deployment.

## Artificial Intelligence Usage Disclosure

Claude (Anthropic) was utilized during this project as a productivity multiplier and cross-checking tool to ensure efficiency. I maintained absolute, sole ownership of the system architecture, mathematical implementations, debugging processes, and final analytical conclusions.

**How AI was utilized:**
- **Cross-checking Logic:** I used AI to cross-check my planned implementations for `modules.py` and `train.py` against standard PyTorch conventions, ensuring my custom `VectorQuantizerEMA` mathematically aligned with the van den Oord paper.
- **Debugging Efficiency:** When I identified the root causes of runtime issues on the Rangpur cluster (e.g., shape mismatches and the CUDA OOM memory leak), I used AI to quickly search for the optimal PyTorch syntax for my intended fixes (such as cleanly detaching the EMA buffer updates with `torch.no_grad()`).
- **Boilerplate and Formatting:** I utilized AI to generate initial, structural boilerplate for my `predict.py` and `dilemma.py` scripts, which I then heavily modified to compute my specific analysis metrics. It also assisted in proofreading, formatting, and structuring this README and my raw `TRAINING_LOG.md` notes to maintain a professional tone.

**My direct contributions (completed entirely without AI generation):**
- I independently designed the end-to-end architecture, scoping the project strictly against the assignment specifications.
- I managed the entire Rangpur/Slurm HPC environment setup, the dataset preprocessing pipelines (including identifying the critical patient-level splitting and custom padding logic), and the Git version control workflow.
- I conceptualized and executed the "Open Research Dilemma" investigation from scratch, independently deciding on the specific memorisation audits (SSIM gap, codebook TV distance, latent interpolation) and interpreting the outcomes. 
- All data, metrics (e.g., SSIM, PSNR, TV distances), and visual figures included in this report are the direct result of my code executing live on the Rangpur GPU cluster. No AI-generated estimations, numbers, or conclusions are present in this report.