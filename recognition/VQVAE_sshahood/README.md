# VQ-VAE for HipMRI Prostate Slice Reconstruction

> TODO: one-paragraph description of the algorithm and the problem it solves,
> plus the architecture figure, goes here.

## How It Works

> TODO: paragraph on the VQ-VAE mechanism -- encoder, codebook lookup,
> straight-through estimator, the three loss terms (reconstruction,
> codebook, commitment) with the loss equation.

## Feasibility Review

### 1. User Need, Scope and Acceptance Criteria

This project builds a VQ-VAE -- a neural network that learns to compress and
then rebuild images -- to reconstruct 2D prostate MRI slices from the
HipMRI dataset. The intended user is a medical imaging researcher who needs
realistic synthetic prostate MRI slices, either for extra training data or
to share scans without exposing real patients. A VQ-VAE fits this need
because it compresses each image down to a small set of entries picked from
a fixed, learned "dictionary" (the codebook) instead of free-form numbers.
That structure is what would let a later model generate new images from
scratch, and it's also what we'll test to confirm the model is learning
general anatomy rather than memorising specific patients.

Only the MRI image slices are used; the `seg_` segmentation folders are
excluded. The project is judged a success if it meets:

1. Mean reconstruction SSIM > 0.6 on the held-out test split.
2. Good use of the codebook, checked by tracking codebook perplexity and
   how many codes actually get used.
3. A fair, like-for-like comparison against a plain convolutional
   autoencoder, same data and training setup.
4. No patient appears in more than one of the train/validation/test splits.
5. Recorded GPU memory and training time, so the approach is known to be
   practical as well as accurate.

### 2. Model Choice and Course Concepts

VQ-VAE is a Hard-difficulty model for this HipMRI task. It has three parts:
an encoder that compresses each MRI slice into a grid of numbers, a vector
quantiser that replaces each of those numbers with its closest match from a
learned codebook, and a decoder that rebuilds the slice from those matched
codes.

The "replace with nearest match" step has no gradient -- you can't train
through a hard lookup the normal way. The fix is the straight-through
estimator: during training, the gradient from the decoder is copied
straight back to the encoder as if the lookup never happened. A commitment
loss is also added, nudging the encoder's output to stay close to whichever
codebook entry it keeps getting matched to. An EMA-based codebook update --
where codebook entries move toward a running average of the encoder outputs
assigned to them, rather than being updated by gradient descent -- will
also be used to reduce the risk of codebook collapse, where the model ends
up only ever using a couple of codes. These choices follow the original
VQ-VAE paper (van den Oord, Vinyals and Kavukcuoglu, 2017).

### 3. Preliminary Feasibility Evidence

An initial dataset audit found 11,460 training slices, 660 validation
slices, and 540 test slices. Every slice is exactly 256 x 128 pixels, and
both numbers divide evenly by the encoder's planned downsampling factor of
8, so no padding or cropping is needed.

The dataset folders ship pre-split, but checking for patient overlap
uncovered a bug: the code was reading the wrong part of the filename as the
patient ID. This has been fixed to read the actual numerical ID (e.g. 040),
and `dataset.py` will enforce that every slice from one patient stays in a
single split.

A second early mistake was accidentally inspecting a segmentation file
(`seg_040_week_5_slice_2.nii.gz`) instead of an MRI slice -- its values
ranged 0-3 because those are label classes, not image intensities. The
pipeline now explicitly selects only raw MRI files, normalised to [0,1] to
match the sigmoid output and MSE loss used for reconstruction.

Before any full training run, a smoke test will confirm a real batch passes
through the whole data loader and model with no shape or gradient errors.

### 4. Risks, Compute Budget and Fallback

The biggest risk is codebook collapse -- the model learning to use only a
handful of codes. This will be caught early by watching codebook perplexity
and per-code usage from the first training run. A second risk, already
partly surfaced above, is patient leakage from filename-parsing mistakes;
this is addressed by the corrected, explicit patient-level split.

Training will run on the Rangpur GPU cluster. Peak GPU memory and
wall-clock training time will be recorded from the first runs to understand
what's actually affordable. The next planned step is a short smoke test,
followed by a first real VQ-VAE training run.

If the quantiser proves hard to get working, the fallback is a plain
convolutional autoencoder -- identical encoder, decoder, data and training
setup, just without the codebook step. This gives a working reconstruction
pipeline while the quantiser is debugged, and doubles as the required
comparison baseline.

**Reference:** A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural
discrete representation learning," in *Proc. NeurIPS*, 2017.

## Dependencies

> TODO: exact versions (Python, PyTorch, nibabel, etc.) and a note on
> reproducibility (seed, run-to-run variance).

## Data and Preprocessing

> TODO: dataset description, normalisation choice and why, train/val/test
> split justification with the patient-leakage argument spelled out.

## Usage

> TODO: exact commands to train and predict, including the Slurm script.

## Results

> TODO: example inputs/outputs, training curves, reconstruction grids.

## Investigation of the Open Research Dilemma

> TODO: benchmarking table vs. baseline, resource profiling table,
> memorisation audit, latent interpolation figure, 3-5 failure autopsies,
> recommendation to the project manager.

## Artificial Intelligence Usage Disclosure

> TODO: which AI tools were used, what specific tasks they assisted with,
> and how outputs were audited/verified/tested. Mandatory, exact heading.
