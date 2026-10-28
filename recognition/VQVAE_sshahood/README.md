# VQ-VAE for HipMRI Prostate Slice Reconstruction

A VQ-VAE (Vector-Quantised Variational Autoencoder, van den Oord et al.,
2017) that learns to compress and reconstruct 2D prostate MRI slices from
the HipMRI dataset, built alongside a plain convolutional autoencoder as a
required baseline. Both models share an identical encoder and decoder;
the only difference between them is whether the bottleneck in the middle
is a continuous vector (autoencoder) or a vector snapped to the nearest
entry in a learned, finite 512-entry codebook (VQ-VAE). That one-variable
difference is what the whole second half of this README is about: the
VQ-VAE reconstructs slices measurably less perfectly than the autoencoder
(test SSIM 0.809 vs. 0.974 -- see Results), and the Investigation section
asks whether that forced information bottleneck buys back something a
real deployment would care about -- specifically, whether it makes the
model less able to memorise individual training patients.

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

The encoder (`modules.Encoder`) downsamples each slice by a factor of 8
through three stride-2 convolutions plus a residual stack, producing a
64-channel, 32x16 feature grid. For the VQ-VAE, every one of those 512
spatial positions is independently replaced by its nearest match (squared
Euclidean distance) in a learned codebook of 512 64-dimensional vectors --
`VectorQuantizerEMA` in `modules.py`. That "replace with nearest match"
step has no gradient (a hard argmin isn't differentiable), so two things
make training possible anyway:

- **Straight-through estimator:** on the forward pass the quantised
  vector is used as normal, but the backward pass treats the quantisation
  step as the identity function, copying the decoder's gradient straight
  back onto the encoder's output as if the lookup never happened
  (`z_q = z_e + (quantized - z_e).detach()` in `VectorQuantizerEMA.forward`).
- **EMA codebook update:** rather than being learned by gradient descent
  (as in the original paper's simplest formulation), each codebook vector
  is nudged towards a running exponential moving average of the encoder
  outputs most recently matched to it. This is more stable and far less
  prone to codebook collapse than a gradient-based update -- though not
  immune to it; see `TRAINING_LOG.md` for the real collapse this project
  hit during the smoke test, and the dead-code-revival fix added on top
  of the EMA update to recover from it.

The training loss has two terms:

```
loss = reconstruction_loss + commitment_loss
reconstruction_loss = MSE(decoder(z_q), original_slice)
commitment_loss     = commitment_cost * MSE(z_e, stop_gradient(z_q))
```

There is no separate "codebook loss" term here (present in the original
paper's formulation) because the EMA update above already moves the
codebook towards the encoder's outputs directly, making a second
gradient-based pull redundant. The commitment loss instead pulls the
*encoder* towards whichever code it keeps being matched to, preventing
its output from drifting arbitrarily far from the codebook (which would
otherwise let `vq_loss` and the reconstruction quality become unstable
against each other). `commitment_cost = 0.25` throughout, matching the
original paper's default.

The autoencoder baseline (`modules.Autoencoder`) uses the identical
`Encoder`/`Decoder`, but swaps `VectorQuantizerEMA` for `IdentityQuantizer`
-- a no-op that passes the continuous feature grid straight through. This
isolates discretisation as the single variable being compared throughout
this report.

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

> **Correction (2026-10-17):** this turned out to be wrong. A minority of
> slices (about 0.5%) are actually (256, 144), only found once real
> training ran on the full dataset rather than the smoke-test subset this
> audit sampled from. See Data and Preprocessing below for the fix. Left
> here unedited, with this note, rather than silently rewritten, because
> this section is a record of what was known *at the feasibility
> check-off*, not a living summary.

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
handful of codes -- and this is no longer a hypothetical risk: the smoke
test (2 epochs, 128 slices) already reproduced it. Validation perplexity
was 1.0 on both epochs, meaning every encoder output in the batch was
mapped to the same single codebook entry out of 512. This is a known
failure mode at initialisation (one random codebook vector ends up nearest
to almost everything, "wins" every comparison, and a code that's never
chosen never receives an EMA update, so it stays dead). The planned fix is
dead-code revival: any codebook entry whose usage has fallen near zero is
reset to a real encoder output sampled from the current batch, giving it a
fresh chance instead of leaving it permanently unused. This will be
implemented and re-verified on a smoke test before any full training run.
A second risk, already partly surfaced above, is patient leakage from
filename-parsing mistakes; this is addressed by the corrected, explicit
patient-level split, which has been run end-to-end on Rangpur (31 train /
4 validate / 3 test patients, no overlap).

Training runs on the Rangpur GPU cluster (A100 nodes). Peak GPU memory
and wall-clock training time will be recorded from the first full runs to
understand what's actually affordable. The immediate next step is the
dead-code revival fix above, re-verified on a smoke test, followed by the
first full VQ-VAE and baseline training runs.

If the quantiser proves hard to get working, the fallback is a plain
convolutional autoencoder -- identical encoder, decoder, data and training
setup, just without the codebook step. This gives a working reconstruction
pipeline while the quantiser is debugged, and doubles as the required
comparison baseline.

**Reference:** A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural
discrete representation learning," in *Proc. NeurIPS*, 2017.

## Dependencies

Trained and evaluated on Rangpur (UQ's HPC cluster) inside a conda
environment named `torch`, Python 3.11, with a single A100 GPU
(`--gres=gpu:1`, partition `comp3710`).

| Package | Role |
|---|---|
| `torch` | model, training loop, autograd |
| `torchvision` | not used by the final pipeline (an earlier resize-based fix to `dataset.py` was replaced with a crop/pad approach that doesn't need it -- see `TRAINING_LOG.md`, 2026-10-17) |
| `nibabel` | loading `.nii`/`.nii.gz` MRI slice files |
| `numpy` | array handling outside `modules.py` (explicitly disallowed inside it per the assignment spec) |
| `scikit-image` | `structural_similarity` for the SSIM metric (`utils.py`) |
| `matplotlib` | all training curves and figures (`Agg` backend -- no display on compute nodes) |

> TODO: pin exact versions here (`pip freeze | grep -iE "torch|nibabel|scikit-image|numpy|matplotlib"` from the Rangpur `torch` environment) once collected.

**Reproducibility:** `train.py` sets `torch.manual_seed(args.seed)`
(default 42) before building the model or loaders. Note this does not
guarantee bit-identical runs across different hardware/CUDA versions --
only that re-running on the same Rangpur setup should reproduce the same
training trajectory. Exact run-to-run variance was not separately
measured (would require re-running full 30-epoch training multiple times,
which the compute budget for this project did not allow for).

## Data and Preprocessing

The HipMRI 2D prostate dataset ships pre-split into `keras_slices_train`,
`keras_slices_validate`, and `keras_slices_test` folders of `.nii.gz`
files, 11,460 / 660 / 540 slices respectively. Only raw MRI slices are
used -- `dataset.is_segmentation_file()` explicitly excludes any `seg_`
file, after an early mistake during the dataset audit accidentally loaded
one as if it were an image (see the Feasibility Review above).

**Patient-level splitting.** The dataset ships pre-split, but nothing
guarantees a single patient's slices don't appear in more than one split
by filename coincidence -- and an early bug in `get_patient_id()` (reading
the wrong part of the filename) would have silently allowed exactly that.
Leakage here matters more than usual: a model that has seen *some* of a
patient's slices during training could reconstruct their *other* slices
suspiciously well not because it generalises, but because it has partial
information about that specific patient's anatomy -- which would quietly
inflate every metric in this report and undermine the whole memorisation
investigation. `dataset.verify_patient_level_split()` checks this
explicitly across all three splits and was run live on Rangpur: 31 / 4 / 3
unique patients in train/validate/test, zero overlap.

**Shape normalisation.** Most slices are (256, 128), but a minority
(about 0.5%, 60 out of 11,460 training slices) are (256, 144) -- a
different field of view, discovered when the first full training run
crashed trying to batch mismatched shapes (`TRAINING_LOG.md`, 2026-10-17).
Every slice is forced to exactly (256, 128) via `dataset.pad_or_crop()`:
a centre-crop/zero-pad, deliberately not a resize. Resizing would
interpolate (blur) real pixel intensities before the model or the SSIM
metric ever see them, which would quietly contaminate the fidelity
numbers this whole report is built on; cropping/padding only ever trims
or adds a narrow margin (observed mismatches are at most 16px) without
altering a single surviving pixel's value.

**Normalisation.** Each slice is independently min-max normalised to
[0, 1] (`(x - x.min()) / (x.max() - x.min())`), matching the decoder's
sigmoid output and the MSE reconstruction loss.

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
# (needs outputs/vqvae_best.pt and outputs/autoencoder_best.pt from the
# runs above)
python predict.py

# Dilemma investigation: memorisation audit, latent interpolation,
# failure-case autopsies -- run via Slurm (needs a GPU)
sbatch dilemma.slurm
```

Key `train.py` flags (all have sensible defaults -- see
`build_argparser()`): `--model {vqvae,autoencoder}`, `--epochs`,
`--batch-size`, `--dead-code-threshold` (VQ-VAE dead-code revival
sensitivity, default 1.0), `--verify-split` (run the patient-leakage
check before training starts). Every run writes
`outputs/<model>_metrics.csv` (per-epoch loss, SSIM, perplexity and
codes-revived counts), `outputs/<model>_best.pt` (best-validation-SSIM
checkpoint), and `figures/<model>_*.png` (training curves).

## Results

Both models were trained for 30 epochs on the full dataset (11,460 train /
660 validation / 540 test slices) on a Rangpur A100 GPU. See
[`TRAINING_LOG.md`](TRAINING_LOG.md) for the full debugging history --
three crashes before the first successful full run, then a codebook
collapse and a CUDA memory leak, each diagnosed and fixed with evidence
rather than guessed at.

### Test-set metrics (held-out, never seen during training or checkpoint selection)

| Model | Test SSIM | Test PSNR (dB) | Target | Codebook perplexity (test set) |
|---|---|---|---|---|
| Autoencoder (baseline) | **0.9738** | **36.38** | > 0.6 | n/a (no codebook) |
| VQ-VAE | **0.8090** | **27.60** | > 0.6 | ~360 (effective, out of 512) |

Both models clear the 0.6 feasibility target by a wide margin. The
autoencoder's continuous bottleneck reconstructs almost perfectly;
the VQ-VAE, constrained to describe every slice using a shared 512-entry
"vocabulary", is measurably lossier -- see the reconstruction figure
below, where fine muscle striations are visibly smoothed even though the
main anatomy is preserved. That fidelity gap is not a failure: it is the
VQ-VAE's defining trade-off, and it is the evidence the Investigation
section below builds on to ask whether that lower fidelity corresponds to
less patient-specific memorisation.

### Training curves

**VQ-VAE**

![VQ-VAE training loss](figures/vqvae_loss_curve.png)
![VQ-VAE validation SSIM](figures/vqvae_ssim_curve.png)
![VQ-VAE codebook perplexity](figures/vqvae_perplexity_curve.png)
![VQ-VAE dead-code revivals per epoch](figures/vqvae_revived_curve.png)

Perplexity climbs from 232 to a peak of ~437 within the first 6 epochs
(the dead-code-revival fix seeding the codebook from real encoder
outputs; see `TRAINING_LOG.md`, 2026-10-14 and 2026-10-17 entries), then
drifts gently down to ~370 by epoch 30. That gradual decline is the model
consolidating onto the codes that matter for this dataset, not collapse
-- collapse would mean a crash back toward 1, and 370/512 (~72% of the
codebook) is still healthy, broad usage. `codes_revived` confirms this
directly: it drops from 1836 in epoch 1 to effectively 0 by epoch 5 and
stays there for the remaining 25 epochs, meaning the codebook stabilised
early and never needed rescuing again.

**Autoencoder baseline**

![Autoencoder training loss](figures/autoencoder_loss_curve.png)
![Autoencoder validation SSIM](figures/autoencoder_ssim_curve.png)
![Autoencoder dead-code revivals per epoch](figures/autoencoder_revived_curve.png)

The flat-zero revived-codes curve is expected and confirms the logging
code handles the baseline's codebook-free `IdentityQuantizer` correctly
rather than silently erroring.

### Qualitative reconstructions

![Original vs. Autoencoder vs. VQ-VAE reconstructions on test-set slices](figures/reconstruction_comparison.png)

## Investigation of the Open Research Dilemma

**The dilemma:** the Results section already shows the VQ-VAE is measurably
lossier than the plain autoencoder (test SSIM 0.809 vs. 0.974). Forcing
every image through a finite, shared 512-entry codebook is a real cost.
Is it buying anything back? Specifically: does that forced bottleneck make
the VQ-VAE less able to memorise individual training patients than the
unconstrained autoencoder -- a genuine privacy argument for accepting
lower fidelity -- or is the fidelity simply lost for nothing? This section
gathers evidence rather than assuming an answer either way. All numbers
come from `dilemma.py` (full script in this folder); raw output is in
`outputs/memorisation_audit.csv` and `outputs/test_metrics.csv`
(gitignored alongside the checkpoints -- the numbers below are
transcribed exactly from them).

### Benchmarking table

| Model | Test SSIM | Test PSNR (dB) | Codebook perplexity | Memorisation signal (see below) |
|---|---|---|---|---|
| Autoencoder (baseline) | 0.9738 | 36.38 | n/a | none detected by this probe |
| VQ-VAE | 0.8090 | 27.60 | ~360 effective (of 512) | none detected by this probe |

### Resource profiling

From Rangpur's Slurm accounting (`sacct`) for both full 30-epoch training
runs, one A100 GPU each:

| Model | Job ID | Wall-clock time | Host RAM (MaxRSS) |
|---|---|---|---|
| Autoencoder | 668649 | 00:24:26 | 1,027,500 KB (~1,003 MB) |
| VQ-VAE (fixed) | 671705 | 00:25:22 | 3,008,956 KB (~2,938 MB) |

Standard Slurm accounting (`sacct`'s `MaxRSS`) reports host/CPU memory,
not GPU VRAM, so these numbers are *not* directly comparable to the
35.6GB GPU out-of-memory figure from the pre-fix crash in
`TRAINING_LOG.md` (2026-10-17) -- that figure came from PyTorch's own
CUDA error message, not from the job scheduler. Taken together, though,
they tell a consistent story: the VQ-VAE costs noticeably more host
memory than the autoencoder (~2.9GB vs. ~1.0GB, roughly 3x, consistent
with the extra codebook bookkeeping -- distance computation against 512
codes, EMA buffers, dead-code revival -- that the autoencoder's
`IdentityQuantizer` skips entirely), while wall-clock time is nearly
identical (~25 minutes either way, so the quantisation step itself is
cheap relative to the shared encoder/decoder). Both are comfortably
affordable on a single Rangpur A100 node within the 4-hour Slurm time
limit requested in `train.slurm`.

### Memorisation audit

The most direct check: does either model do suspiciously better on a
random sample of its own training data than on the test patients it never
saw? A random, fixed-size sample of 540 training slices (matching the
test set's size, so the comparison isn't confounded by sample size) was
evaluated the same way as the test set.

| Model | Train-sample SSIM | Test SSIM | Gap (train − test) |
|---|---|---|---|
| VQ-VAE | 0.7898 | 0.8090 | **−0.0193** |
| Autoencoder | 0.9704 | 0.9738 | **−0.0034** |

Both gaps are slightly *negative* -- both models reconstruct the held-out
test set marginally better than a random training sample, not worse. A
positive gap (train performance inflated above test) is the classic
memorisation signature; neither model shows it. Read narrowly, this says
neither model is grossly overfitting to individual training images in a
way that shows up as inflated train-set SSIM.

That said, this is a weak probe, and a weak probe returning a null result
is not strong evidence of safety -- it only means *this particular test*
didn't catch anything. A model could still memorise fine-grained,
patient-identifying texture that doesn't move the aggregate SSIM gap at
all. A proper privacy audit would need a sharper tool (e.g. a
membership-inference-style test, or nearest-training-neighbour distance
correlated with reconstruction error) than the one built here, and that
is flagged explicitly as future work in the recommendation below rather
than glossed over.

**VQ-VAE-specific check: does the codebook itself generalise?** Unlike the
autoencoder, the VQ-VAE's bottleneck is a finite, shared vocabulary learned
from training patients. If describing an unseen test patient required a
visibly different mix of codes than training patients use, that would
suggest the codebook has specialised to specific training images rather
than general anatomy. `dilemma.py` compares the codebook's usage
distribution (what fraction of all encoder positions land on each of the
512 entries) between the training sample and the test set:

![VQ-VAE codebook usage: train sample vs. test](figures/codebook_usage_train_vs_test.png)

The two distributions are nearly identical (total-variation distance =
**0.062**, where 0 = identical and 1 = completely disjoint). The same
roughly 350-400 codes that describe training patients also describe test
patients, in close to the same proportions. This is a genuine point in
the VQ-VAE's favour that the autoencoder has no equivalent of: its
"vocabulary" is shared and demonstrably not patient-specific.

### Latent interpolation

Walking between two test slices in latent space and decoding every step
tests whether the learned representation is smooth and generalisable, or
just a lookup table of memorised points (which would produce garbage
between them):

![Latent interpolation between two test slices, VQ-VAE and autoencoder](figures/latent_interpolation.png)

Both models produce a smooth, anatomically plausible transition -- the
bright circular structure gradually changes shape and the surrounding
tissue morphs continuously, with no sudden jumps or artefacts at any
intermediate step. This held even for the VQ-VAE, which was the more
surprising result going in: interpolating in the *continuous*
pre-quantisation space and re-quantising at every step could plausibly
have produced visible jumps whenever the interpolated point crossed from
one code's territory to another's, but the decoder appears smooth enough
across nearby codes that no such discontinuity is visible. Both models
pass this check; neither shows the "memorised isolated points" failure
mode.

### Failure-case autopsies

The 5 worst test-set reconstructions by SSIM, for each model:

![VQ-VAE worst 5 test reconstructions](figures/vqvae_failure_cases.png)
![Autoencoder worst 5 test reconstructions](figures/autoencoder_failure_cases.png)

**VQ-VAE (worst SSIM 0.729-0.735):** all five worst cases are consecutive
slices (36, 38, 39, 40, 41) from a single patient, `case_042`. This is not
five independent failures -- it's one systematic weak spot. The slices
show denser, more granular internal texture than the simpler
bladder-dominated slices typical elsewhere in the dataset, and the VQ-VAE
visibly over-smooths this region, losing the fine granularity while still
preserving the overall shape and boundaries. This reads as a
*representation* limitation (this particular texture pattern is
under-represented in what the 512-entry codebook learned to describe)
rather than a random failure, which matters for the privacy argument:
a systematic blind spot on one patient's specific anatomy is a
*weaker* fidelity hit than it first looks (it is not reconstructing that
patient any better than it reconstructs anyone else -- if anything,
worse), but it also means the codebook has not specialised to memorise
that patient's detail, consistent with the audit above.

**Autoencoder (worst SSIM 0.962-0.963):** worth stating plainly -- these
are not real failures. The autoencoder's "worst" reconstructions on the
entire test set are still visually almost indistinguishable from the
originals. Its weakest cases (`case_042`/`case_040`, slices 57-59) show
fine muscle striations very slightly softened, nothing more. This itself
is informative for the dilemma: the autoencoder has essentially *no*
regime where it fails to reconstruct a patient with high fidelity, which
is exactly the property that makes its memorisation risk (even though
this audit's SSIM-gap probe didn't detect it directly) a live concern
rather than a hypothetical one -- there's no fidelity ceiling pushing back
against it anywhere in this dataset.

### Recommendation to the project manager

Neither model showed an overt memorisation signature under the tests run
here, so this is not a case of "the VQ-VAE is safe and the autoencoder is
not" -- the evidence doesn't support that strong a claim. What the
evidence does support:

1. The VQ-VAE's codebook is demonstrably shared and general (TV distance
   0.062 between train and test usage), which is a structural privacy
   property the autoencoder cannot offer by construction -- there is
   nothing analogous to audit in a free-form continuous bottleneck.
2. That structural property costs roughly 0.16 SSIM (0.974 -> 0.809) and
   is concentrated as a systematic weak spot on unusual anatomy rather
   than spread evenly, which matters if this model is ever used for
   diagnostic-adjacent purposes rather than only data augmentation.
3. The memorisation probe used here (train-vs-test SSIM gap) is a
   deliberately simple first pass, and a null result from it is weak
   evidence of safety either way.

**Recommendation:** for any deployment where reconstructions, codes, or
derived synthetic images might leave a controlled environment (shared
with collaborators, used to generate a public synthetic dataset, etc.),
prefer the VQ-VAE and treat its lower fidelity as the cost of a real,
structural privacy property rather than a pure downside. For purely
internal use where maximum fidelity matters and nothing derived from the
model ever leaves a trusted environment, the autoencoder's near-perfect
reconstruction is hard to justify giving up. Before trusting either
model's privacy properties for a real deployment, the next step should be
a sharper memorisation test than the one here -- e.g. a
membership-inference-style classifier, or correlating reconstruction
error with each test image's distance to its nearest training neighbour
-- since the simple SSIM-gap test used in this investigation is a
reasonable first check but not a strong enough one to deploy on alone.

## Artificial Intelligence Usage Disclosure

Claude (Anthropic) was used throughout this project as a pair-programming
assistant, in an interactive, conversational workflow rather than a single
one-off generation. The full chronology of what was tried, what broke, and
what was changed is in [`TRAINING_LOG.md`](TRAINING_LOG.md); this section
summarises the role AI played and how its output was checked.

**What it assisted with:**

- Initial design and implementation of `modules.py` (encoder, decoder,
  `VectorQuantizerEMA`, `IdentityQuantizer`), `dataset.py`, and the
  `train.py` training loop, from a plan I had already scoped against the
  assignment spec.
- Diagnosing and fixing six real runtime bugs hit on Rangpur's actual
  GPU cluster, each with a specific technical explanation rather than a
  guess: a `ZeroDivisionError` from `drop_last` dropping the only batch
  on a small dataset; a `RuntimeError` from an in-place ReLU corrupting a
  residual connection's gradient; a `NotImplementedError` from
  `VectorQuantizerEMA.forward` losing its class indentation during a
  manual edit of my own; the codebook collapse (perplexity stuck at 1.0)
  seen in the smoke test, fixed with a dead-code-revival mechanism; a
  `RuntimeError` from a minority of slices being (256, 144) instead of
  (256, 128), only surfacing once real (non-smoke) training ran on the
  full dataset, fixed with `dataset.pad_or_crop()`; and a CUDA
  out-of-memory crash on the first full training run, traced to the EMA
  codebook update silently picking up a live gradient history that grew
  every batch, fixed by wrapping that block in `torch.no_grad()`.
- Writing `predict.py` (originally crashed on a missing `get_dataloaders`
  import; while fixing it, also switched its evaluation from the
  validation split to the held-out test split, and replaced hardcoded
  placeholder SSIM numbers in its comparison figure with real computed
  metrics) and `dilemma.py` (the memorisation audit, latent
  interpolation, and failure-autopsy code), and drafting most of this
  README, `TRAINING_LOG.md`, and the feasibility review.
- Explaining course concepts on request while preparing for the
  compulsory feasibility check-off -- SSIM, the straight-through
  estimator, why the codebook lookup has no gradient, and how the EMA
  update differs from gradient-descent codebook learning -- which I was
  then able to explain back during the consultation itself.

**What I did without AI assistance:** the Rangpur/Slurm environment setup
(partition, account, conda environment activation in `train.slurm`),
several of my own debugging commits during the smoke-test phase (see
`TRAINING_LOG.md` and the git history on the `topic-recognition` branch),
all Git/GitHub/SSH authentication troubleshooting, and every decision
about what to actually submit.

**How outputs were audited:** every code change was syntax-checked
(`python -m py_compile`) before being run, and nothing in this report is
AI-generated numbers or estimates -- every figure and metric quoted here
(SSIM, PSNR, perplexity, the codebook total-variation distance, the
failure-case filenames and scores) was produced by actually running
`train.py`, `predict.py`, or `dilemma.py` on Rangpur's GPU against the
real dataset, and is transcribed from the resulting console output or
CSV files (`outputs/*.csv`, gitignored alongside the checkpoints, but
the exact numbers are quoted directly in this README and in
`TRAINING_LOG.md`). Each proposed bug fix was treated as a hypothesis,
not an answer: it was only accepted once the next real run's behaviour
confirmed it -- e.g. the codebook-collapse fix wasn't considered done
until perplexity was actually observed climbing on real training data,
and the OOM fix wasn't considered done until a full 30-epoch run actually
completed without crashing.
