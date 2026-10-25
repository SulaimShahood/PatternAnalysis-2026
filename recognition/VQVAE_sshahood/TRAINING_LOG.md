# Training Log

A chronological record of every run attempted for this project, what
happened, why, and what was changed in response. This exists because the
assessment (and our tutor, at the feasibility check-off) asks for an
honest account of debugging effort, not just a final number -- if the
model never reaches the SSIM target, this log is the evidence that we
tried, diagnosed why, and made a documented decision about it.

Each entry: **what we ran -> what happened -> why -> what we changed.**
Newest entries are added at the bottom. Metrics referenced here come from
`outputs/<model>_metrics.csv` and `figures/*.png`, written by `train.py`
every run.

---

## 2026-10-10 -- Data audit

**Ran:** `python dataset.py` (patient-level split verification) on Rangpur.

**Result:** Two real bugs found and fixed before any training:
1. Patient ID was parsed incorrectly from filenames, which could have let
   the same patient's slices appear in more than one of train/validation/
   test (data leakage -- the model could "cheat" by memorising a patient
   it had already seen).
2. A segmentation mask file was briefly being loaded as if it were an MRI
   intensity image.

**Fix verified:** re-ran the split check live -- 31 / 4 / 3 patients across
train / validation / test, zero overlap. 11,460 / 660 / 540 slices total.

---

## 2026-10-12 -- Smoke test, attempt 1: `ZeroDivisionError`

**Ran:** `python train.py --model vqvae --smoke` on Rangpur.

**Result:** Crashed with `ZeroDivisionError` while computing the average
loss for the epoch (division by `n_batches`, which was 0).

**Why:** the `DataLoader` was created with `drop_last=True`. On the smoke
subset (20 slices) with `batch_size=32`, that one partial batch was the
*only* batch, so dropping it left the loader producing zero batches total
for that epoch.

**Fix:** set `drop_last=False` in `train.py` (there's no BatchNorm in the
model, so nothing needs a fixed batch size), and added
`if args.smoke: args.batch_size = min(args.batch_size, 4)` so the smoke
path always has multiple batches regardless of how many slices it uses.

---

## 2026-10-12 -- Smoke test, attempt 2: in-place autograd error

**Ran:** `python train.py --model vqvae --smoke` on Rangpur (after the
`drop_last` fix).

**Result:** Crashed with a `RuntimeError` about an in-place operation
having modified a tensor needed for gradient computation.

**Why:** `ResidualBlock.forward` computes `x + self.block(x)`, and the
first layer inside `self.block` was `nn.ReLU(inplace=True)`. That
in-place ReLU was overwriting `x` itself before the addition could use
the original value, which autograd correctly refused to allow once it
tried to backprop through it.

**Fix:** changed every `nn.ReLU(inplace=True)` to `inplace=False` in
`modules.py` (4 occurrences, across `Encoder`, `Decoder`, and
`ResidualBlock`).

---

## 2026-10-12 -- Smoke test, attempt 3: missing `forward`

**Ran:** `python train.py --model vqvae --smoke` on Rangpur.

**Result:** Crashed with `NotImplementedError: Module [VectorQuantizerEMA]
is missing the required "forward" function`.

**Why:** an earlier manual edit had accidentally de-indented
`def forward(self, z_e):` from a class method (4-space indent) to
module level (0-space indent). Still valid Python syntax (so
`py_compile` passed), but it meant the function no longer belonged to the
`VectorQuantizerEMA` class, so PyTorch's `nn.Module` fell back to its
default `forward`, which just raises.

**Fix:** re-indented `forward` back under the class in `modules.py`.

---

## 2026-10-12 -- Smoke test, attempt 4: runs end-to-end, but codebook collapses

**Ran:** `python train.py --model vqvae --smoke` (2 epochs, 20 slices) on
Rangpur's GPU.

**Result:** Completed without crashing -- data loaded, both epochs ran,
loss decreased, a checkpoint was saved, metrics were written to CSV. But
**codebook perplexity was stuck at 1.0 on both epochs.** Perplexity equals
`num_embeddings` (512) if every code is used equally often, and collapses
toward 1 if the encoder is only ever being matched to one single code.
1.0 is total collapse: effectively the whole 512-entry codebook had
shrunk to one usable entry.

**Why (diagnosis):** the codebook starts as 512 random vectors
(`torch.randn`), completely disconnected from what the encoder actually
outputs. Very early in training, whichever random code happens to be
closest to most encoder outputs "wins" almost every nearest-neighbour
lookup. Codes that are never chosen never appear in the EMA update, so
they never get a chance to move towards anything useful -- they just stay
as useless random noise forever while everything collapses onto the one
or two codes closest at initialisation. This is the textbook VQ-VAE
failure mode, and exactly the risk our feasibility review flagged.

**Fix decided (implemented 2026-10-14, see below):** dead-code revival --
periodically reset any codebook entry whose EMA usage has stayed below a
small threshold to a real encoder output actually seen in the current
batch, instead of leaving it as untouched random noise.

---

## 2026-10-14 -- Dead-code revival implemented

**Changed:** `modules.py`, `VectorQuantizerEMA.forward()`. Immediately
after the existing EMA codebook update, any code whose `ema_cluster_size`
has stayed below `dead_code_threshold` (default 1.0) is reset: its
embedding and EMA accumulator are overwritten with a real encoder output
vector sampled (with replacement) from the current batch, and its EMA
usage count is reset to 1.0 so it isn't immediately flagged dead again.
`self.last_num_revived` records how many codes were reset on each
forward call, threaded through to `VQVAE` as a `dead_code_threshold`
constructor argument (also exposed as `--dead-code-threshold` on the
command line).

**Also changed:** `train.py` now sums `quantizer.last_num_revived` over
every training batch each epoch, prints it alongside the other metrics
(`codes_revived=...`), writes it as a `codes_revived` column in
`outputs/<model>_metrics.csv`, and plots it as
`figures/<model>_revived_curve.png`.

**Expected effect:** because the EMA usage buffer starts at all zeros,
every code will be "dead" and get reset to a real encoder output on the
very first training batch -- which replaces the useless random
initialisation with codes seeded from real data before training proper
even begins (a standard codebook-init trick, not a bug). After that, only
genuinely under-used codes should keep getting reset.

**Known limitation to watch for:** because EMA usage decays every batch
(`decay=0.99`), a code used only occasionally can hover right around
`dead_code_threshold` and get reset repeatedly rather than settling. If
perplexity still looks collapsed after this fix, or `codes_revived` stays
suspiciously high deep into training, next things to try in order: (1)
lower `dead_code_threshold`, (2) only check for dead codes every N
batches, (3) raise `commitment_cost`, (4) lower the learning rate.

**Status at the time:** written and syntax-checked, not yet run (no GPU
in the environment this was written in). Queued behind: committing and
re-running the smoke test on Rangpur.

---

## 2026-10-17 -- First real (non-smoke) training run: crash on uneven slice sizes

**Ran:** `sbatch train.slurm vqvae` and `sbatch train.slurm autoencoder`
on Rangpur -- the first attempt at *real* training (all 11,460 train /
660 validation slices, not the 128-slice smoke subset), after the
dead-code-revival fix above.

**Result:** Both jobs loaded the full dataset fine (`Train slices: 11460,
validation slices: 660` printed for both), then both crashed partway
through the very first training epoch with:

```
RuntimeError: stack expects each tensor to be equal size,
but got [1, 256, 128] at entry 0 and [1, 256, 144] at entry 21
```

**Why:** the smoke test never hit this because it only ever looked at the
first 128 files in the (alphabetically) sorted file list, and those
happened to all be the common (256, 128) shape. Scanning further into the
real 11,460-slice dataset turns up slices from a minority of patients
that are (256, 144) instead -- a different field of view from a different
scanner/protocol. `dataset.py` was assuming every slice was already the
same shape and had no code to enforce it, so the moment a batch contained
a mix of the two shapes, PyTorch's default batch-collation (`torch.stack`)
had no way to stack them into one tensor and raised.

**Also seen in the same run (not the actual crash cause):** both `.err`
files also show `EnvironmentNameNotFound: Could not find conda
environment: vqvae` right at the top, before the real traceback. This
looked alarming at first glance but is not what crashed the job --
`train.slurm` itself correctly runs `conda activate torch` (checked; no
`vqvae` environment is referenced anywhere in it), and the traceback
confirms `torch`, `nibabel` etc. were all successfully imported from
`~/miniconda3/envs/torch/...`, i.e. the right environment was active for
the whole run. This stray message is harmless noise, almost certainly
from something in the login shell's own startup file (e.g. `.bashrc`)
trying to auto-activate an environment called `vqvae` that doesn't exist
on this account, unrelated to this script. Left alone for now since it
doesn't affect correctness; worth a 30-second look in `~/.bashrc` later
if the noise is annoying, but not before the deadline.

**Fix:** added `pad_or_crop()` to `dataset.py` and apply it in
`HipMRISlices.__getitem__` to every slice that isn't already exactly
(256, 128). Deliberately a centre-crop/zero-pad, not a resize/
interpolation -- interpolating would blur real pixel intensities before
the model or the SSIM metric ever sees them, which would quietly
contaminate the fidelity comparison this whole project is measuring.
Observed mismatches are only ever off by 16px in width, so this only ever
trims or pads a narrow margin. Also added a `num_shape_fixed` counter on
the dataset and a one-line print in `train.py` after epoch 1
(`Shape fixups: X/11460 train slices, Y/660 validation slices needed
pad_or_crop`) so the actual prevalence of this is on the record rather
than silently patched over.

**Status:** code written and syntax-checked, not yet re-run on Rangpur.
**Next step: push, pull on Rangpur, resubmit both jobs, confirm they
get past epoch 1 this time, and watch `val_perplexity` / `codes_revived`
to confirm the dead-code fix is also doing its job on the real dataset.**

---

## 2026-10-17 -- First real training run, part 2: autoencoder succeeds, VQVAE hits CUDA OOM

**Ran:** the same `sbatch train.slurm vqvae` / `sbatch train.slurm
autoencoder` from the previous entry, after the `pad_or_crop` fix.

**Result -- autoencoder baseline:** ran all 30 epochs to completion.
`Shape fixups: 60/11460 train slices, 0/660 validation slices needed
pad_or_crop` confirms the size-mismatch fix is working and shows exactly
how common the bug was (60 out of 11,460 -- about 0.5%). Final checkpoint
at epoch 30: **best validation SSIM = 0.9696.** Comfortably clears the
0.6 feasibility target -- this is the working fallback/baseline the
feasibility review promised.

**Result -- VQVAE, the good news first:** the dead-code-revival fix from
2026-10-14 is demonstrably working on real data, not just in theory:

| epoch | val_ssim | val_perplexity | codes_revived |
|-------|----------|----------------|----------------|
| 1     | 0.5377   | 232.6          | 1836           |
| 2     | 0.6730   | 335.7          | 189            |
| 3     | 0.6892   | 414.2          | 36             |

Perplexity climbing from 232 to 414 (out of a possible 512) and
`codes_revived` collapsing from 1836 down to 36 within 3 epochs is
exactly the predicted signature: nearly every code looked "dead" against
the all-zero initial EMA usage count on epoch 1 (expected -- that's the
real-data codebook seeding effect noted in the 10-14 entry), and by epoch
3 the codebook has mostly settled onto real, varied usage. Validation
SSIM was already at 0.6892 by epoch 3, above target. **The codebook
collapse problem is fixed.**

**Result -- VQVAE, the bad news:** training crashed partway through
(after epoch 3, before finishing) with `torch.OutOfMemoryError: CUDA out
of memory`, having allocated **35.6 GB** of a 39.49 GB GPU -- not
fragmentation (the error breaks out "35.60 GiB is allocated by PyTorch"
specifically, as actually-live memory, vs. only 3.39 GiB "reserved but
unallocated"). A model this size (hidden_channels=128, batch_size=32,
256x128 images) has no legitimate reason to need anywhere near that much
memory, so this is a genuine leak, not just "the model is big."

**Why:** found in `modules.py`, `VectorQuantizerEMA.forward()`. The EMA
buffer-update block was never wrapped in `torch.no_grad()`, despite its
own comment claiming "no gradient involved." `dw = encodings.t() @ flat`
uses `flat` (the encoder's output, which *does* require grad for
backprop), so `dw` silently inherits `requires_grad=True`. That then
"infects" `self.ema_w` and `self.embedding` -- both persistent buffers --
through the in-place updates (`.add_(dw, ...)`, `.copy_(...)`), attaching
a live autograd history to them. Because these buffers survive for the
*entire* training run rather than being freed after each batch (unlike
ordinary activations), that attached history grew by one more linked
batch every single step. A 2-epoch smoke test on 20-128 slices never ran
enough batches for this to visibly matter; a real 30-epoch run over
11,460 slices (~358 batches/epoch) does, reliably, around epoch 4.

**Fix:** wrapped the entire EMA-update-and-dead-code-revival block in
`with torch.no_grad():`. None of it needs to be differentiable -- the
encoder's actual gradient signal comes through the straight-through
estimator path later in `forward()`, not through this bookkeeping.

**Status:** fix written and syntax-checked, not yet re-run. **Next step:
push, pull on Rangpur, resubmit `sbatch train.slurm vqvae` (the
autoencoder baseline is already done, no need to rerun it), and confirm
it completes all 30 epochs without an OOM this time, with perplexity
staying healthy throughout -- not just for the first 3 epochs.**

---

## 2026-10-25 -- predict.py: fixed import crash, switched to the test split, dropped hardcoded numbers

**Ran:** `python predict.py` on a Rangpur CPU node (`srun --partition=cpu`).

**Result:** Crashed immediately with
`ImportError: cannot import name 'get_dataloaders' from 'dataset'` --
`predict.py` called a helper function that `dataset.py` never defined.

**Fix:** added `get_dataloaders(data_dir, batch_size, split_for_eval)` to
`dataset.py`, mirroring the exact `DataLoader` construction `train.py`
already uses (`drop_last=False`, shuffle only for the training split).

**Also changed while fixing it (not just the crash):**
1. The original draft evaluated on the **validation** split. Switched to
   **test**: validation already influenced which checkpoint got saved
   (`train.py` keeps the best-val-SSIM epoch), so it's no longer a clean
   held-out measurement -- test is the one split that never touched
   training or model selection at all.
2. The comparison figure's subplot titles were hardcoded guesses
   (`"SSIM: 0.96"`, `"SSIM: 0.77"`) rather than computed values.
   `predict.py` now runs every model over the *entire* test set with
   `utils.batch_ssim` / `batch_psnr` (the same metrics `train.py` already
   uses for validation), reports the real mean SSIM/PSNR/perplexity for
   both models, and writes them to `outputs/test_metrics.csv` -- this is
   the source for the README's Results table and the dilemma
   investigation's benchmarking table, so it needs to be the real number,
   not a placeholder.

**Status:** written and syntax-checked (`py_compile`), not yet run.
**Next step: push, pull on Rangpur, and run `python predict.py` once the
re-run VQVAE training (previous entry) has finished and
`outputs/vqvae_best.pt` exists.**

---

<!--
Add new entries below this line as more runs happen, oldest to newest.
Each entry should say: what command was run, what happened (quote the
actual numbers from the CSV/console), why (the diagnosis), and what -- if
anything -- was changed as a result. If a run is simply a successful full
training run with nothing to debug, that's still worth an entry: record
the final best validation SSIM/perplexity and where the checkpoint and
figures were written.
-->
