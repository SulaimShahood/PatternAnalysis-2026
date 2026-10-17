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
**Next step (urgent -- deadline is 2026-10-18): push, pull on Rangpur,
resubmit both jobs, confirm they get past epoch 1 this time, and watch
`val_perplexity` / `codes_revived` to confirm the dead-code fix is also
doing its job on the real dataset.**

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
