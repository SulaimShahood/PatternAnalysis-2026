"""
utils.py

Small shared helpers that don't belong in modules.py, dataset.py, train.py
or predict.py individually: evaluation metrics used by both the training
loop (for the per-epoch validation numbers) and predict.py (for the final
reported results and the dilemma investigation).

NumPy is used here deliberately -- it is explicitly allowed everywhere
except modules.py.
"""

import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def to_numpy_image(tensor):
    """(1, H, W) or (C, H, W) torch tensor on any device -> (H, W) numpy
    array. Assumes a single-channel image, which is all this project
    uses."""
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return array


def batch_ssim(preds, targets, data_range=1.0):
    """Mean and standard deviation of SSIM across a batch of images.

    preds, targets: (B, 1, H, W) torch tensors in the same normalised
        space (both in [0, 1], matching the sigmoid output and the min-max
        normalisation used in dataset.py -- comparing images normalised in
        different spaces makes the number meaningless).
    Returns: (mean_ssim, std_ssim, per_image_ssim_list)
    """
    scores = []
    for pred, target in zip(preds, targets):
        pred_img = to_numpy_image(pred)
        target_img = to_numpy_image(target)
        score = sk_ssim(target_img, pred_img, data_range=data_range)
        scores.append(score)
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std()), scores


def batch_psnr(preds, targets, data_range=1.0):
    """Mean and standard deviation of PSNR (dB) across a batch of images,
    in the same normalised space as batch_ssim."""
    scores = []
    for pred, target in zip(preds, targets):
        pred_img = to_numpy_image(pred)
        target_img = to_numpy_image(target)
        mse = np.mean((pred_img - target_img) ** 2)
        if mse == 0:
            scores.append(float('inf'))
        else:
            scores.append(20 * np.log10(data_range) - 10 * np.log10(mse))
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std()), scores
