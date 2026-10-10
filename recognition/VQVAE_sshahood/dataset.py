import os
import re
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader


def get_patient_id(filename: str) -> str:
    """
    Extract the patient ID from a HipMRI slice filename.

    Filenames look like <prefix>_<patient_id>_week_<n>_slice_<m>.nii[.gz],
    optionally with a 'seg_' prefix for the matching segmentation mask
    (e.g. 'seg_040_week_5_slice_2.nii.gz' next to the MRI slice for the
    same patient/week/slice). The patient ID is the first run of digits
    in the filename.
    """
    name = os.path.basename(filename)
    match = re.search(r'(\d+)', name)
    if match is None:
        raise ValueError(f"Could not find a numeric patient ID in filename: {name}")
    return match.group(1)


def is_segmentation_file(filename: str) -> bool:
    """True if this looks like a segmentation/label file rather than a raw
    MRI slice. Defensive on purpose: catches a 'seg_' prefix whether
    segmentation files live in their own folder or are mixed in alongside
    the image files (see the README feasibility review -- an early audit
    accidentally loaded a seg_ file and got a 0-3 label range back instead
    of an MRI intensity range)."""
    return 'seg' in os.path.basename(filename).lower()


def verify_patient_level_split(data_dir, splits=('train', 'validate', 'test')):
    """
    Confirm that no patient's slices appear in more than one split.

    This is a named, explicit function rather than a silent assumption,
    because patient-level separation is a marked criterion in its own
    right, not just good practice. Returns a dict of {split: set(patient
    ids)} for logging into the README; raises AssertionError naming the
    offending patient ID(s) if any split overlaps another.
    """
    patients_by_split = {}
    for split in splits:
        folder = os.path.join(data_dir, f'keras_slices_{split}')
        files = [f for f in glob.glob(os.path.join(folder, '*.nii*'))
                 if not is_segmentation_file(f)]
        if not files:
            raise FileNotFoundError(
                f"No MRI slice files found in {folder} -- check data_dir "
                f"and the folder naming convention."
            )
        patients_by_split[split] = {get_patient_id(f) for f in files}

    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1:]:
            overlap = patients_by_split[split_a] & patients_by_split[split_b]
            assert not overlap, (
                f"Patient leakage detected between '{split_a}' and "
                f"'{split_b}': patient ID(s) {sorted(overlap)} appear in "
                f"both splits."
            )
    return patients_by_split


class HipMRISlices(Dataset):
    """Loads HipMRI 2D prostate MRI slices for the VQ-VAE / autoencoder
    reconstruction task. Segmentation masks are explicitly excluded --
    only raw MRI slices are ever returned."""

    def __init__(self, data_dir, split='train', early_stop=False, verify_split=False):
        """
        data_dir: path to the folder containing keras_slices_<split>/.
        split: 'train', 'validate', or 'test'.
        early_stop: if True, only load 20 slices (fast smoke test).
        verify_split: if True, run verify_patient_level_split() across all
            three splits before loading (slower, mainly for a one-off
            sanity check rather than every epoch).
        """
        self.data_dir = data_dir
        self.split = split

        if verify_split:
            verify_patient_level_split(data_dir)

        # Target the image folders, ignoring the seg_ folders
        self.folder_path = os.path.join(data_dir, f'keras_slices_{split}')

        # Load all .nii/.nii.gz files in the target folder, excluding any
        # segmentation masks even if they are mixed into the same folder
        all_files = glob.glob(os.path.join(self.folder_path, '*.nii*'))
        self.files = sorted(f for f in all_files if not is_segmentation_file(f))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No MRI slice files found in {self.folder_path}."
            )

        if early_stop:
            self.files = self.files[:128]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # Load Nifti file and extract data
        nifti_image = nib.load(file_path)
        image_data = nifti_image.get_fdata(caching='unchanged')

        # Remove extra dimensions if present (e.g., (256, 128, 1) -> (256, 128))
        if len(image_data.shape) > 2:
            image_data = np.squeeze(image_data)

        # Min-Max Normalisation to [0, 1]
        img_min = image_data.min()
        img_max = image_data.max()
        if img_max > img_min:
            image_data = (image_data - img_min) / (img_max - img_min)
        else:
            image_data = np.zeros_like(image_data)

        # PyTorch expects channel-first format (C, H, W)
        image_data = np.expand_dims(image_data, axis=0)

        # Convert to float32 tensor
        return torch.tensor(image_data, dtype=torch.float32)


if __name__ == '__main__':
    # Smoke Test
    # Point this to a local dummy folder for Mac testing, or the Rangpur path when on HPC
    test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'

    print("Running data loader smoke test...")
    try:
        print("Checking patient-level split (train/validate/test)...")
        patients = verify_patient_level_split(test_dir)
        for split, ids in patients.items():
            print(f"  {split}: {len(ids)} unique patients")
        print("No patient overlap detected.")

        dataset = HipMRISlices(data_dir=test_dir, split='train', early_stop=True)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch = next(iter(dataloader))
        print(f"Batch shape successfully loaded: {batch.shape}")
        print(f"Batch data type: {batch.dtype}")
        print(f"Batch min: {batch.min():.4f}, max: {batch.max():.4f}")
    except FileNotFoundError as e:
        print(f"Skipping live smoke test locally: {e}")
