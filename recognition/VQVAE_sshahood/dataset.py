import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class HipMRISlices(Dataset):
    def __init__(self, data_dir, split='train', early_stop=False):
        """
        Loads HipMRI 2D slices. 
        split: 'train', 'validate', or 'test'
        """
        self.data_dir = data_dir
        self.split = split
        # Target the image folders, ignoring the seg_ folders
        self.folder_path = os.path.join(data_dir, f'keras_slices_{split}')
        
        # Load all .nii or .nii.gz files in the target folder
        self.files = glob.glob(os.path.join(self.folder_path, '*.nii*'))
        
        if early_stop:
            self.files = self.files[:20]
            
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
        dataset = HipMRISlices(data_dir=test_dir, split='train', early_stop=True)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        batch = next(iter(dataloader))
        print(f"Batch shape successfully loaded: {batch.shape}")
        print(f"Batch data type: {batch.dtype}")
        print(f"Batch min: {batch.min():.4f}, max: {batch.max():.4f}")
    except FileNotFoundError:
        print(f"Skipping live smoke test locally. Path {test_dir} not found on this machine.")