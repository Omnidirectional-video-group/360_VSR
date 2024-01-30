"""
Author:
    Ahmed Telili, Research Engineer @ Technology Innovation Institute (TII)

Description:
    This script, developed and maintained by Ahmed Telili, is part of a challenge focused on super-resolution sponsored by TII. For any queries or contributions regarding this script, please contact ahmed.telili@tii.ae.
"""


import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from dataset.transform_data import *

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_root, lr_root, lr_compression_levels, crop_size=None, transform=None, mode='train'):
        """
        Args:
            hr_root (string): Directory with all the high-resolution images.
            lr_root (string): Directory with all the low-resolution images.
            lr_compression_levels (list of strings): List of sub-directories for different compression levels.
            crop_size (int): The size of the crop to be applied to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.hr_root = hr_root
        self.lr_root = lr_root
        self.lr_compression_levels = lr_compression_levels
        self.crop_size = crop_size
        self.transform = transform
        self.mode = mode
        self.hr_image_folders = sorted(os.listdir(hr_root))
        self.image_pairs = self._create_image_pairs()

    def _create_image_pairs(self):
        image_pairs = []
        for hr_folder in self.hr_image_folders:
            hr_folder_path = os.path.join(self.hr_root, hr_folder)
            hr_images = sorted(os.listdir(hr_folder_path))


            if self.mode != 'train':
                # For validation and test, select only every 10th image
                hr_images = hr_images[::10]
    
            for hr_image in hr_images:
                for compression_level in self.lr_compression_levels:
                    lr_folder_path = os.path.join(self.lr_root, compression_level, hr_folder)
                    lr_image_path = os.path.join(lr_folder_path, hr_image)
                    hr_image_path = os.path.join(hr_folder_path, hr_image)
                    image_pairs.append((hr_image_path, lr_image_path))

        if self.mode == 'train':
            random.shuffle(image_pairs)
        else:
            # Optionally sort the image pairs for test mode
            image_pairs.sort()
            
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):

        hr_image_path, lr_image_path = self.image_pairs[idx]

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        if self.mode == 'train':
            scale_factor = 4  # Adjust scale factor as needed
            hr_crop_size = self.crop_size * scale_factor
            i, j, h, w = transforms.RandomCrop.get_params(lr_image, output_size=(self.crop_size, self.crop_size))
            lr_image = TF.crop(lr_image, i, j, h, w)
            hr_image = TF.crop(hr_image, i * scale_factor, j * scale_factor, hr_crop_size, hr_crop_size)
            if self.transform:
                # List of transformation functions
                transformations = [
                    RandomHorizontalFlipPair(),
                    RandomVerticalFlipPair(),
                    RandomRot90Pair(times=1)  # Example: rotate once
                ]

                # Randomly select one transformation and apply it
                transform_op = random.choice(transformations)
                lr_image, hr_image = transform_op(lr_image, hr_image)


        hr_image = TF.to_tensor(hr_image)
        lr_image = TF.to_tensor(lr_image)

        normalize_pair = NormalizePair()
        lr_image, hr_image = normalize_pair(lr_image, hr_image)

        return lr_image, hr_image

def custom_collate(batch):
    # 'batch' is a list of tuples: [(lr_img_seq1, hr_img_seq1), (lr_img_seq2, hr_img_seq2), ...]
    
    # Separate LR and HR sequences
    lr_sequences, hr_sequences = zip(*batch)

    # Pad sequences to the same length
    # Note: You'll need to decide how to pad them, this is just an example
    lr_sequences_padded = [pad_sequence(seq, pad_size) for seq in lr_sequences]
    hr_sequences_padded = [pad_sequence(seq, pad_size) for seq in hr_sequences]

    # Stack sequences into tensors
    lr_batch = torch.stack(lr_sequences_padded, dim=0)
    hr_batch = torch.stack(hr_sequences_padded, dim=0)

    return lr_batch, hr_batch
