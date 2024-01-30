"""
Module Description:
 
This script ....
 
 
Author:
 
Ahmed Telili
 
Date Created:
 
29 01 2024
 
"""

from dataset.dataset import SuperResolutionDataset
from torch.utils.data import DataLoader

def build_dataset(dataset_config):
    """
    Build the dataset for training and testing.

    Args:
        dataset_config (dict): Configuration for the dataset.
    
    Returns:
        A dictionary containing 'train' and 'test' datasets.
    """

    train_dataset = SuperResolutionDataset(
        hr_root=dataset_config['train']['hr_root'],
        lr_root=dataset_config['train']['lr_root'],
        lr_compression_levels=dataset_config['train']['lr_compression_levels'],
        crop_size=dataset_config['train']['crop_size'],
        transform=dataset_config['train']['transform'],
        mode='train'
        )

    val_dataset = SuperResolutionDataset(
        hr_root=dataset_config['val']['hr_root'],
        lr_root=dataset_config['val']['lr_root'],
        lr_compression_levels=dataset_config['val']['lr_compression_levels'],
        # No crop_size and transform for test mode
        mode='val'
    )

    test_dataset = SuperResolutionDataset(
        hr_root=dataset_config['test']['hr_root'],
        lr_root=dataset_config['test']['lr_root'],
        lr_compression_levels=dataset_config['test']['lr_compression_levels'],
        # No crop_size and transform for test mode
        mode='test'
    )

    # DataLoader for batch processing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=dataset_config['train']['batch_size'], 
        shuffle=dataset_config['train']['shuffle'],
        num_workers=dataset_config['train'].get('num_workers', 4)  
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=dataset_config['val']['batch_size'], 
        shuffle=dataset_config['val']['shuffle'],  # Usually False for validation dataset
        num_workers=dataset_config['val'].get('num_workers', 1)  
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=dataset_config['test']['batch_size'], 
        shuffle=dataset_config['test']['shuffle'],  # Usually False for test dataset
        num_workers=dataset_config['test'].get('num_workers', 1) 
    )

    return {'train': train_loader, 'val':val_loader, 'test': test_loader}
