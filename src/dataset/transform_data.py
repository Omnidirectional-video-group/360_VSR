"""
Author:
    Ahmed Telili, Research Engineer @ Technology Innovation Institute (TII)

Description:
    This script, developed and maintained by Ahmed Telili, is part of a challenge focused on super-resolution sponsored by TII. For any queries or contributions regarding this script, please contact ahmed.telili@tii.ae.
"""


import torch
import random
import torchvision.transforms.functional as TF




class NormalizePair(object):
    """
    Normalize a pair of images (input and target) with mean and standard deviation.
    """

    def __init__(self, input_mean=0., input_std=1., target_mean=0., target_std=1.):
        """
        Args:
            input_mean (float or list): Mean for the input image normalization.
            input_std (float or list): Standard deviation for the input image normalization.
            target_mean (float or list): Mean for the target image normalization.
            target_std (float or list): Standard deviation for the target image normalization.
        """
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, input_img, target_img):
        """
        Args:
            input_img (Tensor): Tensor image of size (C, H, W) to be normalized.
            target_img (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tuple: Normalized input and target images.
        """
        input_img = self.normalize(input_img, self.input_mean, self.input_std)
        target_img = self.normalize(target_img, self.target_mean, self.target_std)
        return input_img, target_img

    def normalize(self, img, mean, std):
        """
        Normalize a tensor image with mean and standard deviation.
        """
        # The following line adjusts dimensions of mean and std if they are scalars.
        mean = mean if isinstance(mean, (list, tuple)) else [mean]
        std = std if isinstance(std, (list, tuple)) else [std]
        
        mean = torch.as_tensor(mean, dtype=torch.float32, device=img.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=img.device)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])
        return img


class RandomHorizontalFlipPair():
    """Horizontally flip the given image pair randomly with a probability of 0.5."""
    def __call__(self, input_img, target_img):
        if random.random() > 0.5:
            return TF.hflip(input_img), TF.hflip(target_img)
        return input_img, target_img

class RandomVerticalFlipPair():
    """Vertically flip the given image pair randomly with a probability of 0.5."""
    def __call__(self, input_img, target_img):
        if random.random() > 0.5:
            return TF.vflip(input_img), TF.vflip(target_img)
        return input_img, target_img

class RandomRot90Pair():
    """Rotate the given image pair by 90 degrees randomly."""
    def __init__(self, times=1):
        self.times = times

    def __call__(self, input_img, target_img):
        if random.random() > 0.5:
            return TF.rotate(input_img, 90 * self.times), TF.rotate(target_img, 90 * self.times)
        return input_img, target_img