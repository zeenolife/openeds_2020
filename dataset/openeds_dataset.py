import os
import cv2
import numpy as np
import torch
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset


class OpenEDSDataset(Dataset):
    def __init__(self, data_path, fold_idx, mode, transforms=None, normalize=None, num_classes=4):
        super().__init__()
        self.data_path = data_path

        assert mode in ('train', 'val'), 'mode should be either "train" or "val"'
        labels_file = os.path.join(data_path, 'fold_{}_{}.txt'.format(fold_idx, mode))
        with open(labels_file) as f:
            self.names = [_.strip() for _ in f.readlines()]

        self.num_classes = num_classes
        self.normalize = normalize
        self.transforms = transforms

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        # Read mask and image
        mask_path = os.path.join(self.data_path, self.names[idx])
        image_path = mask_path.replace('/label_', '/').replace('.npy', '.png')
        mask_orig = np.load(mask_path)
        mask_orig = cv2.copyMakeBorder(mask_orig, 8, 8, 0, 0, cv2.BORDER_CONSTANT, value=0)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.copyMakeBorder(image, 8, 8, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # Apply augmentations
        sample = self.transforms(image=image, mask=mask_orig)

        # Transform single-channel mask to multi-channel
        mask = np.zeros((self.num_classes, *sample["mask"].shape[:2]))
        for i in range(self.num_classes):
            mask[i, sample["mask"] == i] = 1

        # Pack into container
        sample['img_name'] = self.names[idx].replace('/', '_').replace('.npy', '').replace('label_', '')
        sample['mask_orig'] = sample['mask']
        sample['mask'] = torch.from_numpy(np.ascontiguousarray(mask)).float()
        sample['image'] = img_to_tensor(np.ascontiguousarray(sample['image']), self.normalize)

        return sample

