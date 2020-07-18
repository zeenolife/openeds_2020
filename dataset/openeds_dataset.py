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


class OpenEDSDatasetTest(Dataset):
    def __init__(self, data_path, save_path, transforms=None, normalize=None, num_classes=4):
        super().__init__()
        self.data_path = data_path
        self.save_path = save_path

        labels_file = os.path.join(data_path, 'labels.txt')
        with open(labels_file) as f:
            labels = [_.strip() for _ in f.readlines()]
            labels = [_.replace('.npy', '.png').replace('label_', '') for _ in labels]
            labels = set(labels)

        images_file = os.path.join(data_path, 'images.txt')
        with open(images_file) as f:
            self.names = [_.strip() for _ in f.readlines()]
            self.names = [_.replace('.png', '.npy') for _ in self.names if _ not in labels]

        self.init_save_dirs()

        self.num_classes = num_classes
        self.normalize = normalize
        self.transforms = transforms

    def init_save_dirs(self):

        with open(os.path.join(self.save_path, 'output.txt'), 'w') as f:
            for img_name in self.names:
                f.write('{}\n'.format(img_name))

        participant_dirs = set([_.split('/')[0] for _ in self.names])
        for participant_dir in participant_dirs:
            os.makedirs(os.path.join(self.save_path, participant_dir), exist_ok=True)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        # Read image
        image_path = os.path.join(self.data_path, self.names[idx])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.copyMakeBorder(image, 8, 8, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # Apply augmentations
        sample = self.transforms(image=image)

        # Pack into container
        sample['img_name'] = self.names[idx]
        sample['image'] = img_to_tensor(np.ascontiguousarray(sample['image']), self.normalize)

        return sample

