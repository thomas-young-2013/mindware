"""
Yolov3 code borrowed from
https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/datasets.py
"""

import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .base_dl_dataset import DLDataset


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


class ListDataset(Dataset):
    def __init__(self, list_path, classes, img_size=416, augment=True, multiscale=True, normalized_labels=True,
                 mode='fit'):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        if mode != 'test':
            self.label_files = [
                path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]
        self.classes = classes
        self.image_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.image_size - 3 * 32
        self.max_size = self.image_size + 3 * 32
        self.batch_count = 0
        self.mode = mode

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        if self.mode != 'test':
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            targets = None
            if os.path.exists(label_path):
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                # Extract coordinates for unpadded + unscaled image
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                # Returns (x, y, w, h)
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] *= w_factor / padded_w
                boxes[:, 4] *= h_factor / padded_h

                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes

            # Apply augmentations
            if self.augment:
                if np.random.random() < 0.5:
                    img, targets = horisontal_flip(img, targets)

            return img_path, img, targets
        else:
            return img_path, img

    def collate_fn(self, batch):
        if self.mode != 'test':
            paths, imgs, targets = list(zip(*batch))
            # Remove empty placeholder targets
            targets = [boxes for boxes in targets if boxes is not None]
            # Add sample index to targets
            for i, boxes in enumerate(targets):
                boxes[:, 0] = i
            targets = torch.cat(targets, 0)
            # Selects new image size every tenth batch
            if self.multiscale and self.batch_count % 10 == 0:
                self.image_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            # Resize images to input shape
            imgs = torch.stack([resize(img, self.image_size) for img in imgs])
            self.batch_count += 1
            return paths, imgs, targets
        else:
            paths, imgs = list(zip(*batch))
            # Resize images to input shape
            imgs = torch.stack([resize(img, self.image_size) for img in imgs])
            self.batch_count += 1
            return paths, imgs

    def __len__(self):
        return len(self.img_files)


class ODDataset(DLDataset):
    def __init__(self, data_path, image_size=416, augment=True, multiscale=True, normalized_labels=True):
        super().__init__()
        data_config = parse_data_config(data_path)
        self.train_path = data_config["train"]
        self.valid_path = data_config["valid"]
        self.classes = load_classes(data_config["names"])

        self.image_size = image_size
        self.augment = augment
        self.multiscale = multiscale
        self.normlized_labels = normalized_labels

    def load_data(self):
        self.train_dataset = ListDataset(self.train_path, self.classes, self.image_size, self.augment,
                                         self.multiscale,
                                         self.normlized_labels)
        self.val_dataset = ListDataset(self.valid_path, self.classes, self.image_size, augment=False,
                                       multiscale=False)

    def load_test_data(self):
        self.test_dataset = ListDataset(self.test_data_path, self.classes, self.image_size, augment=False,
                                        multiscale=False, mode='test')

    def get_train_samples_num(self):
        if self.train_dataset is None:
            train_dataset = ListDataset(self.train_path, self.classes, self.image_size, self.augment,
                                        self.multiscale,
                                        self.normlized_labels)
            _train_size = len(train_dataset)
        else:
            _train_size = len(self.train_dataset)
        return _train_size
