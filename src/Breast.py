#!/usr/bin/env python
# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load CUB200-2011.

CUB200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>

This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import skimage.io as io
import PIL.Image
from torch.utils import data
import numpy as np
import torch


class BreastCancer(data.Dataset):  # 读数据

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.data_path = root
        self.train = train
        self._transform = transform

        imgs = []
        if self.train:
            self.data = os.path.join(self.data_path, 'train', '1')
            name_path = os.path.join(self.data_path, 'train', 'label', '1zhe_train.txt')
        else:
            self.data = os.path.join(self.data_path, 'train', '1')
            name_path = os.path.join(self.data_path, 'train', 'label', '1zhe_val.txt')
        fh = open(name_path, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((os.path.join(self.data, words[0]), int(words[1])))
        self.imgs = imgs

    def __getitem__(self, index):

        fn, label = self.imgs[index]
        # img = Image.open(fn)
        img = io.imread(fn)
        # img = torch.from_numpy(img).float()
        # img = torch.transpose(img, 2, 0)
        img = PIL.Image.fromarray(img)
        if self._transform is not None:
            img = self._transform(img)
        # img.resize_(460, 460, 3)
        return img, label

    def __len__(self):
        return len(self.imgs)


class BreastCancerReLU(torch.utils.data.Dataset):
    """BreakHis relu5-3 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _train_data, list<torch.Tensor>.
        _train_labels, list<int>.
        _test_data, list<torch.Tensor>.
        _test_labels, list<int>.
    """
    def __init__(self, root, train=True):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train

        if self._checkIntegrity():
            print('BreakHis relu5-3 features already prepared.')
        else:
            raise RuntimeError('BreakHis relu5-3 Dataset not found.'
                'You need to prepare it in advance.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = torch.load(
                os.path.join(self._root, 'feature', 'train.pth'))
            assert (len(self._train_data) == 27132
                    and len(self._train_labels) == 27132)
        else:
            self._test_data, self._test_labels = torch.load(
                os.path.join(self._root, 'feature', 'test.pth'))
            assert (len(self._test_data) == 6783
                    and len(self._test_labels) == 6783)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            feature, torch.Tensor: relu5-3 feature of the given index.
            target, int: target of the given index.
        """
        if self._train:
            return self._train_data[index], self._train_labels[index]
        return self._test_data[index], self._test_labels[index]

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
            os.path.isfile(os.path.join(self._root, 'feature', 'train.pth'))
            and os.path.isfile(os.path.join(self._root, 'feature', 'test.pth')))
