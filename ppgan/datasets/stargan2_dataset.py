#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import paddle
import os.path

from PIL import Image
from .base_dataset import BaseDataset, get_params, get_transform
from .image_folder import make_dataset

from .builder import DATASETS
from .transforms.builder import build_transforms
from paddle.io import Dataset
import numpy as np
import random
from pathlib import Path
from itertools import chain


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


def _make_ref_dataset(root):
    domains = os.listdir(root)
    fnames, fnames2, labels = [], [], []
    for idx, domain in enumerate(sorted(domains)):
        class_dir = os.path.join(root, domain)
        cls_fnames = listdir(class_dir)
        fnames += cls_fnames
        fnames2 += random.sample(cls_fnames, len(cls_fnames))
        labels += [idx] * len(cls_fnames)
    return list(zip(fnames, fnames2)), labels


def _make_src_dataset(root):
    domains = os.listdir(root)
    fnames, labels = [], []
    for idx, domain in enumerate(sorted(domains)):
        class_dir = os.path.join(root, domain)
        cls_fnames = listdir(class_dir)
        fnames += cls_fnames
        # fnames2 += random.sample(cls_fnames, len(cls_fnames))
        labels += [idx] * len(cls_fnames)
    return fnames, labels


# @DATASETS.register()
# class ReferenceDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.src_samples, self.src_targets = _make_src_dataset(root)
#         self.ref_samples, self.ref_targets = _make_ref_dataset(root)
#         self.transform = transform
#
#     def __getitem__(self, index):
#         src_fname = self.src_samples[index]
#         src_label = self.src_targets[index]
#         ref_fname, ref_fname2 = self.ref_samples[index]
#         ref_label = self.ref_targets[index]
#         src_img = np.array(Image.open(src_fname).convert('RGB'))
#         ref_img = np.array(Image.open(ref_fname).convert('RGB'))
#         ref_img2 = np.array(Image.open(ref_fname2).convert('RGB'))
#
#         # if self.transform is not None:
#         #     img = self.transform(img)
#         #     img2 = self.transform(img2)
#         return [src_img, src_label, ref_img, ref_img2, ref_label]
#
#     def __len__(self):
#         return len(self.targets)

@DATASETS.register()
class StarGAN2Dataset(BaseDataset):
    """
    A dataset class for paired image dataset.
    """

    def __init__(self, cfg):
        """
        Initialize this dataset class.
        Args:
            cfg (dict): configs of datasets.
        """
        BaseDataset.__init__(self, cfg)
        self.src_samples, self.src_targets = _make_src_dataset(cfg.dataroot)
        self.ref_samples, self.ref_targets = _make_ref_dataset(cfg.dataroot)
        self.transforms = build_transforms(cfg.transforms)
        self.latent_dim = cfg.latent_dim
        self.batch_size = cfg.batch_size

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        src_fname = self.src_samples[index]
        src_label = self.src_targets[index]
        ref_fname, ref_fname2 = self.ref_samples[index]
        ref_label = self.ref_targets[index]
        src_img = np.array(Image.open(src_fname).convert('RGB'))
        ref_img = np.array(Image.open(ref_fname).convert('RGB'))
        ref_img2 = np.array(Image.open(ref_fname2).convert('RGB'))

        z_trg = np.random.randn(self.latent_dim)
        z_trg2 = np.random.randn(self.latent_dim)

        if self.transforms:
            src_img = self.transforms(src_img)
            ref_img = self.transforms(ref_img)
            ref_img2 = self.transforms(ref_img2)

        return {
            'x_src': src_img, 'y_src': src_label,
            'x_ref': ref_img, 'x_ref2': ref_img2, "y_ref": ref_label,
            "z_trg": z_trg, "z_trg2": z_trg2
        }

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.src_samples)
