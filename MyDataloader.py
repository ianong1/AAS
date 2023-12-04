#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from glob import glob
from torch.utils.data.dataset import Dataset
import random



class MyDataset(Dataset):
    def __init__(self, cover_dir, stego_dir,partition, transform=None):
        random.seed(2023)

        self.transform = transform

        self.cover_dir = cover_dir
        self.stego_dir = stego_dir

        self.covers_list_all = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        random.shuffle(self.covers_list_all)
        if (partition == 0):
            self.cover_list = self.covers_list_all[:4000]
            self.cover_paths= [os.path.join(self.cover_dir, x) for x in  self.cover_list]
            # self.cover_paths_2 = [os.path.join(self.cover_dir_2, x) for x in self.cover_list]

            self.cover_paths = self.cover_paths 
            #print (self.cover_paths_3)
            #print self.cover_paths
            self.stego_paths = [os.path.join(self.stego_dir, x) for x in self.cover_list]
            # self.stego_paths_2 = [os.path.join(self.stego_dir_2, x) for x in self.cover_list]

            self.stego_paths = self.stego_paths

            self.cover_steg = list(zip(self.cover_paths, self.stego_paths))
            random.shuffle(self.cover_steg)
            self.cover_paths, self.stego_paths = zip(*self.cover_steg)


        if (partition == 1):
            self.cover_list = self.covers_list_all[4000:5000]
            self.cover_paths = [os.path.join(self.cover_dir, x) for x in self.cover_list]
            self.stego_paths = [os.path.join(self.stego_dir, x) for x in self.cover_list]
        if (partition == 2):
            self.cover_list = self.covers_list_all[5000:]
            self.cover_paths = [os.path.join(self.cover_dir, x) for x in self.cover_list]
            self.stego_paths = [os.path.join(self.stego_dir, x) for x in self.cover_list]

        assert len(self.cover_paths) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_paths)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = self.cover_paths[file_index]
        stego_path = self.stego_paths[file_index]
        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)
        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

