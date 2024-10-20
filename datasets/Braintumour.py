import os
join = os.path.join

import numpy as np
import torch
from torch.utils.data import Dataset
from Peet4SAM.datasets.build import register
import glob

from uitls import mkdir_if_missing, Datum, read_json, write_json

@register('braintumour')
class Myeloma:
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = join(data_root, "Braintumour")
        self.gt_path = join(self.data_root, "mask")
        self.mask = join(self.data_root, "Tr")
        self.split_path = join(self.data_root, "split_Myeloma.json")
        self.split_fewshot_dir = join(self.data_root, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.gt_path, self.high_path)
        else:
            all_files = os.listdir(self.gt_path)
            train, val, test = self.split_dataset(all_files)
            self.save_split(train, val, test, self.split_path, self.gt_path, self.high_path)

        self.train = train
        self.val = val
        self.test = test

        self.bbox_shift = bbox_shift
        print(f"number of images:{len(self.gt_path_files)}")


   
    
    def split_dataset(all_files, ratios):
        if len(ratios) !=3:
            raise ValueError("The length of ratios should be 3.")
        total_ratio = sum(ratios)
        if total_ratio !=1:
            raise ValueError("The sum of ratios must be equal to 1!")
        
        n = len(all_files)
        results = []
        start = 0
        for i in range(3):
            end = start + int(n * ratios[i])
            results.append(all_files[start:end])
            start = end
        train = results[0]
        val = results[1]
        test = results[2]
        return train, val, test


    
    @staticmethod
    def read_split(filepath, gt_path, high_path):
        def _convert(items):
            out = []
            for item_gt_path, item_high_path in items:
                item_gt_path = join(gt_path, item_gt_path)
                item_high_path = join(high_path, item_high_path)
                item = Datum(gt_path=item_gt_path, high_path=item_high_path)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def save_split(train, val, test, filepath, gt_path, high_path):
        def _extract(items):
            out = []
            # many npy files in each item
            for item in items:
                gt_files = os.listdir(join(gt_path, item))
                for file in gt_files:
                    file_gt_path = join(gt_path, item, file)
                    file_high_path = join(high_path, item, file)
                    out.append((file_gt_path, file_high_path))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")