import os
join = os.path.join

import numpy as np
import torch
from torch.utils.data import Dataset
from Peet4SAM.datasets.build import register
import glob

from uitls import mkdir_if_missing, Datum, read_json, write_json

@register('myeloma')
class Myeloma:
    def __init__(self, data_root, ratios=[0.7, 0.1, 0.2], bbox_shift=20):
        self.data_root = join(data_root, "Myeloma")
        self.gt_path = join(self.data_root, "mask")
        self.high_path = join(self.data_root, "high")
        self.low_path = join(self.data_root, "low")
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


    # def __len__(self):
    #     return len(self.gt_path_files)

    # def __getitem__(self, index):
    #     # load npy files(还是应该先把nii.gz文件全部都拆成npy文件才行)
    #     img_name = os.path.basename(self.gt_path_files[index])
        
    #     high_img = np.load(
    #         join(self.high_path, img_name), "r", allow_pickle=True
    #     ) 
    #     # (1024, 1024, 3)
    #     # convert the shape to (3, H, W)
    #     high_img = np.transpose(high_img, (2, 0, 1))

    #     low_img = np.load(
    #         join(self.low_path, img_name), "r", allow_pickle=True
    #     )
    #     low_img = np.transpose(low_img, (2, 0, 1))
    #     assert (
    #         np.max(high_img) <= 1.0 and np.min(high_img) >= 0.0
    #     ), "image should be normalized to [0, 1]"
    #     assert (
    #         np.max(low_img) <= 1.0 and np.min(low_img) >= 0.0
    #     ),"image should be normalized to [0, 1]"

    #     gt = np.load(
    #         self.gt_path_files[index], "r", allow_pickle=True
    #     ) # multiple labels [0, 1,4,5...], (256,256)
    #     assert img_name == os.path.basename(self.gt_path_files[index]), (
    #         "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
    #     )
    #     label_ids = np.unique(gt)
    #     assert (len(label_ids)==2), "multi label in mask"
    #     assert np.max(gt) == 1 and np.min(gt) == 0.0, "ground truth should be 0, 1"

    #     return {
    #         "image": torch.tensor(high_img).float(),
    #         "low_img": torch.tensor(low_img).float(),
    #         "gt": torch.tensor(gt[None, :, :]).long(),
    #         "name": img_name,
    #         "original_size": high_img.shape,
    #     }
    
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