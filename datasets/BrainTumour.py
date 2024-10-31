import os
join = os.path.join

import numpy as np
import SimpleITK as sitk 
from skimage import transform
import pickle

from datasets import register


from .uitls import mkdir_if_missing, read_json, write_json

@register('braintumour')
class Myeloma:
    def __init__(self, data_root, ratios=[0.7, 0.1, 0.2], bbox_shift=20):
        self.data_root = join(data_root, "BrainTumour")
        self.gt_path = join(self.data_root, "labelsTr")
        self.img_path = join(self.data_root, "imagesTr")
        self.split_path = join(self.data_root, "split_BrainTumour.json")
        self.split_fewshot_dir = join(self.data_root, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.gt_path, self.img_path)
        else:
            all_files = os.listdir(self.gt_path)
            train, val, test = self.split_dataset(all_files, ratios)
            self.save_split(train, val, test, self.data_root, self.split_path, self.gt_path, self.img_path)

        self.train = train
        self.val = val
        self.test = test
        self.mode = 'npy'

        self.bbox_shift = bbox_shift
        # print(f"number of images:{len(self.gt_path_files)}")

   
    def split_dataset(self, all_files, ratios):
        if len(ratios) !=3:
            raise ValueError("The length of ratios should be 3.")
        total_ratio = sum(ratios)
        # if total_ratio != 1.0:
        #     raise ValueError("The sum of ratios must be equal to 1!")
        
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
                # item = Datum(gt_path=item_gt_path, high_path=item_high_path)
                item = (item_high_path, item_gt_path)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def save_split(train, val, test, save_root, filepath, gt_path, img_path):
        save_path = os.path.join(save_root, "npy")
        os.makedirs(save_path,exist_ok=True)

        def preprocessing(item, gt_path, img_path, save_path):
            id = item.split('.nii.gz')[0]
            gt_nii_file = os.path.join(gt_path, item)
            img_nii_file = os.path.join(img_path, item)
            # 处理nii数据为npy
            gt = sitk.ReadImage(gt_nii_file)
            img = sitk.ReadImage(img_nii_file)
            gt = sitk.GetArrayFromImage(gt)
            img = sitk.GetArrayFromImage(img)   

            gt = np.uint8(gt)
            # remove all label except 1
            gt[gt !=1 ] = 0

            z_index, _, _ = np.where(gt>0)
            z_index = np.unique(z_index)

            out = []
            if len(z_index) > 0:
                for slice_i in z_index:
                    gt_i = gt[slice_i, :, :] # z, y, x
                    img_i = img[slice_i, :, :]
                    # covert MRI img to [0-1]
                    img_clip = np.quantile(img_i, 0.99)
                    img_low = min(img_i)
                    img_i = np.where(img_i > img_clip, img_clip, img_i)
                    # Normalization
                    img_i = (img_i - img_low*1.) / (img_clip*1. - img_low*1.)
                    img_i_3c = np.repeat(img_i[:, :, None], 3, axis=-1)

                    resize_img_i = transform.resize(
                        img_i_3c,
                        (256, 256),
                        order=3,
                        preserve_range=True,
                        mode='constant',
                        anti_aliasing=True,
                    )
                
                    resize_gt_i = transform.resize(
                        gt_i,
                        (256, 256),
                        order=0,
                        preserve_range=True,
                        mode='constant',
                        anti_aliasing=False,
                    )
                    resize_gt_i = np.uint8(resize_gt_i)
                    assert resize_img_i.shape[:2] == resize_gt_i.shape
                    
                    re_gt_i_path = join(
                            save_path,
                            "gts",
                            id
                        )
                    os.makedirs(re_gt_i_path , exist_ok=True)
                    with open(join(re_gt_i_path, str(slice_i).zfill(3) + ".pkl"), "wb") as f:
                        pickle.dump(resize_gt_i, f)


                    re_img_i_path = join(
                            save_root,
                            "imgs",
                            id
                        )
                    os.makedirs(re_img_i_path, exist_ok=True)
                    with open(join(re_img_i_path, str(slice_i).zfill(3) + ".pkl"), "wb") as f:
                        pickle.dump(resize_img_i, f)
                    


                    file_gt_path = join(re_gt_i_path, str(slice_i).zfill(3) + ".pkl")
                    file_high_path = join(re_img_i_path, str(slice_i).zfill(3) + ".pkl")
                    out.append((file_gt_path, file_high_path))
            
            return out

                    
        def _extract(items):
            # many npy files in each item
            for item in items:
                out = preprocessing(item, gt_path, img_path, save_path)

            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")
    
    