import os
join = os.path.join
from datasets import register
from PIL import Image
import numpy as np


@register('kvasir_seg')
class Kvasir_SEG:
    def __init__(self, data_root, bbox_shift=None):
        self.data_root = join(data_root, "Kvasir-SEG")
        self.mask_path = join(self.data_root, 'masks')
        self.image_path = join(self.data_root, 'images')
        self.train_split_file_path = join(self.data_root, 'train.txt')
        self.val_split_file_path = join(self.data_root, 'val.txt')

        with open(self.train_split_file_path,'r') as f:
            train_name_list = f.readlines()
        
        train_list=[]
        for train_name in train_name_list:
            train_img_path = join(self.image_path, train_name.split('\n')[0]+'.jpg')
            train_mask_path = join(self.mask_path, train_name.split('\n')[0]+'.jpg')
            train_list.append((train_img_path, train_mask_path))
        
        with open(self.val_split_file_path, 'r') as f:
            val_name_list = f.readlines()
        
        val_list=[]
        for val_name in val_name_list:
            val_img_path = join(self.image_path, val_name.split('\n')[0]+'.jpg')
            val_mask_path = join(self.mask_path, val_name.split('\n')[0]+'.jpg')
            val_list.append((val_img_path, val_mask_path))
        
        self.mode = 'rgb'
        self.train = train_list
        self.val = self.test = val_list
        self.mean, self.std = self.get_data_statistics()
        self.H_max = self.H_min = None
        


    def get_data_statistics(self):
        image_files = os.listdir(self.image_path)
       
        std_val_list = []
        psum_list = []
        
        count_list = []
        for image in image_files:
            image = Image.open(join(self.image_path, image))
            image_arr = np.array(image) #[0-255]

            psum = np.sum(image_arr)
            std_val = np.std(image_arr)

    
            std_val_list.append(std_val)
            psum_list.append(psum)
            

            count_list.append(image_arr.shape[0]*image_arr.shape[1]*image_arr.shape[2])
        
        total_mean = sum(psum_list) / sum(count_list)
        total_std = sum(std_val_list) / sum(count_list)

        return total_mean, total_std



            

