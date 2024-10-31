import os
join = os.path.join
from datasets import register
from PIL import Image
import numpy as np
from skimage import transform
import pickle

from .uitls import mkdir_if_missing, read_json, write_json


@register('kvasir_seg')
class Kvasir_SEG:
    def __init__(self, data_root, bbox_shift=None):
        self.data_root = join(data_root, "Kvasir-SEG")
        self.mask_path = join(self.data_root, 'masks')
        self.image_path = join(self.data_root, 'images')
        self.train_split_file_path = join(self.data_root, 'train.txt')
        self.val_split_file_path = join(self.data_root, 'val.txt')
        self.preprocess_data_path = join(self.data_root, "npy")
        self.split_json_path = join(self.data_root, 'split_kvasir_seg.json')
        

        with open(self.train_split_file_path,'r') as f:
            train_name_list = f.readlines()
        with open(self.val_split_file_path, 'r') as f:
            val_name_list = f.readlines()
        
        
        
        if os.path.exists(self.split_json_path):
            train_list = read_json(self.split_json_path)['train']
            val_list = read_json(self.split_json_path)['val']
        else:
            self.mean, self.std = self.get_data_statistics()
            os.makedirs(join(self.preprocess_data_path,"images"))
            os.makedirs(join(self.preprocess_data_path, "masks"))
            train_list = []
            for train_name in train_name_list:
                train_img_path = join(self.image_path, train_name.split('\n')[0]+'.jpg')
                train_mask_path = join(self.mask_path, train_name.split('\n')[0]+'.jpg')
                image_save_path = join(self.data_root, "npy", "images",train_name.split('\n')[0]+'.pkl')
                mask_save_path = image_save_path.replace('images', 'masks')
                
                self.Normalization(train_img_path, train_mask_path, image_save_path, mask_save_path)
                train_list.append(train_name.split('\n')[0]+'.pkl')
            
            val_list = []
            for val_name in val_name_list:
                val_img_path = join(self.image_path, val_name.split('\n')[0]+'.jpg')
                val_mask_path = join(self.mask_path, val_name.split('\n')[0]+'.jpg')
                image_save_path = join(self.data_root, "npy", "images",val_name.split('\n')[0]+'.pkl')
                mask_save_path = image_save_path.replace('images', 'masks')

                self.Normalization(val_img_path, val_mask_path, image_save_path, mask_save_path)
                val_list.append(val_name.split('\n')[0]+'.pkl')
            split = {'train': train_list, 'val': val_list}

            
            write_json(split, self.split_json_path)
            print("Save split to {}".format(self.split_json_path))
            
        train_list = [(join(self.data_root, "npy", "images", x), join(self.data_root, "npy", "masks", x)) for x in train_list]
        val_list = [(join(self.data_root, "npy", "images", x), join(self.data_root, "npy", "masks", x)) for x in val_list]

        self.mode = 'rgb'
        self.train = train_list
        self.val = self.test = val_list
        self.H_max = self.H_min = None
    
    def Normalization(self, img_path, mask_path, image_save_path, mask_save_path):
        
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert('1')

        img = np.array(img)
        mask = np.array(mask)
        mask = np.uint8(mask)
        mask[mask==255]=1

        img = np.float32(img)
        img = (img - self.mean) / self.std
        img = (img - img.min()) / (img.max() - img.min()+0.0000001)
        
        mask = transform.resize(mask, 
                                (256, 256), 
                                order=0,
                                preserve_range=True,
                                mode='constant',
                                anti_aliasing=False)

        img = transform.resize(img, 
                                (256, 256), 
                                order=3,
                                preserve_range=True,
                                mode='constant',
                                anti_aliasing=False)

        with open(image_save_path, 'wb') as f:
            pickle.dump(img, f)
        with open(mask_save_path, 'wb') as f:
            pickle.dump(mask, f)
            
        

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



            

