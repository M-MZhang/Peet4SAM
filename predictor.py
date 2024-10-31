import json
import yaml
import argparse
import os
import pickle
import torch
import trainers
from PIL import Image
import numpy as np


join = os.path.join
from datasets.uitls import read_json

def main(config):
    json_path = '../data/Kvasir-SEG/split_kvasir_seg.json'
    data_root = '../data/Kvasir-SEG/npy'
    save_root = '../visulization/Kvasir-SEG'
    os.makedirs(save_root, exist_ok=True)

    split_file = read_json(json_path)
    val_list = split_file['val']

    # sam_checkpoint = torch.load(config['sam_checkpoint'])
    model = trainers.make(config['model']).cuda()
    # model_state_dict = model.state_dict()
    # model_state_dict.update(sam_checkpoint)
    # model.load_state_dict(model_state_dict)

    if config.get('resume') is not None:
        try:
            task_specific_embed = torch.load(os.path.join('../save',args.name, 'train','kvasir_seg','1_prompts', "prompt_epoch_"+str(config['resume'])+".pth"))
            model.load_state_dict(task_specific_embed, strict=False)
        except FileNotFoundError:
            print ("File does not exist!")
            raise
    
    # val_list = os.listdir(join(data_root,'masks'))
    for val in val_list:
        val_img_path = join(data_root,'images',val)
        val_mask_path = join(data_root,'masks',val)

        with open(val_img_path,'rb') as f:
            val_img = pickle.load(f)
            val_img = torch.tensor(val_img).unsqueeze(0).cuda()
        with open(val_mask_path,'rb') as f:
            val_mask = pickle.load(f)
            
        
        batched_input = {'image':val_img}
        pred = model.forward(batched_input) 
        pred_mask = pred['masks'].squeeze().cpu()

        val_img = Image.fromarray(np.uint8(np.array(val_img.squeeze().cpu())* 255))
        original_mask = Image.fromarray(np.uint8(val_mask * 255))
        pred_mask = Image.fromarray(np.uint8(np.array(pred_mask)*255))

        val_img.save(join(save_root, val.split('.pkl')[0]+'_img.png'))
        original_mask.save(join(save_root, val.split('.pkl')[0]+'_origianl.png'))
        pred_mask.save(join(save_root, val.split('.pkl')[0]+'_pred.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam-vit-task.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()
    

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config)