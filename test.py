import argparse
import os

# import torch.distributed
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import trainers
import utils
from utils import iou_loss, BinaryDiceLoss
from statistics import mean
import torch
import torch.distributed as dist
import torch.nn as nn
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device_ids=[0, 1, 2, 3]



def make_data_loader(spec, dataset_source, tag=''):
    if spec is None:
        return None

    if tag=='train':
        wrapper = datasets.make(spec, args={'dataset': dataset_source})
    elif tag == 'val':
        wrapper = datasets.make(spec, args={'dataset': dataset_source})
    elif tag == 'test':
        wrapper = datasets.make(spec, args={'dataset':dataset_source})
   
    log('{} dataset: size={}'.format(tag, len(wrapper)))
    for k, v in wrapper[0].items():
        if k!='original_size':
            log('  {}: shape={}'.format(k, v.shape))

    loader = DataLoader(wrapper, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return loader

def make_data_loaders():
    dataset = datasets.make(config.get('dataset'))
    test_loader = make_data_loader(config.get('test_wrapper'), dataset_source=dataset, tag='test')
    return test_loader

def evaluate(loader, model, eval_type=None):
    model.eval()

    pbar = tqdm(total=len(loader), leave=False, desc='test')
    dice_loss = BinaryDiceLoss()
    

    dice_loss_list = []
    iou_loss_list = []
    for batch in loader:
        gt = batch['gt'].to('cuda')

        pred = model.forward(batch)
        loss_1 = dice_loss(pred['low_res_logits'], gt) 
        loss_2 = iou_loss(pred['low_res_logits'], gt)
        dice_loss_list.append(loss_1.item())
        iou_loss_list.append(loss_2.item())

        if pbar is not None:
            pbar.update(1)
    
    if pbar is not None:
        pbar.close()
    
    return mean(dice_loss_list), mean(iou_loss_list)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    test_loader = make_data_loaders()
    
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model = trainers.make(config['model']).cuda()
    model_state_dict = model.state_dict()
    model_state_dict.update(sam_checkpoint)
    model.load_state_dict(model_state_dict)

    if config.get('resume') is not None:
        try:
            task_specific_embed = torch.load(os.path.join('save',args.name, 'train', "prompt_epoch_"+str(config['resume'])+".pth"))
            model.prompt_encoder.task_specific_embed.load_state_dict(task_specific_embed, strict=False)
        except FileNotFoundError:
            print ("File does not exist!")
            raise
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    dice_loss, iou_loss = evaluate(test_loader, model)

    log_info = ['Test result for {}/{}'.format(args.name, config.get('resume'))]
    log_info.append(['Dice loss: {}'.format(dice_loss)])
    log_info.append(['Iou loss: {}'.format(iou_loss)])

    log(','.join(log_info))
    writer.flush
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam-vit-task.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()
    

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('save', save_name, 'test')

    main(config, save_path, args=args)