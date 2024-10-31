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

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6 ,7'
device_ids=[0, 1, 2,3 , 4,5 ,6 ,7]
# torch.cuda.set_device('cuda:{}'.format(device_ids[0]))


def make_data_loader(spec, dataset_source, tag=''):
    if spec is None:
        return None

    if tag=='train':
        wrapper = datasets.make(spec, args={'dataset': dataset_source})
    elif tag == 'val':
        wrapper = datasets.make(spec, args={'dataset': dataset_source})
   
    log('{} dataset: size={}'.format(tag, len(wrapper)), filename)
    for k, v in wrapper[0].items():
        if k!='original_size':
            log('  {}: shape={}'.format(k, v.shape), filename)

    # sampler = torch.utils.data.distributed.DistributedSampler(wrapper)
    loader = DataLoader(wrapper, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return loader


def make_data_loaders():
    dataset = datasets.make(config.get('dataset'))
    train_loader = make_data_loader(config.get('train_wrapper'), dataset_source=dataset, tag='train')
    val_loader = make_data_loader(config.get('val_wrapper'), dataset_source=dataset, tag='val')
    return train_loader, val_loader


def prepare_training():
    model = trainers.make(config['model']).cuda()
    model_state_dict = model.state_dict()
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model_state_dict.update(sam_checkpoint)
    model.load_state_dict(model_state_dict, strict=False)
   
    if config.get('resume') is not None:
        epoch_start = config.get('resume') + 1
        try:
            task_specific_embed = torch.load(os.path.join(save_path, 'prompt_epoch_'+str(config['resume'])+'.pth'))
            model.prompt_encoder.task_specific_embed.load_state_dict(task_specific_embed)
        except FileExistsError:
            print('No such file!')
            raise
    else:
        epoch_start = 1
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])

    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)), filename)
    return model, optimizer, epoch_start, lr_scheduler




def train(train_loader, model, optimizer):
    model.train()
    ce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = BinaryDiceLoss()
    
    pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    
    loss_list = []
    for batch in train_loader:
        high_img = batch['image'].to('cuda') #[B, H, W, C]
        gt = batch['gt'].to('cuda')

        outputs = model.forward(batch)
        optimizer.zero_grad()
        loss = ce_loss(outputs['low_res_logits'], gt.float()) + dice_loss(outputs['low_res_logits'], gt)
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
      
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return mean(loss_list)


def eval_psnr(loader, model, eval_type=None):
    model.eval()
    dice_loss = BinaryDiceLoss()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    
    dice_loss_list = []
    iou_loss_list = []

    for batch in loader:
        high_img = batch['image'].to('cuda') #[B, C, H, W]
        gt = batch['gt'].to('cuda')

        pred = model.forward(batch)['low_res_logits']
        dice_loss_list.append(dice_loss(pred, gt).item())
        iou_loss_list.append(iou_loss(pred, gt).item())

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return mean(dice_loss_list), mean(iou_loss_list)


def main(config_, save_path, args):
    global config, log, writer, log_info, filename
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    if os.path.exists(os.path.join(save_path, 'log.txt')):
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            filename = 'log_' + now_time + '.txt'
    else:
        filename = 'log.txt'
    
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer


    for name, para in model.named_parameters():
        if ("task_specific_embed" not in name) and ("adapter" not in name) and ("prompt_adapter" not in name):
            para.requires_grad_(False)
        else:
            print(name)
        
    
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params/1000000) + ' M', '\n model_total_params:' + str(model_total_params/1000000)+' M')
    
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')

    min_loss = 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, optimizer)
        lr_scheduler.step()

     
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
            
        save(config, model, save_path, 'last')
        
        if (epoch_val is not None) and (epoch % epoch_val == 0):
           
            dice_loss, iou_loss = eval_psnr(val_loader, model, eval_type=config.get('eval_type'))
            
            log_info.append('mDIC: {:.4f}'.format(1-dice_loss))
            writer.add_scalar('mDIC',1-dice_loss, epoch)
            log_info.append('mIOU:{:4f}'.format(1-iou_loss))
            writer.add_scalar('mIOU', 1-iou_loss, epoch)
        
            # if (dice_loss + iou_loss) < min_loss:
            #     min_loss = dice_loss + iou_loss
            
            if epoch % 10 == 0:
                save(config, model, save_path, str(epoch))
            
            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info), filename)
            writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'task_sam':
            task_specific_prompt = model.module.task_specific_embed.state_dict()
            torch.save(task_specific_prompt,
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam-vit-task.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()
    

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('../save', save_name, 'train', config['dataset']['name'], str(config['model']['args']['encoder_mode']['task_num'])+"_prompts")

    main(config, save_path, args=args)
