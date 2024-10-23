import argparse
import os

import torch.distributed
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import trainers
import utils
from statistics import mean
import torch
import torch.distributed as dist
import torch.nn as nn
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
device_ids=[0, 1, 2, 3]


def make_data_loader(spec, dataset_source, tag=''):
    if spec is None:
        return None

    if tag=='train':
        wrapper = datasets.make(spec, args={'dataset': dataset_source})
    elif tag == 'val':
        wrapper = datasets.make(spec, args={'dataset': dataset_source})
   
    log('{} dataset: size={}'.format(tag, len(wrapper)))
    for k, v in wrapper[0].items():
        if k!='original_size':
            log('  {}: shape={}'.format(k, v.shape))

    # sampler = torch.utils.data.distributed.DistributedSampler(wrapper)
    loader = DataLoader(wrapper, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return loader


def make_data_loaders():
    dataset = datasets.make(config.get('dataset'))
    train_loader = make_data_loader(config.get('train_wrapper'), dataset_source=dataset.train, tag='train')
    val_loader = make_data_loader(config.get('val_wrapper'), dataset_source=dataset.val, tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        model = trainers.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = trainers.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1

    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model):
    model.train()
    
    pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    
    loss_list = []
    for batch in train_loader:
        high_img = batch['image'].to('cuda') #[B, C, H, W]
        gt = batch['gt'].to('cuda')

        outputs = model.forward(batch)
        model.optimizer.zero_grad()
        model.backward_G(outputs['low_res_logits'], gt)
        model.optimizer.step()
        
        loss_list.append(model.loss_G.item())
      
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return mean(loss_list)


def eval_psnr(loader, model, eval_type=None):
    model.eval()

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
    
    loss_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        pred = torch.sigmoid(model.forward(batch)['low_res_logits'])

        loss = model.dice_loss(nn.Sigmoid(pred), batch['gt'])
        
        loss_list.extend(loss)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return loss_list


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    if config.get('resume') is not None: # load task_spesific_embed in 
        task_specific_embed = torch.load(os.path.join(save_path, "prompt_epoch_"+str(config['resume'])+".pth"))
        model.load_state_dict(task_specific_embed, strict=False)

    for name, para in model.named_parameters():
        if "task_specific_embed" not in name:
            para.requires_grad_(False)
    
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params/1000000) + ' M', '\n model_total_params:' + str(model_total_params/1000000)+' M')
    
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')

    min_loss = 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        # train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
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
           
            loss_list = eval_psnr(val_loader, model, eval_type=config.get('eval_type'))
            dice_loss = mean(loss_list)
              
            log_info.append('dice_loss: {:.4f}'.format(dice_loss))
            writer.add_scalar(dice_loss, {'val': 'dice_loss'}, epoch)

            if dice_loss < min_loss:
                min_loss = dice_loss
                save(config, model, save_path, str(epoch))
            
            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'task_sam':
            task_specific_prompt = model.prompt_encoder.task_specific_embed.state_dict()
            torch.save({"prompt_encoder.task_specific_embed": task_specific_prompt},
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


    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name, now_time)

    main(config, save_path, args=args)
