import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

from .build import register
from .modules import ImageEncoderViT, TwoWayTransformer, PromptEncoder_task, MaskDecoder
from .iou_loss import IOU
from  utils.transforms import ResizeLongestSide


logger = logging.getLogger(__name__)
from typing import Any, Optional, Tuple


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

@register('task_sam')
class Task_SAM(nn.Module):
    mask_threshold: float = 0.0

    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = ResizeLongestSide(encoder_mode['img_size'])
        self.embed_dim = encoder_mode['embed_dim']
        
        self.image_encoder = ImageEncoderViT(
            depth=encoder_mode['depth'],
            embed_dim=encoder_mode['embed_dim'],
            img_size=encoder_mode['img_size'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_heads=encoder_mode['num_heads'],
            patch_size=encoder_mode['patch_size'],
            qkv_bias=encoder_mode['qkv_bias'],
            use_rel_pos=encoder_mode['use_rel_pos'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
            window_size=encoder_mode['window_size'],
            out_chans=encoder_mode['out_chans'],
            in_chans=encoder_mode['in_chans'],
            rel_pos_zero_init=True,
        )

        image_embedding_size = encoder_mode['img_size'] // encoder_mode['patch_size']

        self.prompt_encoder = PromptEncoder_task(
            embed_dim=encoder_mode['prompt_embed_dim'],
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(inp_size, inp_size),
            mask_in_chans=16,
            task_num=encoder_mode['task_num']
        )

        self.mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=encoder_mode['prompt_embed_dim'],
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=encoder_mode['prompt_embed_dim'],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),

        self.pixel_mean=encoder_mode['pixel_mean']
        self.pixel_std=encoder_mode['pixel_std']

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()
    
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool=False,
    )->List[Dict[str, torch.Tensor]]:
        
        input_images = [self.transform.apply_image(x['image']) for x in batched_input] # B, H, W, C transform.resize
        input_image_torch = torch.as_tensor(input_images, device=self.device).permute(0, 3, 1, 2) # [B, C, H, W]
        input_image_torch = torch.stack([self.preprocess(input_image_torch[x]) for x in range(len(input_image_torch))], dim=0) # padding

        image_embeddings = self.image_encoder(input_image_torch) #[B, C, H, W]
        
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),#[1, C, H, W]
                image_pe=self.prompt_encoder.get_dense_pe(), #[1, C, H, W]
                sparse_prompt_embeddings=sparse_embeddings, #[1, N, C] 
                dense_prompt_embeddings=dense_embeddings,  #[1, C, H, W]
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=input_image_torch.shape[-2:], # 与下一行做修改
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
            # self.pred_mask = low_res_masks
        return outputs

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self, mask, gt):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(mask, gt)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(mask, gt)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
