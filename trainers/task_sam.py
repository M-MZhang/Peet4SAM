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
from .utils.transforms import ResizeLongestSide


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


@register('task_sam')
class Task_SAM(nn.Module):
    mask_threshold: float = 0.0

    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = ResizeLongestSide(encoder_mode['img_size'])
        self.embed_dim = encoder_mode['embed_dim']
        self.original_size = inp_size
        self.register_buffer("pixel_mean", torch.Tensor(encoder_mode['pixel_mean']).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(encoder_mode['pixel_std']).view(-1, 1, 1), False)
        
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
            rel_pos_zero_init=True,
        )

        image_embedding_size = encoder_mode['img_size'] // encoder_mode['patch_size']

        self.prompt_encoder = PromptEncoder_task(
            embed_dim=encoder_mode['prompt_embed_dim'],
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(encoder_mode['img_size'], encoder_mode['img_size']),
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
        )

    
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool=False,
    )->List[Dict[str, torch.Tensor]]:
        images = batched_input['image'] #[B, H, W, C]
        input_images_torch = self.transform.apply_image_torch(images.permute(0, 3, 1, 2)) #[B, C, H, W]
        input_images_torch = input_images_torch.contiguous()
        input_images_torch = torch.stack([self.preprocess(input_images_torch[x]) for x in range(len(input_images_torch))], dim=0) # padding

        image_embeddings = self.image_encoder(input_images_torch) #[B, C, H, W]
        

        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # if (image_record["point_coords"] is not None) or (image_record['boxes'] or image_record['mask_inputs']
        if "point_coords" in batched_input:
            points = (batched_input["point_coords"], batched_input["point_labels"])
        else:
            points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,#[B, C, H, W]
            image_pe=self.prompt_encoder.get_dense_pe(), #[1, C, H, W]
            sparse_prompt_embeddings=sparse_embeddings, #[1, N, C] 
            dense_prompt_embeddings=dense_embeddings,  #[1, C, H, W]
            multimask_output=multimask_output,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_images_torch.shape[-2:], # 与下一行做修改
            original_size=self.original_size,
        )
        masks = masks > self.mask_threshold
        outputs={
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }
        
        # if self.prompt_encoder.task_specific_embed.weight.requires_grad:
        #     return self.backward_G(low_res_masks, batched_input['gt'])
        
        return outputs

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std 
       

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
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, (original_size, original_size), mode="bilinear", align_corners=False)
        return masks

    # def backward_G(self, mask, gt):
    #     """Calculate GAN and L1 loss for the generator"""
    #     self.loss_G = self.criterionBCE(mask, gt)
    #     self.loss_G += self.dice_loss(nn.Sigmoid()(mask), gt)

    #     if self.loss_mode == 'iou':
    #         self.loss_G += self.criterionIOU(mask, gt)

    #     return self.loss_G

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
