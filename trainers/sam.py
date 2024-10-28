# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import Any, Dict, List, Tuple

from .modules import ImageEncoderViT, TwoWayTransformer, PromptEncoder, MaskDecoder
from .utils.transforms import ResizeLongestSide
from .build import register

@register('sam')
class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        inp_size=None, 
        encoder_mode=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()

        self.image_encoder = ImageEncoderViT(
            depth=encoder_mode['depth'],
            embed_dim=encoder_mode['embed_dim'],
            img_size=encoder_mode['img_size'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            norm_layer=torch.nn.LayerNorm,
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

        self.prompt_encoder = PromptEncoder(
            embed_dim=encoder_mode['prompt_embed_dim'],
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(encoder_mode['img_size'], encoder_mode['img_size']),
            mask_in_chans=16,
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

        self.transform = ResizeLongestSide(encoder_mode['img_size'])
        self.original_size = inp_size
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool=False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        images = batched_input['image'] #[B, H, W, C]
        input_images_torch = self.transform.apply_image_torch(images.permute(0, 3, 1, 2))
        input_images_torch = input_images_torch.contiguous()
        input_images_torch = torch.stack([self.preprocess(input_images_torch[x]) for x in range(len(input_images_torch))], dim=0) # padding
        
        image_embeddings = self.image_encoder(input_images_torch) #[B, C, H, W]

        
        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
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
            image_embeddings=image_embeddings,#[1, C, H, W]
            image_pe=self.prompt_encoder.get_dense_pe(), #[1, C, H, W]
            sparse_prompt_embeddings=sparse_embeddings, #[1, N, C] 
            dense_prompt_embeddings=dense_embeddings,  #[1, C, H, W]
            multimask_output=multimask_output,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_images_torch.shape[-2:],
            original_size=self.original_size,
        )
        masks = masks > self.mask_threshold
        outputs={
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }
        
        return outputs

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
