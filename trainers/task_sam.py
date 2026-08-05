import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from functools import partial

from .build import register
from .modules import ImageEncoderViT, PromptEncoder_task, MaskDecoder_task
from .utils.transforms import ResizeLongestSide


# ---------------------------------------------------------------------------
# Task-Specific MLP: Q-prompt output → A-prompt  (Eq.2)
# ---------------------------------------------------------------------------
class QtoA_MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, q_output: torch.Tensor) -> torch.Tensor:
        return self.net(q_output)


@register('task_sam')
class Task_SAM(nn.Module):
    """
    QA-SAM: Self-Prompting SAM with Q&A prompt pairs and hierarchical fusion.
    """
    mask_threshold: float = 0.0

    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.transform = ResizeLongestSide(encoder_mode['img_size'])
        self.prompt_embed_dim = encoder_mode.get('prompt_embed_dim', 256)
        self.original_size = inp_size

        self.register_buffer(
            "pixel_mean",
            torch.Tensor(encoder_mode['pixel_mean']).view(-1, 1, 1), False,
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(encoder_mode['pixel_std']).view(-1, 1, 1), False,
        )

        num_q_prompts = len(encoder_mode['global_attn_indexes'])

        # ---- Image Encoder (with Q-prompts) ----
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
            num_q_prompts=num_q_prompts,
        )

        image_embedding_size = encoder_mode['img_size'] // encoder_mode['patch_size']

        # ---- Q → A  Task-Specific MLPs (Eq.2) ----
        self.q_to_a_mlps = nn.ModuleList([
            QtoA_MLP(in_dim=encoder_mode['out_chans'],
                     out_dim=self.prompt_embed_dim)
            for _ in range(num_q_prompts)
        ])

        # ---- Prompt Encoder (dense PE only) ----
        self.prompt_encoder = PromptEncoder_task(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(encoder_mode['img_size'], encoder_mode['img_size']),
            mask_in_chans=16,
        )

        # ---- Hierarchical Mask Decoder (Eq.3–5) ----
        self.mask_decoder = MaskDecoder_task(
            transformer_dim=self.prompt_embed_dim,
            num_global_layers=num_q_prompts,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            mlp_dim=2048,
            num_heads=8,
            attention_downsample_rate=2,
            encoder_embed_dim=encoder_mode['embed_dim'],
        )

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batched_input: Dict[str, Any],
        multimask_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        images = batched_input['image']  # [B, H, W, C]

        # Preprocess
        input_images_torch = self.transform.apply_image_torch(
            images.permute(0, 3, 1, 2)
        )  # [B, C, H', W']
        input_images_torch = input_images_torch.contiguous()
        input_images_torch = torch.stack(
            [self.preprocess(input_images_torch[x]) for x in range(len(input_images_torch))],
            dim=0,
        )

        # ---- Encoder: image embeddings + intermediate spatial features + Q-vectors ----
        image_embeddings, inter_spatial, inter_q = self.image_encoder(input_images_torch)
        # image_embeddings:  [B, out_chans, H, W] — final encoder output (neck)
        # inter_spatial:     list of [B, out_chans, H, W] — spatial features per global-attn layer
        # inter_q:           list of [B, out_chans]        — Q-prompt output vectors

        # ---- Q → A mapping via task-specific MLPs (Eq.2) ----
        a_prompts = []
        for q_idx, q_vec in enumerate(inter_q):
            a_vec = self.q_to_a_mlps[q_idx](q_vec)  # [B, prompt_embed_dim]
            a_prompts.append(a_vec)
        a_prompts = torch.stack(a_prompts, dim=1)  # [B, N_q, prompt_embed_dim]

        # ---- Prompt Encoder: dense PE only (NO manual prompts, NO independent task embedding) ----
        # A-prompts come EXCLUSIVELY from Q-prompts via Q→A MLPs (Eq.2).
        points = None
        if "point_coords" in batched_input:
            points = (batched_input["point_coords"], batched_input["point_labels"])

        _, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )
        # Use ONLY Q-derived A-prompts as sparse embeddings (有 Q 才有 A)
        sparse_embeddings = a_prompts  # [B, N_q, prompt_embed_dim]

        # ---- Hierarchical Mask Decoder ----
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            inter_spatial=inter_spatial,
            multimask_output=multimask_output,
        )

        # Post-process masks
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_images_torch.shape[-2:],
            original_size=self.original_size,
        )
        masks = masks > self.mask_threshold

        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }
        return outputs

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to a square input."""
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
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks, (original_size, original_size),
            mode="bilinear", align_corners=False,
        )
        return masks
    
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
