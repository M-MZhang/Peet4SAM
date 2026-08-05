# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional

from .common import LayerNorm2d
from .transformer import TwoWayTransformer, TwoWayAttentionBlock, Attention as TransformerAttention


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), # * 2倍
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), # *2倍
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) # c=256/8=32 ? 输出维度为32有什么用？
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) #[1+4, C]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)#[1, 5+N, C]

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) #[1,C,H,W]
        src = src + dense_prompt_embeddings #[1, C, H, W] ? H=W=64(patch)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :] #[1, 256]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w) #[1, 256, 64, 64]
        upscaled_embedding = self.output_upscaling(src) #[1, 32, 64*4=256, 256] keypoint!! *********************
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])) # mask_tokens_out[1, C] ->[1, 32] keypoint!! *****************
        hyper_in = torch.stack(hyper_in_list, dim=1) #[B, 4, 32] 
        b, c, h, w = upscaled_embedding.shape # 1, 32, 256, 256
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
                                                    #[1, 4, 32] @ [1, 32, 256*256] -> [B, 4, 256*256]
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out) #[1, 4]

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# QA-SAM: Hierarchical TwoWayTransformer & MaskDecoder_task
# ---------------------------------------------------------------------------

class HierarchicalTwoWayTransformer(nn.Module):
    """
    Hierarchical TwoWayTransformer for QA-SAM — U-Net style reverse fusion.

    Two modes (controlled by per_layer_a):
      - per_layer_a=False (V1-tk-all):
          All A-prompts in all layers.  Layer differentiation via keys only.
      - per_layer_a=True  (V1-tk-seq):
          Each decoder layer receives only its corresponding A-prompt
          (reverse-index: Layer 0→A₃, Layer 1→A₂, Layer 2→A₁, Layer 3→A₀).
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        per_layer_a: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.per_layer_a = per_layer_a

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = TransformerAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        inter_spatial: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        point_embedding: [B, T+A, C] where T=5 (iou+mask tokens), A=N (A-prompts).
        """
        bs, c, h, w = image_embedding.shape

        base_keys = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe_flat = image_pe.flatten(2).permute(0, 2, 1)

        # Split output tokens (first 5) from A-prompts (last N)
        output_tokens = point_embedding[:, :5, :]     # [B, 5, C]
        a_prompts = point_embedding[:, 5:, :]          # [B, N, C]
        N = a_prompts.shape[1]

        queries = point_embedding  # default: all together
        keys = base_keys

        for i, layer in enumerate(self.layers):
            # --- Keys: U-Net reverse skip ---
            if i > 0 and i <= N:
                skip_idx = N - 1 - i
                inter = inter_spatial[skip_idx].flatten(2).permute(0, 2, 1)
                keys = base_keys + inter
            else:
                keys = base_keys

            # --- Queries: per-layer or all ---
            if self.per_layer_a:
                a_idx = N - 1 - i  # reverse: Layer 0→A₃, Layer 1→A₂, ...
                cur_queries = torch.cat([output_tokens, a_prompts[:, a_idx:a_idx+1, :]], dim=1)
                cur_q_pe = cur_queries
            else:
                cur_queries = queries
                cur_q_pe = point_embedding

            queries, keys = layer(
                queries=cur_queries,
                keys=keys,
                query_pe=cur_q_pe,
                key_pe=image_pe_flat,
            )

        # Final token→image attention
        q = queries + (point_embedding if not self.per_layer_a else
                       torch.cat([output_tokens, a_prompts[:, :1, :]], dim=1))
        k = keys + image_pe_flat
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class MaskDecoder_task(nn.Module):
    """
    QA-SAM Mask Decoder with hierarchical feature fusion (Eq.3–5).

    Accepts A-prompts (from Q→A MLPs) and intermediate spatial features
    (from encoder global-attention layers).  Uses HierarchicalTwoWayTransformer
    to fuse encoder features progressively through decoder blocks.

    Ablation switches (set at init):
      - use_hierarchical: enable hierarchical spatial fusion (default True)
      - use_task_mlp:     use task-specific Q→A MLPs (default True)
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        num_global_layers: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        mlp_dim: int = 2048,
        num_heads: int = 8,
        attention_downsample_rate: int = 2,
        encoder_embed_dim: int = 768,
        per_layer_a: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.use_hierarchical = use_hierarchical
        self.per_layer_a = per_layer_a
        self.num_global_layers = num_global_layers

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1  # 4
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Hierarchical transformer: depth = #global-attn layers
        self.transformer = HierarchicalTwoWayTransformer(
            depth=num_global_layers,
            embedding_dim=transformer_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=nn.ReLU,
            attention_downsample_rate=attention_downsample_rate,
            per_layer_a=per_layer_a,
        )

        # Fallback: standard TwoWayTransformer (for ablation: w/o hierarchical)
        self.transformer_standard = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            attention_downsample_rate=attention_downsample_rate,
        )

        # skip_proj: project raw encoder features (embed_dim) → decoder dim
        self.skip_proj = nn.Sequential(
            nn.Conv2d(encoder_embed_dim, transformer_dim, kernel_size=1, bias=False),
            LayerNorm2d(transformer_dim),
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        intermediate_features: Optional[List[torch.Tensor]] = None,
        inter_spatial: Optional[List[torch.Tensor]] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            inter_spatial=inter_spatial,
            multimask_output=multimask_output,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        inter_spatial: Optional[List[torch.Tensor]] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        if self.use_hierarchical and inter_spatial is not None and len(inter_spatial) > 0:
            # Project raw encoder features (embed_dim → transformer_dim) via skip_proj
            inter_repeated = [
                torch.repeat_interleave(self.skip_proj(f), tokens.shape[0], dim=0)
                for f in inter_spatial
            ]
            hs, src_out = self.transformer(src, pos_src, tokens, inter_repeated)
        else:
            # Fallback: standard TwoWayTransformer (ablation)
            hs, src_out = self.transformer_standard(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale and predict masks
        src_out = src_out.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src_out)
        hyper_in_list = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c_up, h_up, w_up = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c_up, h_up * w_up)).view(
            b, -1, h_up, w_up
        )
        iou_pred = self.iou_prediction_head(iou_token_out)

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            masks = masks[:, :1, :, :]
            iou_pred = iou_pred[:, :1]

        return masks, iou_pred
