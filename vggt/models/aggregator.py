# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.layers.pruning import TokenPruner

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=None,
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        # pruning config (applies to global attention only)
        enable_global_pruning: bool = False,
        prune_keep_ratio: float = 0.7,
        prune_hidden_ratio: float = 0.25,
        prune_dropout: float = 0.0,
        # differentiable gating + losses
        prune_use_gumbel: bool = True,
        prune_tau: float = 1.0,
        prune_ratio_weight: float = 0.0,
        prune_distill_weight: float = 0.0,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = ["frame", "global"] if aa_order is None else aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.enable_global_pruning = enable_global_pruning
        self.prune_keep_ratio = prune_keep_ratio
        self.prune_use_gumbel = prune_use_gumbel
        self.prune_tau = prune_tau
        self.prune_ratio_weight = prune_ratio_weight
        self.prune_distill_weight = prune_distill_weight
        logger.info("PRUNE DISTILL WEIGHT: %s", self.prune_distill_weight)
        
        # Sparse pruning: only apply at specific layers (e.g., every 8 layers for 24 total)
        # This applies pruning 3 times across 24 global blocks: at layers 7, 15, 23
        self.prune_layers = [7, 15, 23] if depth == 24 else []
        logger.info("Pruning will be applied at global block indices: %s", self.prune_layers)
        
        self.global_pruner = None
        if self.enable_global_pruning:
            self.global_pruner = TokenPruner(embed_dim, hidden_ratio=prune_hidden_ratio, dropout=prune_dropout)

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape
        if B <= 0 or S <= 0:
            raise RuntimeError(
                f"Aggregator received empty batch or sequence: images.shape={tuple(images.shape)}. "
                "Upstream dataloader likely produced an empty batch."
            )

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        if P <= 0:
            raise RuntimeError(
                f"Aggregator produced zero patch tokens. Check img_size/patch_size so H//patch_size and W//patch_size are > 0. "
                f"Got H={H}, W={W}, patch_size={self.patch_size}."
            )

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        # reset pruning losses accumulation
        self._prune_ratio_loss_total = torch.zeros((), device=images.device)
        self._prune_distill_loss_total = torch.zeros((), device=images.device)
        self._prune_loss_count = 0

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates

        # finalize averaged losses for external consumption
        if self._prune_loss_count > 0:
            self.prune_ratio_loss = self._prune_ratio_loss_total / float(self._prune_loss_count)
            self.prune_distill_loss = self._prune_distill_loss_total / float(self._prune_loss_count)
        else:
            self.prune_ratio_loss = torch.zeros((), device=images.device)
            self.prune_distill_loss = torch.zeros((), device=images.device)

        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            # Only apply pruning at specific layers
            should_prune = (self.enable_global_pruning and 
                          self.global_pruner is not None and 
                          global_idx in self.prune_layers)
            
            if should_prune:
                # keep special tokens and prune only patch tokens
                num_special = self.patch_start_idx
                special = tokens[:, :num_special, :]
                patches = tokens[:, num_special:, :]

                # compute per-token scores on patch tokens
                scores = self.global_pruner(patches)  # [B, S*P - num_special]
                keep_k = max(1, int(scores.shape[1] * self.prune_keep_ratio))
                if self.training and self.prune_use_gumbel:
                    # Teacher: full sequence forward (no grad to teacher for distill)
                    teacher_out = self.global_blocks[global_idx](tokens, pos=pos).detach()

                    # Straight-through hard top-k gate: hard forward, soft gradients
                    soft = torch.sigmoid(scores / max(self.prune_tau, 1e-6))  # [B, Np]
                    topk_idx = torch.topk(scores, k=keep_k, dim=1, largest=True, sorted=False).indices
                    hard = torch.zeros_like(scores)
                    hard.scatter_(1, topk_idx, 1.0)
                    st_mask = hard + soft - soft.detach()  # [B, Np]

                    # Apply ST mask at full length
                    patches_masked = patches * st_mask.unsqueeze(-1)
                    tokens_in = torch.cat([special, patches_masked], dim=1)
                    pos_in = pos

                    student_out = self.global_blocks[global_idx](tokens_in, pos=pos_in)

                    # Distillation loss between student and teacher on patch tokens
                    distill_loss = F.mse_loss(student_out[:, num_special:, :], teacher_out[:, num_special:, :])
                    # Ratio loss to match target keep ratio (use soft mask expectation)
                    ratio_target = keep_k / scores.shape[1]
                    ratio_loss = ((soft.mean(dim=1) - ratio_target) ** 2).mean()

                    # Accumulate weighted losses
                    if self.prune_distill_weight > 0.0:
                        self._prune_distill_loss_total = self._prune_distill_loss_total + self.prune_distill_weight * distill_loss
                    if self.prune_ratio_weight > 0.0:
                        self._prune_ratio_loss_total = self._prune_ratio_loss_total + self.prune_ratio_weight * ratio_loss
                    self._prune_loss_count += 1

                    tokens = student_out
                else:
                    # Inference-time hard top-k pruning and scatter back
                    topk = torch.topk(scores, k=keep_k, dim=1, largest=True, sorted=False)
                    keep_idx = topk.indices  # [B, keep_k]
                    '''
                    keep_idx = torch.stack([
                        torch.randperm(scores.shape[1], device=scores.device)[:keep_k]
                        for _ in range(scores.shape[0])
                    ])
                    '''
                    print(f'only keeping {keep_k} out of {scores.shape[1]}')


                    # gather kept tokens per batch
                    gather_indices = keep_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
                    kept = torch.gather(patches, dim=1, index=gather_indices)

                    # concatenate special tokens back
                    pruned_tokens = torch.cat([special, kept], dim=1)

                    # run the global block on pruned sequence
                    if pos is not None:
                        pos_special = pos[:, :num_special, :]
                        pos_patches = pos[:, num_special:, :]
                        pos_kept = torch.gather(pos_patches, dim=1, index=keep_idx.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))
                        pos_pruned = torch.cat([pos_special, pos_kept], dim=1)
                    else:
                        pos_pruned = None

                    pruned_out = self.global_blocks[global_idx](pruned_tokens, pos=pos_pruned)

                    # scatter back to full sequence
                    out = tokens.clone()
                    out[:, :num_special, :] = pruned_out[:, :num_special, :]
                    patch_out = pruned_out[:, num_special:, :]
                    scatter_indices = (keep_idx + num_special).unsqueeze(-1).expand(-1, -1, patches.shape[-1])
                    out.scatter_(dim=1, index=scatter_indices, src=patch_out)
                    tokens = out
            else:
                if self.training:
                    tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = self.global_blocks[global_idx](tokens, pos=pos)

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
