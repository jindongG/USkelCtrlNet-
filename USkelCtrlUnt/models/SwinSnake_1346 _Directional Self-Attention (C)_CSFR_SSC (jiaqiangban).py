# SwinSnake_reconstruct.py
import torch
from torch import nn
import torch.nn.functional as F
import os
import einops
from typing import Optional, Callable, List
from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth
from torchinfo import summary
# ========================================
# Directional Axial Attention
# (global attention along H then W)
# ========================================
class AxialAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_h = nn.Linear(dim, dim * 3)
        self.qkv_w = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, H, W, C] (NHWC)
        """
        B, H, W, C = x.shape

        # --------- Height (H) attention ---------
        x_h = x.permute(0, 2, 1, 3)  # B, W, H, C
        qkv = self.qkv_h(x_h).reshape(B * W, H, 3, self.heads, C // self.heads)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q * self.scale
        attn_h = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        out_h = attn_h @ v
        out_h = out_h.reshape(B, W, H, C).permute(0, 2, 1, 3)  # B,H,W,C

        # --------- Width (W) attention ---------
        x_w = x
        qkv = self.qkv_w(x_w).reshape(B * H, W, 3, self.heads, C // self.heads)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q * self.scale
        attn_w = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        out_w = attn_w @ v
        out_w = out_w.reshape(B, H, W, C)

        # combine
        out = out_h + out_w
        out = self.proj(out)
        out = self.dropout(out)
        return out

class AttentionProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        return torch.sigmoid(self.proj(x))

# ========================================
# Window-based Directional Self-Attention
# ========================================

# -------------------------
# Utility: patch merging pad (kept for compatibility)
# -------------------------
def _patch_merging_pad(x):
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    return x

torch.fx.wrap("_patch_merging_pad")

# -------------------------
# PatchMerging (kept)
# -------------------------
class PatchMerging(nn.Module):
    def __init__(self, dim: int = 3, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor):
        # x: [..., H, W, C] expected as NHWC in this path in original code
        x = x.permute(0, 2, 3, 1)
        x = _patch_merging_pad(x)
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.reduction(self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return x


# -------------------------
# Shifted-window attention (original implementation reused)
# -------------------------
def shifted_window_attention(
    input: torch.Tensor,
    qkv_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    relative_position_bias: torch.Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[torch.Tensor] = None,
    proj_bias: Optional[torch.Tensor] = None,
):
    # exactly as in original code (kept)
    B, H, W, C = input.shape
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    if window_size[0] >= pad_H: shift_size[0] = 0
    if window_size[1] >= pad_W: shift_size[1] = 0

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)

    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0]: h[1], w[0]: w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    x = x[:, :H, :W, :].contiguous()
    return x

torch.fx.wrap("shifted_window_attention")


# -------------------------
# ShiftedWindowAttention wrapper (kept but will be wrapped for deformable)
# -------------------------
class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int,
                 qkv_bias: bool = True, proj_bias: bool = True, attention_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor):
        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )

# ========================================
# Window-based Directional Self-Attention
# ========================================
class WindowDirectionalAttention(ShiftedWindowAttention):
    def forward(self, x):
        B,H,W,C = x.shape
        ws = self.window_size[0]

        # padding to multiples of window size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_pad = F.pad(x, (0,0,0,pad_r,0,pad_b))

        Hp, Wp = x_pad.shape[1], x_pad.shape[2]
        xw = x_pad.view(B, Hp//ws, ws, Wp//ws, ws, C)
        xw = xw.permute(0,1,3,2,4,5).reshape(-1, ws*ws, C)

        # reshape to window tokens
        xw_ = xw.reshape(-1, ws, ws, C)

        # --------- Height attention ---------
        h_feat = xw_.permute(0,2,1,3)   # B*nw, W,H,C
        qkv = self.qkv(h_feat).reshape(h_feat.size(0)*ws, ws, 3, self.num_heads, C//self.num_heads)
        q,k,v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q * (C//self.num_heads)**-0.5
        attn = (q @ k.transpose(-2,-1)).softmax(dim=-1)
        out_h = attn @ v
        out_h = out_h.reshape(-1, ws, ws, C)

        # --------- Width attention ---------
        w_feat = out_h
        qkv = self.qkv(w_feat).reshape(w_feat.size(0)*ws, ws, 3, self.num_heads, C//self.num_heads)
        q,k,v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q * (C//self.num_heads)**-0.5
        attn = (q @ k.transpose(-2,-1)).softmax(dim=-1)
        out_w = attn @ v
        out_w = out_w.reshape(-1, ws*ws, C)

        # linear projection
        out = self.proj(out_w)

        # restore spatial
        out = out.reshape(B, Hp//ws, Wp//ws, ws, ws, C)
        out = out.permute(0,1,3,2,4,5).reshape(B, Hp, Wp, C)
        return out[:, :H, :W, :]

# -------------------------
# Swin block (kept, but attn_layer can be DeformableWindowAttention wrapper)
# -------------------------
class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        num_heads: int = 3,
        window_size: List[int] = [7, 7],
        shift_size: List[int] = [3, 3],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        # x expected NHWC inside attention usage in this repo
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


# -------------------------
# Conv + basic blocks
# -------------------------
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(max(1, out_ch // 4), out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


# -------------------------
# Original DSConv (kept) with modularization
# -------------------------
class DSConv(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, kernel_size: int = 9,
                 extend_scope: float = 1.0, morph: int = 0, if_offset: bool = True, device: str | torch.device = "cuda"):
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        # normalization for offset prediction; original used GroupNorm
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(max(1, out_channels // 4), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)
        self.dsc_conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(kernel_size, 1), padding=0)
        self.dsc_conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, kernel_size), padding=0)

    def forward(self, input: torch.Tensor):
        offset = self.offset_conv(input)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(offset=offset, morph=self.morph,
                                                                  extend_scope=self.extend_scope, device=self.device)
        deformed_feature = get_interpolated_feature(input, y_coordinate_map, x_coordinate_map)
        output = self.dsc_conv_y(deformed_feature) if self.morph else self.dsc_conv_x(deformed_feature)
        output = self.gn(output)
        output = self.relu(output)
        return output


# -------------------------
# Direction-aware DSConv: small extension of DSConv
# -------------------------
class DA_DSConv(DSConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # direction bias: two scalars (horizontal / vertical). Learnable small scalars.
        self.direction_bias = nn.Parameter(torch.zeros(2))
        nn.init.normal_(self.direction_bias, std=0.02)
        # simple channel-wise scaling
        self.channel_scale = nn.Parameter(torch.ones(1, 1, 1, 1))  # broadcastable

    def forward(self, x):
        out = super().forward(x)
        bias = self.direction_bias[self.morph]  # morph 0 -> horizontal, 1 -> vertical
        out = out * (1.0 + bias) * self.channel_scale
        return out


# -------------------------
# coordinate functions reused from your DSConv (kept identical)
# -------------------------
def get_coordinate_map_2D(offset: torch.Tensor, morph: int, extend_scope: float = 1.0, device: str | torch.device = "cuda"):
    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)
    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)
    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)
    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)
        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()
        y_offset_new_[center] = 0
        for index in range(1, center + 1):
            y_offset_new_[center + index] = y_offset_new_[center + index - 1] + y_offset_[center + index]
            y_offset_new_[center - index] = y_offset_new_[center - index + 1] + y_offset_[center - index]
        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")
    else:
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)
        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()
        x_offset_new_[center] = 0
        for index in range(1, center + 1):
            x_offset_new_[center + index] = x_offset_new_[center + index - 1] + x_offset_[center + index]
            x_offset_new_[center - index] = x_offset_new_[center - index + 1] + x_offset_[center - index]
        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(input_feature: torch.Tensor, y_coordinate_map: torch.Tensor, x_coordinate_map: torch.Tensor, interpolate_mode: str = "bilinear"):
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")
    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1
    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])
    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)
    interpolated_feature = nn.functional.grid_sample(input=input_feature, grid=grid, mode=interpolate_mode, padding_mode="zeros", align_corners=True)
    return interpolated_feature


def _coordinate_map_scaling(coordinate_map: torch.Tensor, origin: list, target: list = [-1, 1]):
    minv, maxv = origin
    a, b = target
    coordinate_map_scaled = torch.clamp(coordinate_map, minv, maxv)
    scale_factor = (b - a) / (maxv - minv)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - minv)
    return coordinate_map_scaled


# -------------------------
# MultiView_DSConv using DA_DSConv
# -------------------------
class MultiView_DSConv(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, kernel_size: int = 9, extend_scope: float = 1.0, device_id: str | torch.device = "cuda"):
        super().__init__()
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")
        self.dsconv_x = DA_DSConv(in_channels, out_channels, kernel_size, extend_scope, 1, True, device_id).to(device)
        self.dsconv_y = DA_DSConv(in_channels, out_channels, kernel_size, extend_scope, 0, True, device_id).to(device)
        self.conv = Conv(in_channels, out_channels)
        self.conv_fusion = Conv(out_channels * 3, out_channels)

    def forward(self, x):
        conv_x = self.conv(x)
        dsconvx_x = self.dsconv_x(x)
        dsconvy_x = self.dsconv_y(x)
        x = self.conv_fusion(torch.cat([conv_x, dsconvx_x, dsconvy_x], dim=1))
        return x


# -------------------------
# Deformable Window wrapper (S1): compute small per-pixel offset field, warp feature map, then call shifted_window_attention.
# This is simpler and stable to train.
# -------------------------
class DeformableWindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int,
                 offset_scale: float = 0.125, qkv_bias: bool = True, proj_bias: bool = True,
                 attention_dropout: float = 0.0, dropout: float = 0.0):
        """
        offset_scale: maximum relative offset (in pixels) scaled by feature map size; small value (0.1~0.2) is recommended
        """
        super().__init__()
        # embed underlying shifted window attention (kept)
        self.inner_attn = ShiftedWindowAttention(dim, window_size, shift_size, num_heads, qkv_bias, proj_bias, attention_dropout, dropout)
        # small conv to predict 2-channel offset field (dx, dy) in pixel units (not normalized)
        self.offset_conv = nn.Conv2d(dim, 2, kernel_size=3, padding=1)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.offset_scale = offset_scale  # scale relative to min(H,W) to keep offsets small

    def forward(self, x: torch.Tensor):
        # x is expected NHWC shape as earlier usage
        B, H, W, C = x.shape
        # convert to NCHW for conv
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        # predict offsets per-pixel (dx, dy) in a small range
        offset = self.offset_conv(x_nchw)  # (B,2,H,W)
        # scale offsets to reasonable pixel range (tanh + scale)
        # use tanh to restrict
        offset = torch.tanh(offset) * (self.offset_scale * min(H, W))
        # create normalized sampling grid then add offsets and grid_sample
        # grid_sample expects normalized coords in [-1,1]; build base grid:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing='ij')
        base_grid = torch.stack((xx, yy), dim=-1)  # H W 2
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # B H W 2
        # convert pixel offset to normalized offsets: offset_px / (W-1)/ (H-1) scaled accordingly
        # offset shape (B,2,H,W) -> (B,H,W,2)
        offset_perm = offset.permute(0, 2, 3, 1)
        # normalize offsets: dx normalized by (W-1)/2, dy by (H-1)/2
        dx = offset_perm[..., 0] / max(1.0, (W - 1) / 2.0)
        dy = offset_perm[..., 1] / max(1.0, (H - 1) / 2.0)
        norm_offset = torch.stack([dx, dy], dim=-1)
        warped_grid = base_grid + norm_offset  # B H W 2
        # grid_sample expects NCHW input and grid shape N H W 2 (x,y)
        warped = F.grid_sample(x_nchw, warped_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # convert back NHWC
        warped_nhwc = warped.permute(0, 2, 3, 1).contiguous()
        # now call shifted-window attention on warped features (this preserves Swin behavior)
        return self.inner_attn(warped_nhwc)


# -------------------------
# Deformable SwinLayer wrapper (uses DeformableWindowAttention)
# -------------------------
class DeformableSwinLayer(nn.Module):
    def __init__(self, channels=36, is_shift=1, use_axial=False):
        super().__init__()

        if use_axial:
            # Low-resolution: global axial attention
            attn_layer = lambda dim, window, shift, heads, **kw: \
                AxialAttention(dim, num_heads=heads)
        else:
            # High-resolution: window directional + deformable warp
            attn_layer = lambda dim, window, shift, heads, qkv_bias=True, proj_bias=True, attention_dropout=0.0, dropout=0.0: \
                DeformableWindowAttention(
                    dim, window, shift, heads,
                    offset_scale=0.125
                )

        self.swin_block = SwinTransformerBlock(
            channels,
            channels // 12,
            [7, 7],
            [3 * is_shift, 3 * is_shift],
            attn_layer=attn_layer
        )

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.swin_block(x)
        return x.permute(0,3,1,2)



# -------------------------
# Spatial-Structure Calibration (C1): Edge-aware refinement block
#   - Edge extraction via simple sobel kernels (frozen)
#   - small fusion conv to modulate features
# -------------------------
class EdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-channel sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0.,  0.,  0.],
                                [1.,  2.,  1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # conv: (1→1)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        with torch.no_grad():
            self.conv_x.weight.copy_(sobel_x)
            self.conv_y.weight.copy_(sobel_y)

        # freeze parameters
        for p in self.conv_x.parameters(): p.requires_grad = False
        for p in self.conv_y.parameters(): p.requires_grad = False

    def forward(self, x):
        """
        x: B,C,H,W
        → convert to 1-channel by averaging (更适合医学影像 edge 提取)
        """
        gray = torch.mean(x, dim=1, keepdim=True)  # B,1,H,W

        gx = self.conv_x(gray)
        gy = self.conv_y(gray)

        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return edge


class SSCBlock(nn.Module):
    """
    SSC升级版：
    1) Attention-guided edge / skeleton enhancement
    2) Uncertainty-aware suppression
    输出：
      - calibrated feature
      - skeleton confidence map（供 CSFR 使用）
    """
    def __init__(self, in_ch):
        super().__init__()

        self.edge_extractor = EdgeExtractor()

        # skeleton approximation: edge → soft center confidence
        self.skel_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 1, 1),
            nn.Sigmoid()
        )

        # fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, in_ch // 4), in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.GroupNorm(max(1, in_ch // 4), in_ch),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))  # residual strength

    def forward(self, feat, attn_map):
        """
        feat: B,C,H,W
        attn_map: B,1,H,W  (from Attention)
        """
        # edge
        x_for_edge = feat[:, :3] if feat.shape[1] >= 3 else feat.mean(1, keepdim=True)
        edge = self.edge_extractor(x_for_edge)  # B,1,H,W

        # skeleton confidence
        skel = self.skel_conv(edge)

        # uncertainty (high = unreliable)
        uncert = self.uncertainty_head(feat)

        # attention-guided & uncertainty-aware structure prior
        struct_prior = skel * attn_map * (1.0 - uncert)

        fused = self.fuse(torch.cat([feat, struct_prior], dim=1))
        out = feat + self.alpha * fused

        return out, skel


class SkeletonConditionedCSFR(nn.Module):
    """
    Scale-aware Attention Router
    Skeleton-conditioned soft routing
    """
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2

        # three scale branches
        self.same = nn.Conv2d(channels, mid, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.down_conv = nn.Conv2d(channels, mid, 3, padding=1)
        self.up_conv = nn.Conv2d(channels, mid, 3, padding=1)

        # routing network (conditioned on skeleton)
        self.router = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 1)
        )

        self.proj = nn.Conv2d(mid, channels, 1)

    def forward(self, x, skel):
        """
        x: B,C,H,W
        skel: B,1,H,W  (from SSC)
        """
        # branches
        f_same = self.same(x)

        f_down = self.up(self.down_conv(self.down(x)))
        f_up = self.down(self.up_conv(self.up(x)))

        # ensure size match
        if f_down.shape[-2:] != x.shape[-2:]:
            f_down = F.interpolate(f_down, size=x.shape[-2:], mode="bilinear", align_corners=True)
        if f_up.shape[-2:] != x.shape[-2:]:
            f_up = F.interpolate(f_up, size=x.shape[-2:], mode="bilinear", align_corners=True)

        feats = torch.stack([f_same, f_down, f_up], dim=1)  # B,3,mid,H,W

        # skeleton-conditioned routing weights
        weights = self.router(skel)               # B,3,H,W
        weights = F.softmax(weights, dim=1)
        weights = weights.unsqueeze(2)             # B,3,1,H,W

        fused = (feats * weights).sum(dim=1)       # B,mid,H,W
        return self.proj(fused)



# -------------------------
class SwinSnake_Alter(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, kernel_size=5, extend_scope=3, layer_depth=5, rate=72, dim=1, repeat_n=1, down_layer="MaxPooling", device_id="0"):
        super().__init__()
        self.layer_depth = layer_depth
        device_id = "cuda:{}".format(device_id)
        basic_feature = [2 ** x for x in range(layer_depth)]
        basic_feature += basic_feature[:-1][::-1]

        # in/out channels (keeps same style as original)
        if down_layer == "MaxPooling":
            in_channels = [img_ch] + [x * rate for x in basic_feature[:layer_depth - 1]] + [3 * x * rate for x in basic_feature[-layer_depth + 1:]]
            out_channels = [x * rate for x in basic_feature]
            self.down_ops = nn.ModuleList([nn.MaxPool2d(2) for _ in range(layer_depth)])
        elif down_layer == "PatchMerging":
            in_channels = [img_ch] + [2 * x * rate for x in basic_feature[:layer_depth - 1]] + [3 * x * rate for x in basic_feature[-layer_depth + 1:]]
            out_channels = [x * rate for x in basic_feature]
            self.down_ops = nn.ModuleList([PatchMerging(x) for x in out_channels[:layer_depth]])
        else:
            raise ValueError("down_layer must be 'MaxPooling' or 'PatchMerging'")

        # build encoder DSConv stacks (use MultiView_DSConv which uses DA_DSConv)
        init_DSConvFusion = lambda in_ch, out_ch: nn.Sequential(
            MultiView_DSConv(in_ch, out_ch, kernel_size, extend_scope, device_id),
            *[MultiView_DSConv(out_ch, out_ch, kernel_size, extend_scope, device_id) for _ in range(max(0, repeat_n - 1))]
        )
        self.dsconvs = nn.ModuleList([init_DSConvFusion(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)])
        self.attn_probes = nn.ModuleList(
            [AttentionProbe(ch) for ch in out_channels]
        )

        # Deformable swin blocks for each stage
        # Hybrid: stage 0,1 use window directional; stage 2,3,4 use axial
        self.swins = nn.ModuleList()
        for i, out_ch in enumerate(out_channels):
            use_axial = (i >= 2)  # stage 2+ use axial attention

            block = nn.Sequential(*[
                DeformableSwinLayer(out_ch, is_shift=(i & 1), use_axial=use_axial)
                for _ in range(repeat_n)
            ])
            self.swins.append(block)

        # Cross-scale router and SSC per stage

        self.routers = nn.Sequential(
            *[SkeletonConditionedCSFR(out_ch) for out_ch in out_channels[:layer_depth]]
        )

        self.sscs = nn.ModuleList([SSCBlock(out_ch) for out_ch in out_channels])

        # decoder uses same dsconv modules in reverse (we will index into dsconvs accordingly)
        self.out_conv = nn.Conv2d(rate, output_ch, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        layer_depth = self.layer_depth
        enc_feats = []
        enc_skels = []

        feat = x

        # ========== Encoder ==========
        for i in range(layer_depth):
            # 1. local directional conv
            feat = self.dsconvs[i](feat)  # B,C,H,W

            # 2. attention (window / axial)
            feat = self.swins[i](feat)

            # 3. attention probe (structure saliency)
            attn_map = self.attn_probes[i](feat)

            # 4. SSC: structure + uncertainty calibration
            feat, skel = self.sscs[i](feat, attn_map)

            # 5. save skip features BEFORE routing
            if i < layer_depth - 1:
                enc_feats.append(feat)
                enc_skels.append(skel)

            # 6. skeleton-conditioned cross-scale routing
            feat = self.routers[i](feat, skel)

            # 7. downsample except bottleneck
            if i < layer_depth - 1:
                feat = self.down_ops[i](feat)

        # ========== Decoder ==========
        feat = self.up(feat)

        for i in range(1, layer_depth - 1)[::-1]:
            feat = torch.cat([feat, enc_feats[i]], dim=1)

            # ★ skeleton-guided decoder gating（关键）
            gate = torch.sigmoid(enc_skels[i])  # B,1,H,W
            feat = feat * (1 + gate)

            idx = 2 * (layer_depth - 1) - i
            if idx < len(self.dsconvs):
                feat = self.dsconvs[idx](feat)
            else:
                feat = Conv(feat.shape[1], feat.shape[1] // 2)(feat)

            feat = self.up(feat)

        # final stage
        feat = torch.cat([feat, enc_feats[0]], dim=1)

        # skeleton-guided refinement (use stage-0 skeleton)
        feat = feat * (1 + enc_skels[0])

        feat = self.dsconvs[2 * (layer_depth - 1)](feat)
        out = self.out_conv(feat)

        return self.sigmoid(out)


# -------------------------
# quick smoke test
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SwinSnake_Alter()
    size = 256
    try:
        summary(model, (1, 3, size, size))
    except Exception as e:
        print("summary failed (common if torchinfo not available), testing forward pass ...", e)
        x = torch.randn(1, 3, size, size).to(device)
        model = model.to(device)
        with torch.no_grad():
            y = model(x)
        print("output shape:", y.shape)
