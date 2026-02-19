import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from model.MaxViT.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from model.MaxViT.maxxvit_4out import maxvit_rmlp_tiny_rw_256
from model.MaxViT.maxxvit_4out import MaxxVitCfg
from model.MaxViT.lib.helpers import named_apply
from model.MaxViT.lib.layers import ClassifierHead
from model.fusion_block import FusionBlock
from functools import partial

# MaxViT_small Encoder + SwinunetV2 Cross_attention Decoder + multi-task Distillation & multi-scale Aggregation

class Mlp(nn.Module):
    """The Mlp class is a standard two-layer feed-forward network with an activation and dropout after each layer."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    The window_partition function is a utility used in vision transformer architectures (like Swin Transformer and MaxViT) to split a feature map into smaller, non-overlapping windows.
    This is essential for applying window-based self-attention, where attention is computed within local windows rather than globally.
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    The window_reverse function is the inverse of window_partition. 
    It takes the small, non-overlapping windows (created by window_partition) and reconstructs the original feature map by placing each window back in its correct spatial location.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp_Relu(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(Mlp_Relu, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class WindowAttentionUp(nn.Module):
    r""" 
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Swin v1
        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        # Swin v1
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        # Swin v2, log-spaced coordinates, Eq.(4)
        # log_relative_position_index = torch.mul(torch.sign(relative_coords), torch.log(torch.abs(relative_coords) + 1))
        log_relative_position_index = torch.sign(relative_coords) * torch.log(1. + relative_coords.abs())
        self.register_buffer("log_relative_position_index", log_relative_position_index)

        # Swin v2, small meta network, Eq.(3)
        self.cpb = Mlp_Relu(in_features=2,  # delta x, delta y
                            hidden_features=256,  # hidden dims
                            out_features=self.num_heads,
                            dropout=0.0)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Swin v1
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # Swin v2, Scaled cosine attention
        self.tau = nn.Parameter(torch.ones((num_heads, window_size[0] * window_size[1],
                                            window_size[0] * window_size[1])))

    def get_continuous_relative_position_bias(self, N):
        # The continuous position bias approach adopts a small meta network on the relative coordinates
        continuous_relative_position_bias = self.cpb(self.log_relative_position_index[:N, :N])
        return continuous_relative_position_bias

    def forward(self, x, KV, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(KV).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]


        # Swin v1
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))

        # Swin v2, Scaled cosine attention
        # q = q * self.scale
        # qk = q @ k.transpose(-2, -1)
        # q2 = torch.mul(q, q).sum(-1).sqrt().unsqueeze(3)
        # k2 = torch.mul(k, k).sum(-1).sqrt().unsqueeze(3)
        # attn = qk / torch.clip(q2 @ k2.transpose(-2, -1), min=1e-6)
        # attn = attn / torch.clip(self.tau[:, :N, :N].unsqueeze(0), min=0.01)

        # Swin v2, Scaled cosine attention
        q = q * self.scale
        attn = torch.einsum("bhqd, bhkd -> bhqk", q, k) / torch.maximum(
            torch.norm(q, dim=-1, keepdim=True) * torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1),
            torch.tensor(1e-06, device=q.device, dtype=q.dtype))
        attn = attn / torch.clip(self.tau[:, :N, :N].unsqueeze(0), min=0.01)

        # Swin v1
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        # Swin v2
        relative_position_bias = self.get_continuous_relative_position_bias(N)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlockUp(nn.Module):
    r""" Swin Transformer Block.
    It uses window-based (and optionally shifted) self-attention, followed by a feed-forward network, with normalization and residual connections.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # self.norm1 = norm_layer(dim)
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.attn = WindowAttentionUp(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, KV):
        H, W = self.input_resolution
        B, L, C = x.shape
        B1, L1, C1 = KV.shape
        assert L == H * W, "input feature has wrong size"
        assert (B1 == B and L1 == L and C1 == C), "x and KV have different size"

        shortcut = x

        # Swin v1
        # x = self.norm1(x)

        x = x.view(B, H, W, C)
        kv = KV.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_kv = torch.roll(kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_kv = kv

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        kv_windows = window_partition(shifted_kv, self.window_size)  # nW*B, window_size, window_size, C
        kv_windows = kv_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, kv_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        kv = kv.view(B, H * W, C)

        # Swin v2
        x = self.norm1_1(x)
        kv = self.norm1_2(kv)

        # FFN
        x = shortcut + self.drop_path(x)

        # Swin v1
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Swin v2
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x, kv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchExpand(nn.Module):
    """In transformer-based decoders, after encoding an image into a low-resolution feature map, you need to progressively upsample it to reconstruct a high-resolution output (e.g., a segmentation mask).
    Both the main features and the key/value features used in attention must be upsampled to maintain alignment.
    PatchExpand is used to upsample (increase the spatial resolution of) a feature map, typically in the decoder of a vision transformer.
    It is analogous to the "patch merging" operation in the encoder, but in reverse: it increases the height and width by a factor (usually 2), reducing the number of channels accordingly."""
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class KVExpand(nn.Module):
    """KVExpand is similar to PatchExpand, but is specifically used for upsampling the key/value (KV) feature maps in the decoder.
    In multi-scale or multi-task transformer decoders, you may need to upsample both the main feature map and the key/value maps used for attention."""
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution

    def forward(self, KV):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = KV.shape
        assert L == H * W, "input feature has wrong size"

        KV = KV.permute(0, 2, 1)
        KV = KV.reshape(B, C, H, W)
        KV = F.interpolate(KV, 2 * H, mode='bilinear', align_corners=False)
        KV = KV.reshape(B, C, -1)
        KV = KV.permute(0, 2, 1)
        KV = F.interpolate(KV, int(C / 2), mode='linear', align_corners=False)

        return KV


class FirstLayerExpand(nn.Module):
    """FirstLayerExpand is a utility module that, at the first upsampling stage of the decoder, increases the spatial resolution of both the main and key/value feature maps, preparing them for further decoding and attention operations."""
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.norm_layer = norm_layer

        self.patchexpand = PatchExpand(self.input_resolution, self.dim, self.dim_scale, self.norm_layer)
        self.kvexpand = KVExpand(self.input_resolution, self.dim, self.dim_scale, self.norm_layer)

    def forward(self, x, KV):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = KV.shape
        assert L == H * W, "input feature has wrong size"

        x = self.patchexpand(x)
        KV = self.kvexpand(KV)

        return x, KV


class MultiModalDistallationExpand(nn.Module):
    """MultiModalDistallationExpand is a module designed to upsample and fuse multi-modal (multi-task) features in the decoder of a vision transformer.
    It is typically used in multi-task learning settings, such as segmentation + contour detection, where features from different tasks or scales need to be upsampled and combined."""
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.permute(0, 3, 1, 2)

        return x


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    A decoder stage that stacks several Swin Transformer blocks and upsamples the output, preparing features for the next stage or for final output in a segmentation or multi-task vision transformer.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlockUp(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   # shift_size=0,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            self.kvupsample = KVExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, KV):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, KV)
            else:
                x, KV = blk(x, KV)
        decoder_feature = x
        if self.upsample is not None:
            x = self.upsample(x)
            KV = self.kvupsample(KV)
        return x, KV, decoder_feature


class ChannelAttention(nn.Module):
    """The ChannelAttention module focuses on which channels (feature maps) are important for the current task.
    Channels that are more important for the task are emphasized, while less important ones are suppressed."""
    def __init__(self, in_planes, rotio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    """The SpatialAttention module focuses on which spatial locations (pixels or regions) are important in the feature map.
    Spatial regions that are more important for the task are emphasized, while less important ones are suppressed."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class cbam(nn.Module):
    """CBAM is a lightweight, plug-and-play attention module for vision transformers.
    It sequentially applies Channel Attention and Spatial Attention to refine feature maps, helping the network focus on "what" and "where" to emphasize in the feature representation.
    Channel Attention: Learns "what" to focus on (which feature maps are important).
    Spatial Attention: Learns "where" to focus (which spatial locations are important).
    Sequential application: First channel, then spatial attention, for maximum effect."""
    def __init__(self, planes, rotio):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(planes, rotio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class MultitaskDistillation(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    Multi-Task Multi-Scale distillation
    in this work, img_size=224*224, window_size=7
    decoder feature map shape= [batch, resolution * resolution, Channel]
    torch.Size([4, 49, 768])
    torch.Size([4, 196, 384])
    torch.Size([4, 784, 192])
    torch.Size([4, 3136, 96])

    MultiModalDistallationExpand:
    """

    def __init__(self):
        super(MultitaskDistillation, self).__init__()
        self.multi_modal_up1_1 = MultiModalDistallationExpand(input_resolution=[8, 8], dim=MaxxVitCfg.embed_dim[3])
        self.multi_modal_up2_1 = MultiModalDistallationExpand(input_resolution=[8, 8], dim=MaxxVitCfg.embed_dim[3])

        self.multi_modal_up1_2 = MultiModalDistallationExpand(input_resolution=[16, 16], dim=MaxxVitCfg.embed_dim[2])
        self.multi_modal_up2_2 = MultiModalDistallationExpand(input_resolution=[16, 16], dim=MaxxVitCfg.embed_dim[2])

        self.multi_modal_up1_3 = MultiModalDistallationExpand(input_resolution=[32, 32], dim=MaxxVitCfg.embed_dim[1])
        self.multi_modal_up2_3 = MultiModalDistallationExpand(input_resolution=[32, 32], dim=MaxxVitCfg.embed_dim[1])

        self.multi_modal_up1_4 = MultiModalDistallationExpand(input_resolution=[64, 64], dim=MaxxVitCfg.embed_dim[0])
        self.multi_modal_up2_4 = MultiModalDistallationExpand(input_resolution=[64, 64], dim=MaxxVitCfg.embed_dim[0])

        self.CBAM_contour2seg1 = cbam(planes=MaxxVitCfg.embed_dim[3], rotio=16)
        self.CBAM_seg2contour1 = cbam(planes=MaxxVitCfg.embed_dim[3], rotio=16)

        self.CBAM_contour2seg2 = cbam(planes=MaxxVitCfg.embed_dim[2], rotio=16)
        self.CBAM_seg2contour2 = cbam(planes=MaxxVitCfg.embed_dim[2], rotio=16)

        self.CBAM_contour2seg3 = cbam(planes=MaxxVitCfg.embed_dim[1], rotio=16)
        self.CBAM_seg2contour3 = cbam(planes=MaxxVitCfg.embed_dim[1], rotio=16)

        self.CBAM_contour2seg4 = cbam(planes=MaxxVitCfg.embed_dim[0], rotio=16)
        self.CBAM_seg2contour4 = cbam(planes=MaxxVitCfg.embed_dim[0], rotio=16)

        self.sigmoid = nn.Sigmoid()

    def forward(self, task1_feature, task2_feature):  # multi-task distillation + interpolate + concat  sub + 2CBAM
        x1_1 = self.multi_modal_up1_1(task1_feature[0])
        x2_1 = self.multi_modal_up2_1(task2_feature[0])
        x1_2 = self.multi_modal_up1_2(task1_feature[1])
        x2_2 = self.multi_modal_up2_2(task2_feature[1])
        x1_3 = self.multi_modal_up1_3(task1_feature[2])
        x2_3 = self.multi_modal_up2_3(task2_feature[2])
        x1_4 = self.multi_modal_up1_4(task1_feature[3])
        x2_4 = self.multi_modal_up2_4(task2_feature[3])

        feature1 = torch.abs(x1_1 - x2_1)  # Substraction with multiple CBAM
        x1_out1 = x1_1 + self.sigmoid(self.CBAM_contour2seg1(feature1))
        x2_out1 = x2_1 + self.sigmoid(self.CBAM_seg2contour1(feature1))

        x1_out1 = F.interpolate(x1_out1, size=256, mode='bilinear', align_corners=True)
        x2_out1 = F.interpolate(x2_out1, size=256, mode='bilinear', align_corners=True)
        feature2 = torch.abs(x1_2 - x2_2)
        x1_out2 = x1_2 + self.sigmoid(self.CBAM_contour2seg2(feature2))
        x2_out2 = x2_2 + self.sigmoid(self.CBAM_seg2contour2(feature2))

        x1_out2 = F.interpolate(x1_out2, size=256, mode='bilinear', align_corners=True)
        x2_out2 = F.interpolate(x2_out2, size=256, mode='bilinear', align_corners=True)
        feature3 = torch.abs(x1_3 - x2_3)
        x1_out3 = x1_3 + self.sigmoid(self.CBAM_contour2seg3(feature3))
        x2_out3 = x2_3 + self.sigmoid(self.CBAM_seg2contour3(feature3))

        x1_out3 = F.interpolate(x1_out3, size=256, mode='bilinear', align_corners=True)
        x2_out3 = F.interpolate(x2_out3, size=256, mode='bilinear', align_corners=True)
        feature4 = torch.abs(x1_4 - x2_4)
        x1_out4 = x1_4 + self.sigmoid(self.CBAM_contour2seg4(feature4))
        x2_out4 = x2_4 + self.sigmoid(self.CBAM_seg2contour4(feature4))

        x1_out = torch.cat([x1_out1, x1_out2, x1_out3, x1_out4], dim=1)
        x2_out = torch.cat([x2_out1, x2_out2, x2_out3, x2_out4], dim=1)
        return x1_out, x2_out

class MT_MaxViT(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", has_dropout=False, **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.has_dropout = has_dropout
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0]*patches_resolution[1]
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        """Extracts hierarchical features from the input image at multiple scales."""
        self.backbone = maxvit_rmlp_tiny_rw_256(pretrained=True)  # use maxxivit and pretrained model
        # print('Loading: model/pretrained_pth/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        # state_dict = torch.load('model/pretrained_pth/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        # self.backbone.load_state_dict(state_dict, strict=False)

        """The decoder is built as a sequence of upsampling stages, each consisting of:
            BasicLayer_up or FirstLayerExpand (for upsampling and Swin-style attention)
            Linear layers for feature concatenation and channel adjustment
            Multi-task distillation modules for fusing features from different tasks
        For each decoder stage:
            The spatial resolution is increased (upsampled).
            Features from the encoder and previous decoder stage are fused.
            Multi-task features are processed in parallel (e.g., for segmentation and contour)."""
        # build decoder layers
        self.layers_up1 = nn.ModuleList()
        self.layers_up2 = nn.ModuleList()

        self.concat_back_dim1 = nn.ModuleList()
        self.concat_back_dim2 = nn.ModuleList()
        self.concat_back_KV = nn.ModuleList()
        for i_layer in range(self.num_layers):  # in range(4)
            concat_linear1 = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            concat_linear2 = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            concat_linear_KV = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                layer_up1 = FirstLayerExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                layer_up2 = FirstLayerExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up1 = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
                layer_up2 = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          input_resolution=(
                                              patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                              patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                          depth=depths[(self.num_layers - 1 - i_layer)],
                                          num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                          window_size=window_size,
                                          mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                              depths[:(self.num_layers - 1 - i_layer) + 1])],
                                          norm_layer=norm_layer,
                                          upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                          use_checkpoint=use_checkpoint)
            self.layers_up1.append(layer_up1)
            self.layers_up2.append(layer_up2)
            self.concat_back_dim1.append(concat_linear1)
            self.concat_back_dim2.append(concat_linear2)
            self.concat_back_KV.append(concat_linear_KV)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.norm = norm_layer(self.num_features)
        self.norm_up1 = norm_layer(self.embed_dim)
        self.norm_up2 = norm_layer(self.embed_dim)

        """Separate output heads for each task (e.g., segmentation and contour)."""
        self.OutConv1 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=96, out_channels=self.num_classes, kernel_size=1, bias=False)
        )
        self.OutConv2 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=96, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=96, out_channels=2, kernel_size=1, bias=False)
        )
        self.Decoder_Feature1_SE = nn.Sequential(
            cbam(planes=MaxxVitCfg.embed_dim[2], rotio=16),
            cbam(planes=MaxxVitCfg.embed_dim[1], rotio=16),
            cbam(planes=MaxxVitCfg.embed_dim[0], rotio=16),
            cbam(planes=MaxxVitCfg.embed_dim[0], rotio=16)
        )
        self.Decoder_Feature2_SE = nn.Sequential(
            cbam(planes=MaxxVitCfg.embed_dim[2], rotio=16),
            cbam(planes=MaxxVitCfg.embed_dim[1], rotio=16),
            cbam(planes=MaxxVitCfg.embed_dim[0], rotio=16),
            cbam(planes=MaxxVitCfg.embed_dim[0], rotio=16)
        )
        self.multitask_distillation = MultitaskDistillation()

        self.fusion_block = FusionBlock(embed_dim=MaxxVitCfg.embed_dim[-1], norm_layer=norm_layer)

        self.classifier = ClassifierHead(in_chs=MaxxVitCfg.embed_dim[-1], num_classes=3, pool_type='', dropout=0.5)

        named_apply(partial(self._init_weights, scheme='vit_eff'), self)

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    """Passes input through the MaxViT encoder, returns multi-scale features."""
    # Encoder and Bottleneck
    def forward_features(self, x):
        x_downsample = self.backbone(x)
        
        for i in range(len(x_downsample)):
            B, N = x_downsample[i].shape[0], x_downsample[i].shape[1]
            x_downsample[i] = x_downsample[i].reshape([B, N, -1])
            x_downsample[i] = x_downsample[i].permute(0, 2, 1)

        feature_map = x_downsample[-1]

        return feature_map, x_downsample

    """Passes features through the hierarchical decoder, upsampling and fusing at each stage."""
    # Dencoder and Skip connection
    def forward_up_feature(self, x, x_downsample, KV):
        x_upsample1 = []
        x_upsample2 = []
        x_size = [16, 32, 64, 64]
        for inx in range(len(self.layers_up1)):
            if inx == 0:
                x_upsample1.append(x)
                x_upsample2.append(x)
                x1, KV1 = self.layers_up1[inx](x, KV)
                x2, KV2 = self.layers_up2[inx](x, KV)
            else:
                x1 = torch.cat([x1, x_downsample[3 - inx]], -1)
                KV1 = torch.cat([KV1, x_downsample[3 - inx]], -1)
                x1 = self.concat_back_dim1[inx](x1)
                KV1 = self.concat_back_KV[inx](KV1)
                x1, KV1, decoder_feature1 = self.layers_up1[inx](x1, KV1)
                x_upsample1.append(decoder_feature1)

                x2 = torch.cat([x2, x_downsample[3 - inx]], -1)
                KV2 = torch.cat([KV2, x_downsample[3 - inx]], -1)
                x2 = self.concat_back_dim2[inx](x2)
                KV2 = self.concat_back_KV[inx](KV2)
                x2, KV2, decoder_feature2 = self.layers_up2[inx](x2, KV2)
                x_upsample2.append(decoder_feature2)
            # # sum fusion
            # x1 = x1 + x2
            # x2 = x1 + x2
            # # concat + conv_back fusion
            # x1 = x1.permute(0, 2, 1)
            # x2 = x2.permute(0, 2, 1)
            # cat_feature = torch.cat([x1, x2], dim=1)
            # B, C = cat_feature.shape[0], cat_feature.shape[1]
            # cat_feature = cat_feature.reshape(B, C, x_size[inx], x_size[inx])
            # x1 = self.Decoder_Feature1_Conv[inx](cat_feature)
            # x2 = self.Decoder_Feature2_Conv[inx](cat_feature)
            # x1 = x1.reshape(B, C // 2, -1).permute(0, 2, 1)
            # x2 = x2.reshape(B, C // 2, -1).permute(0, 2, 1)
            # sum SE fusion
            x1 = x1.permute(0, 2, 1)
            x2 = x2.permute(0, 2, 1)
            cat_feature = x1 + x2
            B, C = cat_feature.shape[0], cat_feature.shape[1]
            cat_feature = cat_feature.reshape(B, C, x_size[inx], x_size[inx])
            x1 = self.Decoder_Feature1_SE[inx](cat_feature)
            x2 = self.Decoder_Feature2_SE[inx](cat_feature)
            x1 = x1.reshape(B, C, -1).permute(0, 2, 1)
            x2 = x2.reshape(B, C, -1).permute(0, 2, 1)

        return x_upsample1, x_upsample2

    def forward(self, x):
        """The main forward pass, combining encoding, decoding, distillation, and output.
        Input is passed through the MaxViT backbone to extract multi-scale features."""

        # feature_map: Deepest feature map (lowest resolution, highest abstraction).
        # x_downsample: List of features at different scales.
        feature_map, x_downsample = self.forward_features(x)
        KV = feature_map

        """Features are progressively upsampled and fused at each stage.
        Two parallel decoding paths for multi-task outputs (e.g., segmentation and contour)."""
        x_upsample1, x_upsample2 = self.forward_up_feature(feature_map, x_downsample, KV)
        """Multi-task distillation: Fuses features from both decoding paths using attention and upsampling.
        Output heads: Produce final predictions for each task.
        Dropout: Optionally applied for regularization.
        Returns: Outputs for both tasks (e.g., segmentation mask and contour map)."""
        out1, out2 = self.multitask_distillation(x_upsample1, x_upsample2)

        fused_feature = self.fusion_block(feature_map, out2)
        fused_feature = fused_feature.mean(dim=1)
        out3 = self.classifier(fused_feature)

        out1 = self.sigmoid(self.OutConv1(out1))
        out2 = self.sigmoid(self.OutConv2(out2))
        
        if self.has_dropout:
            out1 = self.dropout(out1)
            out2 = self.dropout(out2)

        return out1, out2, out3

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops