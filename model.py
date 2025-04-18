import torch
from torch import nn
from einops import rearrange

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
    
class RDN_M(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers, **kwargs):
        super(RDN_M, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)

# Multi_Scale
class RDN_Multi_Scale(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN_Multi_Scale, self).__init__()
        
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        self.upscale2 = nn.Sequential(
            nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(2),
            nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        )
        self.upscale4 = nn.Sequential(
            nn.Conv2d(self.G0, self.G0 * (4 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(4),
            nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        )

        
        # down-sampling
        self.downscale2 = nn.Sequential(
            nn.Conv2d(self.G0, self.G0 // (2 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelUnshuffle(2),
            nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        )
        self.downscale4 = nn.Sequential(
            nn.Conv2d(self.G0, self.G0 // (4 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelUnshuffle(4),
            nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        )
        
        '''
        self.upscale8 = nn.Sequential(
            nn.Conv2d(self.G0, self.G0 * (8 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(8),
            nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        )
        self.downscale8 = nn.Sequential(
            nn.Conv2d(self.G0, self.G0 // (8 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelUnshuffle(8),
            nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        )
        '''
    
    def forward_up(self, x, scale_factor):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1
        if scale_factor == 2:
            x = self.upscale2(x)
        elif scale_factor == 4:
            x = self.upscale4(x)
        return x
    
    def forward_down(self, x, scale_factor):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1
        if scale_factor == 2:
            x = self.downscale2(x)
        elif scale_factor == 4:
            x = self.downscale4(x)
        return x
    
    
    def forward(self, x,scale_factors = [2,4]):
        ups1,downs1,downs2,ups2 = [],[],[],[]
        for s in scale_factors:
            up1 = self.forward_up(x,s)
            down1 = self.forward_down(up1,s)
            
            down2 = self.forward_down(x,s)
            up2 = self.forward_up(down2,s)
            
            ups1.append(up1)
            downs1.append(down1)
            downs2.append(down2)
            ups2.append(up2)
        return ups1,downs1,downs2,ups2



# For HAB
import math
from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn.init import trunc_normal_

#from basicsr.utils.registry import ARCH_REGISTRY
#from basicsr.archs.arch_util import to_2tuple, trunc_normal_


def to_ntuple(n, x):
    return x if isinstance(x, (tuple, list)) else (x,) * n

to_2tuple = partial(to_ntuple, 2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):

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
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
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
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
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

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
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
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# For RDN_HAB
def calculate_rpi_sa(window_size):
    # calculate relative position index for SA
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index

def calculate_mask(x_size,window_size,shift_size):
    # calculate attention mask for SW-MSA
    h, w = x_size
    img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
    h_slices = (slice(0, -window_size), slice(-window_size,
                                                    -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size,
                                                    -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nw, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class RDN_HAB(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers,window_size = 8,**kwargs):
        super(RDN_HAB, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.window_size = window_size
        
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        # HAB        
        self.habs = nn.ModuleList()
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=0,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=self.window_size//2,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))
        
        self.attn_mask_64 = calculate_mask((64,64),self.window_size,self.window_size//2)
        self.attn_mask_32 = calculate_mask((32,32),self.window_size,self.window_size//2)
        self.attn_mask_16 = calculate_mask((16,16),self.window_size,self.window_size//2)
        self.rpi_sa = calculate_rpi_sa(self.window_size)
        
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        b,c,h,w = x.shape
        if h == 64 and w ==64:
            attn_mask = self.attn_mask_64
        elif h == 32 and w ==32:
            attn_mask = self.attn_mask_32
        elif h == 16 and w ==16:
            attn_mask = self.attn_mask_16
        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) 
        x = rearrange(x,"b c h w -> b (h w) c")
        x = self.habs[0](x,(h,w),self.rpi_sa,None)
        x = self.habs[1](x,(h,w),self.rpi_sa,attn_mask)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = x + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
               
class RDN_HAB_M(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers,window_size = 8, **kwargs):
        super(RDN_HAB_M, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.window_size = window_size
        
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        # HAB        
        self.habs = nn.ModuleList()
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=0,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=self.window_size//2,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))

        self.attn_mask_64 = calculate_mask((64,64),self.window_size,self.window_size//2)
        self.attn_mask_32 = calculate_mask((32,32),self.window_size,self.window_size//2)
        self.attn_mask_16 = calculate_mask((16,16),self.window_size,self.window_size//2)
        self.rpi_sa = calculate_rpi_sa(self.window_size)
        
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        b,c,h,w = x.shape
        if h == 64 and w ==64:
            attn_mask = self.attn_mask_64
        elif h == 32 and w ==32:
            attn_mask = self.attn_mask_32
        elif h == 16 and w ==16:
            attn_mask = self.attn_mask_16
        
        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) 
        x = rearrange(x,"b c h w -> b (h w) c")
        x = self.habs[0](x,(h,w),self.rpi_sa,None)
        x = self.habs[1](x,(h,w),self.rpi_sa,attn_mask)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = x + sfe1  # global residual learning
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
    
    
    
class RDB_HAB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers,window_size = 8,shift =None, **kwargs):
        super(RDB_HAB, self).__init__()
        self.shift = shift
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        self.window_size = window_size
        
        self.habs = nn.ModuleList()
        if self.shift is None or self.shift == False:
            self.habs.append(HAB(dim = growth_rate,  # input_dim
                                input_resolution = (10000,10000),
                                num_heads=4, # 4
                                window_size=self.window_size,
                                shift_size=0,
                                compress_ratio=4,
                                squeeze_factor=8,# 8,16
                                conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                                drop=0.,
                                attn_drop=0.,
                                drop_path=0.,
                                act_layer=nn.GELU,
                                norm_layer=nn.LayerNorm))
        if self.shift is None or self.shift == True:
            self.habs.append(HAB(dim = growth_rate,  # input_dim
                                input_resolution = (10000,10000),
                                num_heads=4, # 4
                                window_size=self.window_size,
                                shift_size=self.window_size//2,
                                compress_ratio=4,
                                squeeze_factor=8,# 8,16
                                conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                                drop=0.,
                                attn_drop=0.,
                                drop_path=0.,
                                act_layer=nn.GELU,
                                norm_layer=nn.LayerNorm))
        
        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x,attn_mask,rpi_sa):
        b,c,h,w = x.shape
        
        x_ = self.lff(self.layers(x))
        
        x_ = rearrange(x_,"b c h w -> b (h w) c")
        if self.shift is None:
            x_ = self.habs[0](x_,(h,w),rpi_sa,None)
            x_ = self.habs[1](x_,(h,w),rpi_sa,attn_mask)
        elif self.shift is False:
            x_ = self.habs[0](x_,(h,w),rpi_sa,None)
        elif self.shift is True:
            x_ = self.habs[0](x_,(h,w),rpi_sa,attn_mask)
        x_ = rearrange(x_, "b (h w) c -> b c h w", h=h, w=w)
        
        return x + x_  # local residual learning
    

class RDN_HABs(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers,window_size = 8, **kwargs):
        super(RDN_HABs, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.window_size = window_size

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB_HAB(self.G0, self.G, self.C,window_size = self.window_size)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB_HAB(self.G, self.G, self.C,window_size = self.window_size))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        # HAB
        '''
        self.habs = nn.ModuleList()
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=0,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=self.window_size//2,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))
        '''
        attn_mask_64 = calculate_mask((64,64),self.window_size,self.window_size//2)
        attn_mask_32 = calculate_mask((32,32),self.window_size,self.window_size//2)
        attn_mask_16 = calculate_mask((16,16),self.window_size,self.window_size//2)
        rpi_sa = calculate_rpi_sa(self.window_size)
        
        self.register_buffer('attn_mask_64', attn_mask_64)
        self.register_buffer('attn_mask_32', attn_mask_32)
        self.register_buffer('attn_mask_16', attn_mask_16)
        self.register_buffer('rpi_sa', rpi_sa)
        
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        b,c,h,w = x.shape
        if h == 64 and w ==64:
            attn_mask = self.attn_mask_64
        elif h == 32 and w ==32:
            attn_mask = self.attn_mask_32
        elif h == 16 and w ==16:
            attn_mask = self.attn_mask_16
        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x,attn_mask,self.rpi_sa)
            local_features.append(x)
        
        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        
        #x = self.gff(torch.cat(local_features, 1)) 
        #x = rearrange(x,"b c h w -> b (h w) c")
        #x = self.habs[0](x,(h,w),self.rpi_sa,None)
        #x = self.habs[1](x,(h,w),self.rpi_sa,attn_mask)
        #x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        #x = x + sfe1  # global residual learning
        
        x = self.upscale(x)
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
               
class RDN_HABs_M(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers,window_size = 8, **kwargs):
        super(RDN_HABs_M, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.window_size = window_size
        
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB_HAB(self.G0, self.G, self.C,window_size = self.window_size)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB_HAB(self.G, self.G, self.C,window_size = self.window_size))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        # HAB        
        '''
        self.habs = nn.ModuleList()
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=0,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))
        self.habs.append(HAB(dim = growth_rate,  # input_dim
                            input_resolution = (10000,10000),
                            num_heads=4, # 4
                            window_size=self.window_size,
                            shift_size=self.window_size//2,
                            compress_ratio=4,
                            squeeze_factor=8,# 8,16
                            conv_scale=0.01,# To avoid the possible conflict of CAB and MSA on optimization and visual representation, a small constant α is multiplied to the output of CAB. 
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,# Override default qk scale of head_dim ** -0.5 if set
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm))
        '''
        attn_mask_64 = calculate_mask((64,64),self.window_size,self.window_size//2)
        attn_mask_32 = calculate_mask((32,32),self.window_size,self.window_size//2)
        attn_mask_16 = calculate_mask((16,16),self.window_size,self.window_size//2)
        rpi_sa = calculate_rpi_sa(self.window_size)
        
        self.register_buffer('attn_mask_64', attn_mask_64)
        self.register_buffer('attn_mask_32', attn_mask_32)
        self.register_buffer('attn_mask_16', attn_mask_16)
        self.register_buffer('rpi_sa', rpi_sa)
        
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        b,c,h,w = x.shape
        if h == 64 and w ==64:
            attn_mask = self.attn_mask_64
        elif h == 32 and w ==32:
            attn_mask = self.attn_mask_32
        elif h == 16 and w ==16:
            attn_mask = self.attn_mask_16
        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x,attn_mask,self.rpi_sa)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        
        #x = self.gff(torch.cat(local_features, 1)) 
        #x = rearrange(x,"b c h w -> b (h w) c")
        #x = self.habs[0](x,(h,w),self.rpi_sa,None)
        #x = self.habs[1](x,(h,w),self.rpi_sa,attn_mask)
        #x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        #x = x + sfe1  # global residual learning
        
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
    
    
class RDN_HABs_M2(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers,window_size = 8, **kwargs):
        super(RDN_HABs_M2, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.window_size = window_size
        
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        assert (self.D % 2 ) == 0, "D should be even"
        for _ in range(self.D - 1):
            if (_ % 2) == 0:
                self.rdbs.append(RDB_HAB(self.G, self.G, self.C,window_size = self.window_size))
            else:
                self.rdbs.append(RDB(self.G, self.G, self.C))
                
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        attn_mask_64 = calculate_mask((64,64),self.window_size,self.window_size//2)
        attn_mask_32 = calculate_mask((32,32),self.window_size,self.window_size//2)
        attn_mask_16 = calculate_mask((16,16),self.window_size,self.window_size//2)
        rpi_sa = calculate_rpi_sa(self.window_size)
        
        self.register_buffer('attn_mask_64', attn_mask_64)
        self.register_buffer('attn_mask_32', attn_mask_32)
        self.register_buffer('attn_mask_16', attn_mask_16)
        self.register_buffer('rpi_sa', rpi_sa)
        
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        b,c,h,w = x.shape
        if h == 64 and w ==64:
            attn_mask = self.attn_mask_64
        elif h == 32 and w ==32:
            attn_mask = self.attn_mask_32
        elif h == 16 and w ==16:
            attn_mask = self.attn_mask_16
        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            if isinstance(self.rdbs[i],RDB_HAB):
                x = self.rdbs[i](x,attn_mask,self.rpi_sa)
            else:
                x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x) 
    

class RDN_HABs_M3(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers,window_size = 8, **kwargs):
        super(RDN_HABs_M3, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.window_size = window_size
        
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB_HAB(self.G, self.G, self.C,window_size = self.window_size,shift = False)])
        assert (self.D % 2 ) == 0, "D should be even"
        for _ in range(self.D - 1):
            if (_ % 2) == 0:
                self.rdbs.append(RDB_HAB(self.G, self.G, self.C,window_size = self.window_size,shift = True))
            else:
                self.rdbs.append(RDB_HAB(self.G, self.G, self.C,window_size = self.window_size,shift = False))
                
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        attn_mask_64 = calculate_mask((64,64),self.window_size,self.window_size//2)
        attn_mask_32 = calculate_mask((32,32),self.window_size,self.window_size//2)
        attn_mask_16 = calculate_mask((16,16),self.window_size,self.window_size//2)
        rpi_sa = calculate_rpi_sa(self.window_size)
        
        self.register_buffer('attn_mask_64', attn_mask_64)
        self.register_buffer('attn_mask_32', attn_mask_32)
        self.register_buffer('attn_mask_16', attn_mask_16)
        self.register_buffer('rpi_sa', rpi_sa)
        
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        b,c,h,w = x.shape
        if h == 64 and w ==64:
            attn_mask = self.attn_mask_64
        elif h == 32 and w ==32:
            attn_mask = self.attn_mask_32
        elif h == 16 and w ==16:
            attn_mask = self.attn_mask_16
        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            if isinstance(self.rdbs[i],RDB_HAB):
                x = self.rdbs[i](x,attn_mask,self.rpi_sa)
            else:
                x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x) 
    
# # RDN+ Span

class RDB_SPAN(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB_SPAN, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        x_ = self.lff(self.layers(x))
        sim_att = torch.sigmoid(x_) - 0.5
        
        return (x + x_) * sim_att # local residual learning
    


class RDN_SPANs(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN_SPANs, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB_SPAN(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB_SPAN(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        # original model, but not good, so commented
        # x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.gff(torch.cat(local_features, 1))
        sim_att = torch.sigmoid(x) - 0.5
        x = (x + sfe1) * sim_att  # global residual learning
        
        x = self.upscale(x)
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
               
class RDN_SPANs_M(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN_SPANs_M, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB_SPAN(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB_SPAN(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        # original model, but not good, so commented
        # x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.gff(torch.cat(local_features, 1))
        sim_att = torch.sigmoid(x) - 0.5
        x = (x + sfe1) * sim_att  # global residual learning
        
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
    

# Each 2 for one Span    
class RDN_SPANs2(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN_SPANs2, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        assert (self.D % 2 ) == 0, "D should be even"
        for _ in range(self.D - 1):
            if (_ % 2) == 0:
                self.rdbs.append(RDB_SPAN(self.G, self.G, self.C))
            else:
                self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D // 2, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            if not isinstance(self.rdbs[i],RDB):
                local_features.append(x)
        
        # original model, but not good, so commented
        # x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.gff(torch.cat(local_features, 1))
        sim_att = torch.sigmoid(x) - 0.5
        x = (x + sfe1) * sim_att  # global residual learning
        
        x = self.upscale(x)
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)
               
class RDN_SPANs2_M(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN_SPANs2_M, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        assert (self.D % 2 ) == 0, "D should be even"
        for _ in range(self.D - 1):
            if (_ % 2) == 0:
                self.rdbs.append(RDB_SPAN(self.G, self.G, self.C))
            else:
                self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D // 2, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):        
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            if not isinstance(self.rdbs[i],RDB):
                local_features.append(x)
        
        # original model, but not good, so commented
        # x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.gff(torch.cat(local_features, 1))
        sim_att = torch.sigmoid(x) - 0.5
        x = (x + sfe1) * sim_att  # global residual learning
        
        x = self.output(x)
        return x
    
    def sample(self, x):
        return self.forward(x)