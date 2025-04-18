import math
import torch
from torch import nn, einsum
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange, reduce
from inspect import isfunction

############################################################################################
################################UNet Model##################################################
############################################################################################
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        if noise_embed is None:
            return x
        
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class BlockWithFWA(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)
        self.block = Block(dim, dim_out, groups=norm_groups, dropout=dropout)
    
    def forward(self, x, noise_level):
        x = self.block(x)
        return self.noise_func(x, noise_level)
    
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, noise_level):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, noise_level)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32, head_dim=None,mlp_ratio=None):
        super().__init__()

        self.n_head = n_head # h
        self.head_dim = head_dim # d
        self.mlp_ratio = mlp_ratio
        self.channel = n_head * head_dim if head_dim is not None else in_channel # c
        self.in_channel = in_channel # c_
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, self.channel, 1, bias=False) # b,c_,x,y -> b,c,x,y
        self.k = nn.Conv2d(in_channel, self.channel, 1, bias=False) # b,c_,x,y -> b,c,x,y
        self.v = nn.Conv2d(in_channel, self.channel, 1, bias=False) # b,c_,x,y -> b,c,x,y
        self.out = nn.Conv2d(self.channel, in_channel, 1)           # b,c,x,y  -> b,c_,x,y

        self.mlp = None
        if mlp_ratio is not None:
            self.norm2 = nn.GroupNorm(norm_groups, self.channel)
            hidden_dim = int(self.in_channel*mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Conv2d(self.in_channel, hidden_dim, 1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, self.in_channel, 1)
            )
            
    def forward(self, input):
        batch, channel_, height, width = input.shape # b,c_,x,y
        channel = self.channel # c
        n_head = self.n_head # h
        head_dim = channel // n_head # d = c/h

        norm = self.norm(input)
        query = self.q(norm)
        key = self.k(norm)
        value = self.v(norm)
        query = rearrange(query, "b (h d) x y -> b h (x y) d", h=n_head) # h*d = c, x*y = n, m = n
        key = rearrange(key, "b (h d) x y -> b h (x y) d", h=n_head)
        value = rearrange(value, "b (h d) x y -> b h (x y) d", h=n_head)
        
        attn = torch.einsum(
            "bhnd, bhmd -> bhnm", query, key                            # m = n
        ).contiguous() / math.sqrt(channel)
        attn = torch.softmax(attn, -1)
        out = torch.einsum("bhnm, bhmd -> bhnd", attn, value)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", h=n_head, x=height, y=width)
        out = self.out(out) + input
        
        if self.mlp is not None:
            out = self.mlp(self.norm2(out)) + out
        return out

############################################################################################
###############################Cross Attention##############################################
############################################################################################
class CrossAttention(nn.Module):
    def __init__(self, in_channel, condition_channel, n_head=1, norm_groups=32,head_dim=None,mlp_ratio=None):
        super().__init__()
        
        self.n_head = n_head # h
        self.head_dim = head_dim # d
        self.mlp_ratio = mlp_ratio
        self.channel = n_head * head_dim if head_dim is not None else in_channel # c
        self.in_channel = in_channel # c_
        self.condition_channel = condition_channel # c_cond
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, self.channel, 1, bias=False)         # b,c_,x1,y1     -> b,c,x1,y1
        self.k = nn.Conv2d(condition_channel, self.channel, 1, bias=False)  # b,c_cond,x2,y2 -> b,c,x2,y2
        self.v = nn.Conv2d(condition_channel, self.channel, 1, bias=False)  # b,c_cond,x2,y2 -> b,c,x2,y2
        self.out = nn.Conv2d(self.channel, in_channel, 1)                   # b,c,x1,y1      -> b,c_,x1,y1
        
        self.mlp = None
        if mlp_ratio is not None:
            self.norm2 = nn.GroupNorm(norm_groups, self.channel)
            hidden_dim = int(self.in_channel*mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Conv2d(self.in_channel, hidden_dim, 1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, self.in_channel, 1)
            )
        
    def forward(self, input, condition):
        batch, channel_, height, width = input.shape # b,c_,x1,y1
        channel = self.channel # c
        n_head = self.n_head # h
        head_dim = channel // n_head  # d = c/h

        norm = self.norm(input)
        query = self.q(norm)
        key = self.k(condition)
        value = self.v(condition)
        query = rearrange(query, "b (h d) x y -> b h (x y) d", h=n_head)    # h*d = c, x1*y1 = n
        key = rearrange(key, "b (h d) x y -> b h (x y) d", h=n_head)        # h*d = c, x2*y2 = m
        value = rearrange(value, "b (h d) x y -> b h (x y) d", h=n_head)    # h*d = c, x2*y2 = m
        
        attn = torch.einsum(
            "bhnd, bhmd -> bhnm", query, key                            # m != n
        ).contiguous() / math.sqrt(channel)
        attn = torch.softmax(attn, -1)
        out = torch.einsum("bhnm, bhmd -> bhnd", attn, value)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", h=n_head, x=height, y=width)
        out = self.out(out) + input
        
        if self.mlp is not None:
            out = self.mlp(self.norm2(out)) + out
        return out 

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None,n_head=1, 
                 norm_groups=32, dropout=0,use_affine_level=False, 
                 with_attn=False,head_dim = None,mlp_ratio=None,
                 cross_attn=False,condition_channel=None):
        super().__init__()
        self.with_attn = with_attn
        self.cross_attn = cross_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, use_affine_level=use_affine_level)
        if with_attn:
            if cross_attn:
                self.attn = CrossAttention(dim_out,condition_channel,n_head, norm_groups=norm_groups,head_dim=head_dim,mlp_ratio=mlp_ratio)
            elif not cross_attn:
                self.attn = SelfAttention(dim_out, n_head,norm_groups=norm_groups,head_dim=head_dim,mlp_ratio=mlp_ratio)

    def forward(self, x, noise_level, condition=None):
        x = self.res_block(x, noise_level)
        if self.with_attn:
            if self.cross_attn:
                x = self.attn(x, condition)
            elif not self.cross_attn:
                x = self.attn(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_channel=512, out_channel=64, kernel_size=7, stride=1, padding=3, norm_groups=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, in_channel)
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm2 = nn.GroupNorm(norm_groups, out_channel)

    def forward(self, x):
        x1 = self.proj(self.norm1(x))
        x = self.norm2(x)
        x = F.gelu(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        # UNet parameters
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8],
        with_noise_level_emb=True,
        use_affine_level=False,
        input_FWA = False,
        # Resnet parameters
        res_blocks=3,
        dropout=0,
        # Conditions
        conditions = [],
        concat_ori_LR=False,
        concat_ori_patch_emb=False,
        # x embedding parameters
        x_emb_out_channel=32,
        x_emb_layers=0,
        # Patch Embedding parameters
        patch_emb_in_channel=512,
        patch_emb_out_channel=64,
        patch_emb_layers=1,
        # Attention parameters
        attn_res=[8], # layer to use attention, the numbers in channel_mults
        patch_emb_cross_attn=False,
        patch_emb_concat=True,
        gene_emb_cross_attn=False,
        gene_emb_concat=True,
        n_head=1,
        head_dim = None,
        mlp_ratio=None,
        # marker_idx embedding parameters
        num_classes=8,
        # LR embedding parameters
        LR_emb_out_channel=None,
        LR_emb_layers=0,
        # LR and patch cross
        LR_patch_cross_attn=False
    ):
        super().__init__()
        # UNet
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.inner_channel = inner_channel
        self.norm_groups = norm_groups
        self.channel_mults = channel_mults
        self.with_noise_level_emb = with_noise_level_emb
        self.use_affine_level = use_affine_level
        self.input_FWA = input_FWA
        self.res_blocks = res_blocks
        self.dropout = dropout
        # Conditions
        self.conditions = conditions if isinstance(conditions, list) else [conditions]
        self.concat_ori_LR = concat_ori_LR
        self.concat_ori_patch_emb = concat_ori_patch_emb
        # x embedding
        self.x_emb_in_channel = out_channel
        self.x_emb_out_channel = x_emb_out_channel
        self.x_emb_layers = x_emb_layers if x_emb_layers is not None else 0
        if "x_emb" in self.conditions:
            if self.x_emb_out_channel is not None and self.x_emb_layers > 0:
                x_emb_channels = [out_channel] + [x_emb_out_channel] * x_emb_layers
                self.x_emb = nn.ModuleList()
                for i in range(x_emb_layers):
                    self.x_emb.append(ResnetBlocWithAttn(x_emb_channels[i],x_emb_channels[i+1],noise_level_emb_dim = inner_channel,
                                                         n_head=n_head,norm_groups=out_channel,use_affine_level=use_affine_level,
                                                         dropout=dropout,with_attn=True,cross_attn=False,
                                                         condition_channel=None,head_dim = None,mlp_ratio=4))        
        
        # LR and patch cross attn
        self.LR_patch_cross_attn = LR_patch_cross_attn
        # Patch Embedding
        self.patch_emb_in_channel = patch_emb_in_channel
        self.patch_emb_out_channel = patch_emb_out_channel
        self.patch_emb_layers = patch_emb_layers if patch_emb_layers is not None else 0
        if "patch_emb" in self.conditions:
            if self.patch_emb_out_channel is not None and self.patch_emb_layers > 0:
                patch_emb_channels= [patch_emb_in_channel] + [patch_emb_out_channel] * patch_emb_layers
                patch_emb_condition_channel = [out_channel] + [LR_emb_out_channel] * patch_emb_layers
                self.patch_emb = nn.ModuleList()
                for i in range(patch_emb_layers):
                    self.patch_emb.append(ResnetBlocWithAttn(patch_emb_channels[i],patch_emb_channels[i+1],noise_level_emb_dim = inner_channel,
                                                             n_head=n_head,norm_groups=norm_groups,use_affine_level=use_affine_level,
                                                             dropout=dropout,with_attn=True,cross_attn=LR_patch_cross_attn,
                                                             condition_channel=patch_emb_condition_channel[i],head_dim = None,mlp_ratio=4))
        # Attention
        self.attn_res = attn_res
        self.patch_emb_cross_attn = patch_emb_cross_attn if "patch_emb" in self.conditions else False
        self.patch_emb_concat = patch_emb_concat if "patch_emb" in self.conditions else False
        self.gene_emb_cross_attn = gene_emb_cross_attn if "LR" in self.conditions else False
        self.gene_emb_concat = gene_emb_concat if "LR" in self.conditions else False
        self.cross_attn = patch_emb_cross_attn or gene_emb_cross_attn
        self.condition_channel = patch_emb_out_channel + LR_emb_out_channel if self.patch_emb_cross_attn and self.gene_emb_cross_attn else patch_emb_out_channel if self.patch_emb_cross_attn else LR_emb_out_channel
        self.n_head = n_head
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        # marker_idx embedding
        self.num_classes = num_classes
        if "marker_idx" in self.conditions:
            self.marker_idx_emb = nn.Sequential(
                nn.Embedding(num_classes, inner_channel),
                nn.Linear(inner_channel, inner_channel*4),
                Swish(),
                nn.Linear(inner_channel*4, inner_channel)
            )
        # LR embedding
        self.LR_emb_out_channel = LR_emb_out_channel
        self.LR_emb_layers = LR_emb_layers if LR_emb_layers is not None else 0
        if "LR" in self.conditions:
            if self.LR_emb_out_channel is not None and self.LR_emb_layers > 0:
                LR_emb_channels = [out_channel] + [LR_emb_out_channel] * LR_emb_layers
                LR_emb_condition_channel = [patch_emb_in_channel] + [patch_emb_out_channel] * LR_emb_layers
                self.LR_emb = nn.ModuleList()
                for i in range(LR_emb_layers):
                    self.LR_emb.append(ResnetBlocWithAttn(LR_emb_channels[i],LR_emb_channels[i+1],noise_level_emb_dim = inner_channel,
                                   n_head=n_head,norm_groups=out_channel,use_affine_level=use_affine_level,                             # use out_channel as norm_groups
                                   dropout=dropout,with_attn=True,cross_attn=LR_patch_cross_attn,
                                   condition_channel=LR_emb_condition_channel[i],head_dim = None,mlp_ratio=4))
        ##################
        
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel*4),
                Swish(),
                nn.Linear(inner_channel*4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        # now_res = image_size

        downs = [nn.Conv2d(in_channel, inner_channel,kernel_size=3, padding=1)] if not self.input_FWA else [ResnetBlock(in_channel, inner_channel, noise_level_emb_dim=inner_channel, dropout=dropout, use_affine_level=use_affine_level, norm_groups=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (ind in attn_res)
            channel_mult = inner_channel*channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,n_head=n_head,
                    norm_groups=norm_groups,use_affine_level=use_affine_level, 
                    dropout=dropout, with_attn=use_attn,cross_attn=self.cross_attn,
                    condition_channel=self.condition_channel,head_dim = head_dim,mlp_ratio=mlp_ratio))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                # now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, n_head=n_head,
                               norm_groups=norm_groups,use_affine_level=use_affine_level,
                               dropout=dropout, with_attn=True,cross_attn=self.cross_attn,
                               condition_channel=self.condition_channel,head_dim = head_dim,mlp_ratio=4),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, n_head=n_head,
                               norm_groups=norm_groups,use_affine_level=use_affine_level,
                               dropout=dropout, with_attn=True,cross_attn=self.cross_attn,
                               condition_channel=self.condition_channel,head_dim = head_dim,mlp_ratio=4)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (ind in attn_res)
            channel_mult = inner_channel*channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, n_head=n_head,
                    norm_groups=norm_groups,use_affine_level=use_affine_level,
                    dropout=dropout, with_attn=use_attn,cross_attn=self.cross_attn,
                    condition_channel=self.condition_channel,head_dim = head_dim,mlp_ratio=mlp_ratio))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                # now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups) if not self.input_FWA else ResnetBlock(inner_channel, out_channel, noise_level_emb_dim=inner_channel, dropout=dropout, use_affine_level=use_affine_level, norm_groups=1)
        
        # self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if exists(m.bias):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if exists(m.bias):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        for m in self.final_conv.block:
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if exists(m.bias):
                    nn.init.zeros_(m.bias)
        
    def get_condition(self, x,cond_data,noise_level=None):
        marker_idx = None
        if "marker_idx" in self.conditions:
            marker_idx = self.marker_idx_emb(cond_data.marker_idx.reshape(-1))
        t = None
        if noise_level is not None:
            t = self.noise_level_mlp(noise_level)
        if marker_idx is not None:
            t = t + marker_idx if t is not None else marker_idx
        
        LR = None
        patch_emb = None
        if "LR" in self.conditions and "patch_emb" in self.conditions:
            LR = cond_data.LR
            patch_emb = cond_data.patch_emb
            if self.patch_emb_layers > 0:
                if self.LR_emb_layers > 0:
                    if self.patch_emb_layers == self.LR_emb_layers:
                        for layer_LR,layer_patch_emb in zip(self.LR_emb,self.patch_emb):
                            LR_ = layer_LR(LR,marker_idx,condition=patch_emb)
                            patch_emb_ = layer_patch_emb(patch_emb,marker_idx,condition=LR)
                            LR,patch_emb = LR_,patch_emb_
                    else:
                        for i,layer in enumerate(self.LR_emb):
                            LR = layer(LR,marker_idx,condition=patch_emb)
                        for i,layer in enumerate(self.patch_emb):
                            patch_emb = layer(patch_emb,marker_idx,condition=LR)
                elif self.LR_emb_layers == 0:
                    for i,layer in enumerate(self.patch_emb):
                        patch_emb = layer(patch_emb,marker_idx,condition=LR)
            elif self.LR_emb_layers > 0:
                for i,layer in enumerate(self.LR_emb):
                    LR = layer(LR,marker_idx,condition=patch_emb)
        elif "LR" in self.conditions:
            LR = cond_data.LR
            if self.LR_emb_layers != 0:
                for i,layer in enumerate(self.LR_emb):
                    LR = layer(LR,marker_idx)
        elif "patch_emb" in self.conditions:
            patch_emb = cond_data.patch_emb
            if self.patch_emb_layers != 0:
                for i,layer in enumerate(self.patch_emb):
                    patch_emb = layer(patch_emb,marker_idx)
        
        if x is not None:
            if "x_emb" in self.conditions:
                for i,layer in enumerate(self.x_emb):
                    x = layer(x,marker_idx,condition=None)
        
        
        condition = None
        if LR is not None:
            if self.gene_emb_cross_attn:
                condition = LR.clone()
            LR = cond_data.LR if self.concat_ori_LR else LR
            if self.gene_emb_concat:
                if x is not None:
                    x = torch.cat([x,LR],dim=1) 
                else:
                    x = LR

        if patch_emb is not None:
            if self.patch_emb_cross_attn:
                if condition is not None:
                    condition = torch.cat([condition,patch_emb],dim=1)
                else:
                    condition = patch_emb.clone()
            patch_emb = cond_data.patch_emb if self.concat_ori_patch_emb else patch_emb
            if self.patch_emb_concat:
                if x is not None:
                    x = torch.cat([x,patch_emb],dim=1)
                else:
                    x = patch_emb

        return x,t,condition
    
    def forward(self, x,cond_data,noise_level=None):
        x,t,condition = self.get_condition(x,cond_data,noise_level)
        
        l = 0
        feats = []
        for layer in self.downs:
            #print(f"Layer Down input {l} layer, imput shape {x.shape}")
            if isinstance(layer, ResnetBlocWithAttn):
                if self.cross_attn:
                    x = layer(x, t,condition=condition)
                else:
                    x = layer(x, t)
            elif isinstance(layer, BlockWithFWA) or isinstance(layer, ResnetBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
            # print(f"Layer Down output {l} layer, output shape {x.shape}")
            # print(f"Downs {l} layer, feats shape {len(feats)}")
            l += 1
            
        l = 0
        for layer in self.mid:
            #print(f"Layer Mid input {l} layer, imput shape {x.shape}")
            if isinstance(layer, ResnetBlocWithAttn):
                if self.cross_attn:
                    x = layer(x, t,condition=condition)
                else:
                    x = layer(x, t)
            else:
                x = layer(x)
            # print(f"Layer Mid output {l} layer, output shape {x.shape}")
            # print(f"Mid {l} layer, feats shape {len(feats)}")
            l += 1
            
        l = 0
        for layer in self.ups:
            #print(f"Layer Up input {l} layer, imput shape {x.shape}")
            if isinstance(layer, ResnetBlocWithAttn):
                if self.cross_attn:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t,condition=condition)
                else:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
            # print(f"Layer Up output {l} layer, output shape {x.shape}")
            # print(f"Up {l} layer, feats shape {len(feats)}")
            l += 1
        
        if isinstance(self.final_conv, BlockWithFWA) or isinstance(self.final_conv, ResnetBlock):
            x = self.final_conv(x, t)
        else:
            x = self.final_conv(x)
        
        return x
    
    def sample(self, x,conda_data,noise_level=None):
        return self.forward(x,conda_data,noise_level)



############################################################################################
###########################Diffusion Model##################################################
############################################################################################
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def normlize_fn(x,scale_factor = 0.2,x_minus=0.):
    return x * scale_factor - x_minus

def denormlize_fn(x,scale_factor = 0.2,x_minus=0.):
    return (x + x_minus) / scale_factor

class Diffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        x_scale_factor = 1.,
        x_minus = 0.,
        residual=False,
        loss_type='l1',
        reduction='none',
        noise_level_type = "sqrt_alpha",
        schedule_opt=None
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.clip_min = -1. if residual or (x_scale_factor != 1.0 or x_minus != 0.) else -0.2
        self.clip_max = 1. if residual else 20.
        self.normlize_fn = partial(normlize_fn,scale_factor = x_scale_factor,x_minus=x_minus) if x_scale_factor != 1.0 or x_minus != 0. else lambda x: x
        self.denormlize_fn = partial(denormlize_fn,scale_factor = x_scale_factor,x_minus=x_minus) if x_scale_factor != 1.0 or x_minus != 0. else lambda x: x

        self.set_new_noise_schedule(schedule_opt)
        self.reduction = reduction
        self.set_loss(loss_type,reduction=reduction)
        self.noise_level_type = noise_level_type
    
    def set_loss(self, loss_type,reduction='none'):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction=reduction)
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t, clip_denoised: bool, cond_data=None):
        batch_size = x_t.shape[0]
        if self.noise_level_type == "sqrt_alpha":
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod[t]]).repeat(batch_size, ).to(x_t.device)
        elif self.noise_level_type == "time":
            noise_level = torch.FloatTensor([t]).repeat(batch_size, ).to(x_t.device)
        if cond_data is not None:
            x_recon = self.predict_start_from_noise(
                x_t, t=t, noise=self.denoise_fn(x_t,cond_data,noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x_t, t=t, noise=self.denoise_fn(x_t, noise_level))

        if clip_denoised:
            x_recon.clamp_(self.clip_min, self.clip_max) # Clip denoised image

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t, clip_denoised=True, cond_data=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x_t=x_t, t=t, clip_denoised=clip_denoised, cond_data=cond_data)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, shape,cond_data, continous=False,clip_denoised=True):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        img = torch.randn(shape, device=device)
        ret_img = img
        for t in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, t, cond_data=cond_data,clip_denoised= clip_denoised)
            if continous:
                if t % sample_inter == 0:
                    ret_img = torch.cat([ret_img, self.denormlize_fn(img)], dim=0)
        if continous:
            return ret_img
        else:
            img = self.denormlize_fn(img)
            return img

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )
    
    def p_losses(self, x_0,cond_data,mask):
        b, c, h, w = x_0.shape
        mask = repeat(mask, 'b h w -> b c h w',c=c)
        
        
        #t = torch.randint(0, self.num_timesteps,(1,)).long()
        #t = t.repeat(b)
        # 使用每个样本不同的 t 值
        t = torch.randint(0, self.num_timesteps, (b,)).long()
        
        sqrt_alpha = torch.tensor(self.sqrt_alphas_cumprod_prev[t+1].reshape(-1,1,1,1)).to(x_0.device).to(torch.float32)
        noise = torch.randn_like(x_0).to(x_0.device)
        
        # if use uniform distribution of noise level
        #t = np.random.randint(0, self.num_timesteps)
        #continuous_sqrt_alpha_cumprod = torch.FloatTensor(
        #    np.random.uniform(
        #        self.sqrt_alphas_cumprod_prev[t],
        #        self.sqrt_alphas_cumprod_prev[t+1],
        #        size=b
        #    )
        #).to(x_0.device).to(torch.float32)

        # Perturbed image obtained by forward diffusion process at random time step t
        x_t = sqrt_alpha * x_0 + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The model predict actual noise added at time step t
        if self.noise_level_type == "sqrt_alpha":
            noise_level = sqrt_alpha.reshape(-1).float().to(x_0.device)
        elif self.noise_level_type == "time":
            noise_level = t.reshape(-1).float().to(x_0.device)
        pred_noise = self.denoise_fn(x_t, cond_data, noise_level=noise_level)
        ##################################################
        if self.reduction == 'none':
            noise = noise * mask
            pred_noise = pred_noise * mask
            loss = self.loss_func(noise, pred_noise) 
            loss = loss.sum(dim=(2,3))/mask.sum(dim=(2,3))
            try:
                loss = loss.mean()
            except:
                pass
        elif self.reduction == 'sum':
            loss = self.loss_func(noise, pred_noise)
            loss = loss/b
        elif self.reduction == 'mean':
            loss = self.loss_func(noise, pred_noise)
        ##################################################
        return loss
    
    def forward(self, x_0,cond_data,mask, *args, **kwargs):
        x_0 = self.normlize_fn(x_0)
        if "LR" in self.denoise_fn.conditions:
            cond_data = cond_data.clone()
            cond_data.LR = self.normlize_fn(cond_data.LR)
        return self.p_losses(x_0,cond_data,mask, *args, **kwargs)