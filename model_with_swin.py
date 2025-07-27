import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def window_partition(x, window_size):
    """
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

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

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

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
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

class SwinTransformerBlock(nn.Module):
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
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = x.view(B, H, W, 4, C//4).permute(0, 1, 3, 2, 4).contiguous().view(B, H*2, W*2, C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinEncoder(nn.Module):
    """Fixed Swin Transformer Encoder"""
    def __init__(self, in_chans, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, out_indices=(0, 1, 2, 3), img_size=224):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        
        # Patch embedding using conv instead of patch embed
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        
        patches_resolution = [img_size // 4, img_size // 4]
        self.patches_resolution = patches_resolution
        
        if self.patch_norm:
            self.norm = norm_layer(embed_dim)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
            
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        
        if self.patch_norm:
            x = self.norm(x)
            
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                # Get current resolution
                curr_H = self.patches_resolution[0] // (2 ** i)
                curr_W = self.patches_resolution[1] // (2 ** i)
                out = x.view(B, curr_H, curr_W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                
        return outs

class SwinDecoder(nn.Module):
    """Fixed Swin Transformer Decoder that mirrors encoder"""
    def __init__(self, embed_dim=768, depths=[2, 6, 2, 2], num_heads=[24, 12, 6, 3],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build decoder layers (reversed structure)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Start from small resolution and expand
            curr_dim = int(embed_dim // (2 ** (self.num_layers - 1 - i_layer)))
            curr_res = 8 * (2 ** i_layer)  # Start from 8x8, then 16x16, 32x32, 64x64
            
            layer = BasicLayer(
                dim=curr_dim,
                input_resolution=(curr_res, curr_res),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None)  # No downsampling in decoder
            self.layers.append(layer)
            
            # Add upsampling between layers (except last)
            if i_layer < self.num_layers - 1:
                self.layers.append(PatchExpand(
                    input_resolution=(curr_res, curr_res),
                    dim=curr_dim,
                    dim_scale=2,
                    norm_layer=norm_layer))
            
    def forward(self, x):
        """
        Args:
            x: input features (B, H*W, C)
        """
        for layer in self.layers:
            x = layer(x)
            
        return x

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class ResidualBlock1(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock1, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.sc = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        s_c = self.main(x)
        y = self.sc(s_c)
        return s_c + y

class Generator(nn.Module):
    """Fixed Generator network with properly aligned Swin Transformer."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, device='cuda:6', num_memory=4):
        super(Generator, self).__init__()
        self.num_memory = num_memory

        # Initial convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(1+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Swin Transformer configuration for tiny model
        swin_config = {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24], 
            'window_size': 7,
            'img_size': 128
        }

        # Swin Encoder 
        self.swin_encoder = SwinEncoder(
            in_chans=conv_dim,
            embed_dim=swin_config['embed_dim'],
            depths=swin_config['depths'],
            num_heads=swin_config['num_heads'],
            window_size=swin_config['window_size'],
            img_size=swin_config['img_size'],
            patch_norm=True  # Enable Swin's native normalization
        )

        # Calculate dimensions after Swin encoder
        curr_dim = swin_config['embed_dim'] * (2 ** (len(swin_config['depths']) - 1))  # 768 for tiny
        
        # Bottleneck layers
        layers = []
        layers.append(nn.Conv2d(curr_dim, curr_dim, 1, 1, 0))
        layers.append(nn.InstanceNorm2d(curr_dim))
        layers.append(nn.ReLU(inplace=True))

        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bn2 = nn.Sequential(*layers)

        # Swin Transformer Decoder (mirrors encoder structure)
        self.swin_decoder = SwinDecoder(
            embed_dim=curr_dim,
            depths=[2, 6, 2, 2],  # Reversed encoder depths
            num_heads=[24, 12, 6, 3],  # Reversed encoder heads
            window_size=7
        )

        # Fixed skip connection projection layers with dynamic dimensions based on embed_dim
        embed_dim = swin_config['embed_dim']
        self.skip_projections = nn.ModuleList([
            nn.Conv2d(embed_dim * 8, curr_dim // 4, 1),  # Stage 3: 768->192
            nn.Conv2d(embed_dim * 4, curr_dim // 8, 1),  # Stage 2: 384->96  
            nn.Conv2d(embed_dim * 2, curr_dim // 16, 1), # Stage 1: 192->48
            nn.Conv2d(embed_dim, curr_dim // 32, 1),     # Stage 0: 96->24
        ])

        # Fixed attention gates with proper dimension handling
        self.gate1 = AttentionGate(F_g=curr_dim//4, F_l=curr_dim//4, n_coefficients=curr_dim//8)
        self.gate2 = AttentionGate(F_g=curr_dim//8, F_l=curr_dim//8, n_coefficients=curr_dim//16)
        self.gate3 = AttentionGate(F_g=curr_dim//16, F_l=curr_dim//16, n_coefficients=curr_dim//32)

        # Final output layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(curr_dim//4 + curr_dim//8 + curr_dim//16, conv_dim, 3, 1, 1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)
        )

    def forward(self, x, bs=1):
        original_input = x
        y = self.layer1(x)

        # Swin Transformer encoder
        swin_features = self.swin_encoder(y)
        
        # FIXED: Properly handle bottleneck processing
        deepest_features = swin_features[-1]  # Use the deepest features (B, C, H, W)
        B, C, H, W = deepest_features.shape
        
        # Convert to spatial format for bottleneck processing
        embedding = self.bn2(deepest_features)
        
        # FIXED: Convert to sequence format for decoder
        embedding_seq = embedding.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Swin Decoder
        decoder_out = self.swin_decoder(embedding_seq)
        B_dec, L_dec, C_dec = decoder_out.shape
        H_dec = W_dec = int(L_dec ** 0.5)
        decoder_spatial = decoder_out.view(B_dec, H_dec, W_dec, C_dec).permute(0, 3, 1, 2)

        # FIXED: Multi-scale feature fusion with proper progressive upsampling
        # Project skip connections to match decoder dimensions
        skip1 = self.skip_projections[0](swin_features[-1])  # 768->192
        skip2 = self.skip_projections[1](swin_features[-2])  # 384->96
        skip3 = self.skip_projections[2](swin_features[-3])  # 192->48

        # Progressive upsampling and fusion (like U-Net)
        # Start from decoder output and progressively upsample while fusing
        current_features = decoder_spatial

        # Level 1: Upsample decoder to match skip1 resolution
        if current_features.shape[-2:] != skip1.shape[-2:]:
            current_features = F.interpolate(current_features, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
        
        # Ensure channel dimensions match for attention gate
        current_features_proj = current_features[:, :skip1.shape[1]]  # Truncate channels if needed
        gated_skip1 = self.gate1(gate=current_features_proj, skip_connection=skip1)
        
        # Level 2: Upsample and fuse with skip2
        fused_1 = current_features_proj + gated_skip1
        fused_1_up = F.interpolate(fused_1, size=skip2.shape[-2:], mode='bilinear', align_corners=False)
        
        # Project to match skip2 channels
        fused_1_proj = fused_1_up[:, :skip2.shape[1]]
        gated_skip2 = self.gate2(gate=fused_1_proj, skip_connection=skip2)
        
        # Level 3: Upsample and fuse with skip3
        fused_2 = fused_1_proj + gated_skip2
        fused_2_up = F.interpolate(fused_2, size=skip3.shape[-2:], mode='bilinear', align_corners=False)
        
        # Project to match skip3 channels
        fused_2_proj = fused_2_up[:, :skip3.shape[1]]
        gated_skip3 = self.gate3(gate=fused_2_proj, skip_connection=skip3)
        
        # Final fusion
        fused_3 = fused_2_proj + gated_skip3

        # FIXED: Resize all features to final output resolution and concatenate
        target_size = original_input.shape[-2:]
        
        # Resize each level to target size
        out1 = F.interpolate(gated_skip1, size=target_size, mode='bilinear', align_corners=False)
        out2 = F.interpolate(gated_skip2, size=target_size, mode='bilinear', align_corners=False)
        out3 = F.interpolate(gated_skip3, size=target_size, mode='bilinear', align_corners=False)

        # Concatenate multi-scale features
        multi_scale_features = torch.cat([out1, out2, out3], dim=1)
        
        # Final output
        output = self.final_conv(multi_scale_features)

        # Training mask (keep original functionality)
        if self.training:
            mask = torch.ones(output.size(0), 1, output.size(-2), output.size(-1)).to(output.device) * 0.95
            mask = torch.bernoulli(mask).float()
            output = mask * output + (1. - mask) * original_input
            
        fake1 = torch.tanh(output + original_input)
        return fake1

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return h, out_src

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def binarize(integer, num_bits=8):   
    """Turn integer tensor to binary representation.        
    Args:           
    integer : torch.Tensor, tensor with integers           
    num_bits : Number of bits to specify the precision. Default: 8.       
    Returns:           
    Tensor: Binary tensor. Adds last dimension to original tensor for           
    bits.    
    """   
    dtype = integer.type()   
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)   
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))   
    out = integer.unsqueeze(-1) / 2 ** exponent_bits   
    return (out - (out % 1)) % 2

class AttentionGate(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionGate, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi

        return out