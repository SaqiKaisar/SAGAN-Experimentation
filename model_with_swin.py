import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def _get_num_heads(dim: int, base: int = 32) -> int:
	"""Heuristic to choose number of heads proportional to channel dim."""
	return max(1, dim // base)

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

def window_reverse_swin(windows, window_size, H, W):
	"""
	Reverse windows partition used in Swin blocks.
	Args:
		windows: (num_windows*B, window_size, window_size, C)
		window_size (int)
		H (int) Height
		W (int) Width
	Returns:
		x: (B, H, W, C)
	"""
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x

class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = nn.GELU()
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
	"""Window based multi-head self attention (W-MSA) module with relative position bias."""
	def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
		super().__init__()
		self.dim = dim
		self.window_size = window_size
		self.num_heads = num_heads
		self.scale = (dim // num_heads) ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		# relative position bias table
		self.relative_position_bias_table = nn.Parameter(
			torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
		)
		coords_h = torch.arange(window_size)
		coords_w = torch.arange(window_size)
		coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
		coords_flatten = torch.flatten(coords, 1)
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()
		relative_coords[:, :, 0] += window_size - 1
		relative_coords[:, :, 1] += window_size - 1
		relative_coords[:, :, 0] *= 2 * window_size - 1
		relative_position_index = relative_coords.sum(-1)
		self.register_buffer("relative_position_index", relative_position_index)
		nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

	def forward(self, x, mask=None):
		B_, N, C = x.shape
		qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
		qkv = qkv.permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]
		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))
		relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
			self.window_size * self.window_size, self.window_size * self.window_size, -1)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
		attn = attn + relative_position_bias.unsqueeze(0)
		if mask is not None:
			# mask: (num_windows, N, N)
			nW = mask.shape[0]
			attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, N, N)
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class SwinTransformerBlock(nn.Module):
	def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		self.norm1 = nn.LayerNorm(dim)
		self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
		self.norm2 = nn.LayerNorm(dim)
		self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

	def create_attn_mask(self, H, W, device):
		if self.shift_size == 0:
			return None
		img_mask = torch.zeros((1, H, W, 1), device=device)
		cnt = 0
		for h in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
			for w in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
				img_mask[:, h, w, :] = cnt
				cnt += 1
		mask_windows = window_partition(img_mask, self.window_size)
		mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
		attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
		attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
		return attn_mask

	def forward(self, x):
		B, C, H, W = x.shape
		# (B, H, W, C)
		x_perm = x.permute(0, 2, 3, 1).contiguous()
		shortcut = x_perm
		if self.shift_size > 0:
			shifted = torch.roll(x_perm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
		else:
			shifted = x_perm
		# partition windows
		x_windows = window_partition(shifted, self.window_size)  # (num_windows*B, w, w, C)
		x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
		attn_mask = self.create_attn_mask(H, W, x.device)
		# W-MSA/SW-MSA
		x_windows = self.norm1(x_windows)
		attn_windows = self.attn(x_windows, mask=attn_mask)
		# merge windows
		attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
		shifted_back = window_reverse_swin(attn_windows, self.window_size, H, W)
		if self.shift_size > 0:
			x_perm = torch.roll(shifted_back, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:
			x_perm = shifted_back
		# FFN
		x_perm = x_perm + shortcut
		x_norm = self.norm2(x_perm.view(B, H * W, C))
		x_mlp = self.mlp(x_norm)
		x_perm = x_perm + x_mlp.view(B, H, W, C)
		return x_perm.permute(0, 3, 1, 2).contiguous()

class SwinStage(nn.Module):
	"""A stage applying multiple Swin blocks to a CNN feature map."""
	def __init__(self, dim, depth, input_resolution, window_size):
		super().__init__()
		blocks = []
		for i in range(depth):
			shift = 0 if (i % 2 == 0) else window_size // 2
			blocks.append(SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=_get_num_heads(dim), window_size=window_size, shift_size=shift))
		self.blocks = nn.ModuleList(blocks)

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x

class PerPixelLinear(nn.Module):
	"""Apply LayerNorm+Linear across channel dimension per spatial location (no conv)."""
	def __init__(self, in_ch, out_ch, act=True):
		super().__init__()
		self.norm = nn.LayerNorm(in_ch)
		self.proj = nn.Linear(in_ch, out_ch)
		self.act = nn.GELU() if act else nn.Identity()

	def forward(self, x):
		B, C, H, W = x.shape
		x = x.permute(0, 2, 3, 1).contiguous()
		x = self.proj(self.norm(x))
		x = self.act(x)
		return x.permute(0, 3, 1, 2).contiguous()

class PositionProjector(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.proj = PerPixelLinear(in_ch, out_ch, act=True)

	def forward(self, x):
		return self.proj(x)

class PatchMerging2d(nn.Module):
	"""Downsample by 2 using 2x2 patch merging and linear projection to 2C."""
	def __init__(self, dim):
		super().__init__()
		self.norm = nn.LayerNorm(4 * dim)
		self.reduction = nn.Linear(4 * dim, 2 * dim)

	def forward(self, x):
		B, C, H, W = x.shape
		# pad if needed to be even
		pad_h = H % 2
		pad_w = W % 2
		if pad_h or pad_w:
			x = F.pad(x, (0, pad_w, 0, pad_h))
			H = x.shape[2]
			W = x.shape[3]
		# 2x2 merge
		x0 = x[:, :, 0::2, 0::2]
		x1 = x[:, :, 1::2, 0::2]
		x2 = x[:, :, 0::2, 1::2]
		x3 = x[:, :, 1::2, 1::2]
		x = torch.cat([x0, x1, x2, x3], dim=1)  # B, 4C, H/2, W/2
		B, C4, H2, W2 = x.shape
		x = x.permute(0, 2, 3, 1).contiguous().view(B, H2 * W2, C4)
		x = self.reduction(self.norm(x))
		x = x.view(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
		return x

class PatchExpand2d(nn.Module):
	"""Upsample by 2 using linear projection and pixel shuffle-like rearrangement."""
	def __init__(self, dim):
		super().__init__()
		# project C -> 4 * (C // 2)
		self.out_dim = max(1, dim // 2)
		self.norm = nn.LayerNorm(dim)
		self.proj = nn.Linear(dim, 4 * self.out_dim)

	def forward(self, x):
		B, C, H, W = x.shape
		x = x.permute(0, 2, 3, 1).contiguous()
		x = self.proj(self.norm(x))  # B,H,W,4C_out
		x = x.view(B, H, W, 2, 2, self.out_dim).permute(0, 5, 1, 3, 2, 4).contiguous()
		x = x.view(B, self.out_dim, H * 2, W * 2)
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
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6,device = 'cuda:6',num_memory = 4):
        super(Generator, self).__init__()
        self.num_memory = num_memory

        # Input projection (replace initial conv with per-pixel linear)
        self.layer1 = PerPixelLinear(1+c_dim, conv_dim, act=True)

        # Down-sampling layers.
        curr_dim = conv_dim
        # Encoder stage 1: Swin + patch merge (replaces strided conv)
        self.swin_e1 = SwinStage(dim=curr_dim, depth=2, input_resolution=None, window_size=8)
        self.enc_merge1 = PatchMerging2d(curr_dim)
        curr_dim = curr_dim * 2
        self.p_bn1 = self.position_bn(curr_dim=curr_dim,num_memory=self.num_memory)


        # Encoder stage 2: Swin + patch merge
        self.swin_e2 = SwinStage(dim=curr_dim, depth=2, input_resolution=None, window_size=4)
        self.enc_merge2 = PatchMerging2d(curr_dim)
        curr_dim = curr_dim * 2
        self.p_bn2 = self.position_bn(curr_dim=curr_dim,num_memory=self.num_memory)

        # Encoder stage 3: Swin + patch merge
        self.swin_e3 = SwinStage(dim=curr_dim, depth=2, input_resolution=None, window_size=4)
        self.enc_merge3 = PatchMerging2d(curr_dim)
        curr_dim = curr_dim * 2
        
        # Bottleneck layers.

        # Bottleneck: condition projection + Swin only
        self.cond_proj = PositionProjector(curr_dim + int(np.log2(num_memory)+1), curr_dim)
        self.swin_bn = SwinStage(dim=curr_dim, depth=repeat_num, input_resolution=None, window_size=4)


        # Up-sampling layers.
        self.gate1 = AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_fuse1 = PerPixelLinear(2*curr_dim, curr_dim, act=True)
        self.swin_d1 = SwinStage(dim=curr_dim, depth=2, input_resolution=None, window_size=4)
        self.dec_up1 = PatchExpand2d(curr_dim)
        curr_dim = curr_dim // 2
        self.up_conv1 = nn.Sequential(
            PatchExpand2d(curr_dim),
            PatchExpand2d(max(1, curr_dim//2))
        )


        self.gate2 = AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_fuse2 = PerPixelLinear(2*curr_dim, curr_dim, act=True)
        self.swin_d2 = SwinStage(dim=curr_dim, depth=2, input_resolution=None, window_size=4)
        self.dec_up2 = PatchExpand2d(curr_dim)
        curr_dim = curr_dim // 2

        self.up_conv2 = nn.Sequential(
            PatchExpand2d(curr_dim)
        )

        self.gate3= AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_fuse3 = PerPixelLinear(2*curr_dim, curr_dim, act=True)
        self.swin_d3 = SwinStage(dim=curr_dim, depth=2, input_resolution=None, window_size=4)
        self.dec_up3 = PatchExpand2d(curr_dim)
        curr_dim = curr_dim // 2
        self.bn3 = PerPixelLinear(3*curr_dim, curr_dim, act=True)

        # Final projection to 1 channel
        self.final_proj = PerPixelLinear(curr_dim, 1, act=False)
        self.binary_encoding = binarize(torch.arange(0,self.num_memory,dtype=torch.int), int(np.log2(num_memory)+1))

    def forward(self, x,bs=1):
        y = self.layer1(x)

        y1 = self.swin_e1(y)
        y1 = self.enc_merge1(y1)
        _,C,W,H = y1.shape
        y1 = y1.view(bs,self.num_memory,C,W,H)
        y1 = self.add_condition(y1,bs,num_windows=self.num_memory)
        y1 = y1.view(bs * self.num_memory,-1,W,H)
        y1 = self.p_bn1(y1)
        sc = []
        sc.append(y1)

        y2 = self.swin_e2(y1)
        y2 = self.enc_merge2(y2)
        _,C,W,H = y2.shape
        y2 = y2.view(bs,self.num_memory,C,W,H)
        y2 = self.add_condition(y2,bs,num_windows=self.num_memory)
        y2 = y2.view(bs * self.num_memory,-1,W,H)
        y2 = self.p_bn2(y2)
        sc.append(y2)

        y3 = self.swin_e3(y2)
        y3 = self.enc_merge3(y3)
        sc.append(y3)

        y = y3
        B,C,W,H = y.shape
        y = y.view(bs,self.num_memory,C,W,H)
        y_p = self.add_condition(y,bs,num_windows=self.num_memory)
        y_p = y_p.view(bs * self.num_memory,-1,W,H)
        embedding = self.cond_proj(y_p)
        embedding = self.swin_bn(embedding)

        i = 0
        sc1 = self.gate1(gate=embedding,skip_connection=embedding)
        out1 = torch.concat([embedding,sc1],dim=1)
        out1 = self.dec_fuse1(out1)
        out1 = self.swin_d1(out1)
        out1 = self.dec_up1(out1)

        i += 1
        sc2 = self.gate2(gate=out1,skip_connection=sc[-1-i])
        out2 = torch.concat([out1,sc2],dim=1)
        out2 = self.dec_fuse2(out2)
        out2 = self.swin_d2(out2)
        out2 = self.dec_up2(out2)

        i += 1
        sc3 = self.gate3(gate=out2,skip_connection=sc[-1-i])
        out3 = torch.concat([out2,sc3],dim=1)
        out3 = self.dec_fuse3(out3)
        out3 = self.swin_d3(out3)
        out3 = self.dec_up3(out3)

        out1 = self.up_conv1(out1)
        out2 = self.up_conv2(out2)
        out = torch.concat([out1,out2,out3],dim=1)
        out = self.bn3(out)

        output = self.final_proj(out)

        if self.training:
            # spatial binary mask
            mask = torch.ones(output.size(0), 1, output.size(-2), output.size(-1)).to(output.device) * 0.95
            mask = torch.bernoulli(mask).float()
            output = mask * output + (1. - mask) * x
        fake1 = torch.tanh(output+x)
    
        return fake1
    
    def add_condition(self, x, bs, num_windows):
        condition = self.binary_encoding.to(x.device).view(1, self.binary_encoding.shape[0], self.binary_encoding.shape[1], 1, 1)
        condition = condition.expand(bs, -1, -1, x.size(-2), x.size(-1)).contiguous().float()
        x = torch.cat((x, condition), dim=2)
        return x
    
    def position_bn(self,curr_dim,num_memory = 4):
        # Replace 1x1 conv with per-pixel linear projection
        return PositionProjector(curr_dim + int(np.log2(num_memory)+1), curr_dim)

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
