import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

class ALiBiPositionalEncoding(nn.Module):
    """ALiBi for 2D images with separate height/width biases"""
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Use standard ALiBi slope formula: 2^(-8i/n) for head i
        slopes = torch.tensor([2**(-8 * (i + 1) / num_heads) for i in range(num_heads)])
        self.register_buffer('slopes', slopes)
    
    def forward(self, attn_scores, h, w):
        """
        Apply ALiBi bias to attention scores
        Args:
            attn_scores: attention scores [B, num_heads, H*W, H*W]
            h: height of feature map
            w: width of feature map
        """
        device = attn_scores.device
        B, num_heads, seq_len, _ = attn_scores.shape
        
        # Create relative position indices for height and width
        pos_h = torch.arange(h, device=device)
        pos_w = torch.arange(w, device=device)
        rel_pos_h = pos_h[:, None] - pos_h[None, :]  # [h, h]
        rel_pos_w = pos_w[:, None] - pos_w[None, :]  # [w, w]
        
        # Combine into 2D relative positions
        rel_pos_h = rel_pos_h.unsqueeze(-1).expand(-1, -1, w).reshape(h * w, h * w)  # [H*W, H*W]
        rel_pos_w = rel_pos_w.unsqueeze(0).expand(h, -1, -1).reshape(h * w, h * w)    # [H*W, H*W]
        
        # Compute biases (negative slopes * distance)
        bias = -self.slopes.view(1, num_heads, 1, 1) * (
            rel_pos_h.abs().unsqueeze(0).unsqueeze(0) +  # [1, 1, H*W, H*W]
            rel_pos_w.abs().unsqueeze(0).unsqueeze(0)
        )
        
        return attn_scores + bias

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with ALiBi positional encoding"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.alibi = ALiBiPositionalEncoding(num_heads)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 3)  # [3, B, num_heads, H, W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Reshape for attention computation
        q = q.reshape(B, self.num_heads, H * W, self.head_dim)
        k = k.reshape(B, self.num_heads, H * W, self.head_dim)
        v = v.reshape(B, self.num_heads, H * W, self.head_dim)
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply ALiBi positional bias
        attn = self.alibi(attn, H, W)
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        
        # Reshape back to feature map format
        out = out.reshape(B, self.num_heads, H, W, self.head_dim)
        out = out.permute(0, 1, 4, 2, 3)  # [B, num_heads, head_dim, H, W]
        out = out.reshape(B, C, H, W)
        
        # Final projection
        out = self.proj(out)
        
        return out

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

        self.layer1 = nn.Sequential(
            nn.Conv2d(1+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Down-sampling layers.
        curr_dim = conv_dim
        self.enc_layer1 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2

        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2

        self.enc_layer3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2
        
        # Bottleneck layers with self-attention and ALiBi
        layers = []

        # Add self-attention layer in the middle of residual blocks
        mid_point = repeat_num // 2
        for i in range(repeat_num):
            if i == mid_point:
                # Ensure num_heads compatibility
                num_heads = min(8, curr_dim // 8)  # Ensure head_dim >= 8
                num_heads = max(1, num_heads)  # At least 1 head
                layers.append(MultiHeadSelfAttention(curr_dim, num_heads=num_heads))
                layers.append(nn.InstanceNorm2d(curr_dim))
                layers.append(nn.Dropout2d(0.1))  # Add dropout for stability
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bn2 = nn.Sequential(*layers)


        # Up-sampling layers.
        self.gate1 = AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_bn1 = ResidualBlock1(dim_in=2*curr_dim,dim_out=curr_dim)
        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(curr_dim//2, curr_dim//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//4, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )


        self.gate2 = AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_bn2 = ResidualBlock1(dim_in=2*curr_dim,dim_out=curr_dim)
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.gate3= AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_bn3 = ResidualBlock1(dim_in=2*curr_dim,dim_out=curr_dim)
        self.dec_layer3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2
        self.bn3 = ResidualBlock1(dim_in=3*curr_dim,dim_out=curr_dim)

        self.conv = nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x,bs=1):
        y = self.layer1(x)

        y1 = self.enc_layer1(y)
        sc = []
        sc.append(y1)

        y2 = self.enc_layer2(y1)
        sc.append(y2)

        y3 = self.enc_layer3(y2)
        sc.append(y3)

        embedding = self.bn2(y3)

        i = 0
        sc1 = self.gate1(gate=embedding,skip_connection=sc[-1-i])
        out1 = torch.concat([embedding,sc1],dim=1)
        out1 = self.dec_bn1(out1)
        out1 = self.dec_layer1(out1)

        i += 1
        sc2 = self.gate2(gate=out1,skip_connection=sc[-1-i])
        out2 = torch.concat([out1,sc2],dim=1)
        out2 = self.dec_bn2(out2)
        out2 = self.dec_layer2(out2)

        i += 1
        sc3 = self.gate3(gate=out2,skip_connection=sc[-1-i])
        out3 = torch.concat([out2,sc3],dim=1)
        out3 = self.dec_bn3(out3)
        out3 = self.dec_layer3(out3)

        out1 = self.up_conv1(out1)
        out2 = self.up_conv2(out2)
        out = torch.concat([out1,out2,out3],dim=1)
        out = self.bn3(out)

        output = self.conv(out)

        if self.training:
            # spatial binary mask
            mask = torch.ones(output.size(0), 1, output.size(-2), output.size(-1)).to(output.device) * 0.95
            mask = torch.bernoulli(mask).float()
            output = mask * output + (1. - mask) * x
        fake1 = torch.tanh(output+x)
    
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