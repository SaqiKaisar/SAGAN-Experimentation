# swin_gan_models.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --------------------------
# Helpers
# --------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def binarize(integer, num_bits=8):
    """Turn integer tensor to binary representation.
    integer: tensor-like or scalar (will be converted to torch.Tensor)
    returns: Float tensor with shape integer.shape + (num_bits,)
    """
    integer = torch.as_tensor(integer).long()
    if integer.dim() == 0:
        integer = integer.unsqueeze(0)
    # bit positions from high->low: num_bits-1 ... 0
    shifts = torch.arange(num_bits - 1, -1, -1, device=integer.device, dtype=integer.dtype)
    out = ((integer.unsqueeze(-1) >> shifts) & 1).float()
    return out


def window_reverse(windows, window_size, H, W):
    """
    Args:
      windows: (num_windows*B, window_size, window_size, C) or (num_windows*B, C, window_size, window_size)
    Returns:
      x: (B, C, H, W)
    """
    # Keep simple 4D -> convert to B, H, W, C then permute back if needed
    # The caller in the original code used a specific format; keep a safe implementation.
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # convert to (B, C, H, W)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


# --------------------------
# Residual & Attention blocks (kept from your original style)
# --------------------------
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.main(x)


class ResidualBlock1(nn.Module):
    """Residual Block with instance normalization (two-stage)."""
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


class AttentionGate(nn.Module):
    """Attention block with learnable parameters"""
    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in gating signal
        :param F_l: channels in skip connection to attend
        :param n_coefficients: intermediate channels
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
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


# --------------------------
# Swin-based Generator (drop-in)
# --------------------------
class SwinGenerator(nn.Module):
    """
    Drop-in Generator replacing CNN U-Net & bottleneck with a Swin encoder + decoder.
    __init__ signature kept similar to your original: conv_dim, c_dim, repeat_num, device, num_memory
    """

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, device='cuda:6', num_memory=4,
                 swin_model_name='swin_tiny_patch4_window7_224', pretrained=False):
        super(SwinGenerator, self).__init__()

        # store some params
        self.num_memory = num_memory
        self.c_dim = c_dim
        self.repeat_num = repeat_num
        self.device = device

        # Instantiate swin backbone with features_only -> returns list of intermediate feature maps
        # in_chans = 1 + c_dim to accept same input as your original generator
        self.swin = timm.create_model(swin_model_name, pretrained=pretrained, features_only=True,
                                      in_chans=1 + c_dim)

        assert hasattr(self.swin, 'feature_info'), "timm model doesn't expose feature_info; ensure timm version is up-to-date."

        # encoder channels (shallow -> deep), typical for swin_tiny: [96, 192, 384, 768]
        self.enc_channels = self.swin.feature_info.channels
        # Ensure we have 4 stages
        if len(self.enc_channels) < 4:
            # replicate last if fewer; usually not needed for standard swin models
            while len(self.enc_channels) < 4:
                self.enc_channels.insert(0, self.enc_channels[0])

        # Bottleneck channel sizing:
        c1, c2, c3, c4 = self.enc_channels[0], self.enc_channels[1], self.enc_channels[2], self.enc_channels[3]
        bottleneck_in_ch = c4
        # choose bottleneck_out_ch to match c3 so decoder upsample can map to c3 channels
        bottleneck_out_ch = c3

        # Bottleneck block: a 1x1 conv + instance norm + residual blocks
        bottleneck_layers = [
            nn.Conv2d(bottleneck_in_ch, bottleneck_out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(bottleneck_out_ch),
            nn.ReLU(inplace=True)
        ]
        for _ in range(repeat_num):
            bottleneck_layers.append(ResidualBlock(bottleneck_out_ch, bottleneck_out_ch))
        self.bn2 = nn.Sequential(*bottleneck_layers)

        # Decoder up-block helper
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Decoder: bottleneck_out_ch -> c3 -> c2 -> c1 -> up to input resolution
        self.up4 = up_block(bottleneck_out_ch, c3)   # H/32 -> H/16
        self.attn3 = AttentionGate(F_g=c3, F_l=c3, n_coefficients=max(1, c3 // 2))
        self.dec3 = ResidualBlock1(dim_in=2 * c3, dim_out=c3)

        self.up3 = up_block(c3, c2)                  # H/16 -> H/8
        self.attn2 = AttentionGate(F_g=c2, F_l=c2, n_coefficients=max(1, c2 // 2))
        self.dec2 = ResidualBlock1(dim_in=2 * c2, dim_out=c2)

        self.up2 = up_block(c2, c1)                  # H/8 -> H/4
        self.attn1 = AttentionGate(F_g=c1, F_l=c1, n_coefficients=max(1, c1 // 2))
        self.dec1 = ResidualBlock1(dim_in=2 * c1, dim_out=c1)

        # final upsampling stages to get back to original H x W (two x2 upsamples from H/4 -> H)
        final_up_channels = max(conv_dim, c1 // 2)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2),  # H/4 -> H/2
            nn.InstanceNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1 // 2, final_up_channels, kernel_size=2, stride=2),  # H/2 -> H
            nn.InstanceNorm2d(final_up_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_up_channels, final_up_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(final_up_channels),
            nn.ReLU(inplace=True),
        )

        # final conv to single channel
        self.conv = nn.Conv2d(final_up_channels, 1, kernel_size=7, stride=1, padding=3, bias=False)

        # compatibility: binary encoding used by older code path
        nb = int(math.log2(max(2, num_memory)) + 1)
        self.binary_encoding = binarize(torch.arange(0, num_memory, dtype=torch.int64), nb)

    def add_condition(self, x, bs, num_windows):
        condition = self.binary_encoding.to(x.device).view(1, self.binary_encoding.shape[0],
                                                            self.binary_encoding.shape[1], 1, 1)
        condition = condition.expand(bs, -1, -1, x.size(-2), x.size(-1)).contiguous().float()
        x = torch.cat((x, condition), dim=2)
        return x

    def position_bn(self, curr_dim, num_memory=4):
        layers = []
        layers.append(nn.Conv2d(curr_dim + int(np.log2(num_memory) + 1), curr_dim, 1, 1, 0))
        layers.append(nn.InstanceNorm2d(curr_dim))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, bs=1):
        """
        x: Tensor shape (B, 1 + c_dim, H, W)
        returns: Tensor (B, 1, H, W)
        """
        # keep first (image) channel for residual additions if needed
        img_for_residual = x[:, :1, :, :]

        # extract features from Swin backbone
        features = self.swin(x)  # list: [f1 (shallow), f2, f3, f4 (deep)]
        if not isinstance(features, (list, tuple)):
            features = [features]

        # pad if fewer than 4 features (robustness)
        while len(features) < 4:
            features.insert(0, features[0])

        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]

        # bottleneck
        bottleneck = self.bn2(f4)  # shape: B x c3 x H/16 x W/16 (if f4 H/32)

        # decode stage 1: up to f3 scale
        d4 = self.up4(bottleneck)  # -> spatial like f3
        g3 = self.attn3(gate=d4, skip_connection=f3)
        out3 = torch.cat([d4, g3], dim=1)
        out3 = self.dec3(out3)

        # decode stage 2: up to f2 scale
        d3 = self.up3(out3)
        g2 = self.attn2(gate=d3, skip_connection=f2)
        out2 = torch.cat([d3, g2], dim=1)
        out2 = self.dec2(out2)

        # decode stage 3: up to f1 scale
        d2 = self.up2(out2)
        g1 = self.attn1(gate=d2, skip_connection=f1)
        out1 = torch.cat([d2, g1], dim=1)
        out1 = self.dec1(out1)

        # final upsampling to input resolution
        up_final = self.final_up(out1)  # should match input H,W or be a close factor

        output = self.conv(up_final)  # B x 1 x h_out x w_out

        # Ensure final spatial dims equal input dims (safeguard)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # training-time masking behaviour (kept from original)
        if self.training:
            mask = torch.ones(output.size(0), 1, output.size(-2), output.size(-1), device=output.device) * 0.95
            mask = torch.bernoulli(mask).float()
            img_for_resized = img_for_residual
            if img_for_resized.shape[-2:] != output.shape[-2:]:
                img_for_resized = F.interpolate(img_for_resized, size=output.shape[-2:], mode='bilinear', align_corners=False)
            output = mask * output + (1. - mask) * img_for_resized

        # final residual add with the original image (first channel) and tanh
        img_res = img_for_residual
        if img_res.shape[-2:] != output.shape[-2:]:
            img_res = F.interpolate(img_res, size=output.shape[-2:], mode='bilinear', align_corners=False)

        fake1 = torch.tanh(output + img_res)
        return fake1


# --------------------------
# Discriminator (PatchGAN-like) — CNN-based (kept from original style)
# --------------------------
class Discriminator(nn.Module):
    """Discriminator network with PatchGAN-like architecture."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        # Keep final conv small (3x3) as in original; receptive field & patch behavior is kept.
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return h, out_src


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # quick smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    c_dim = 5
    H = W = 256
    dummy_in = torch.randn(B, 1 + c_dim, H, W).to(device)

    G = SwinGenerator(conv_dim=64, c_dim=c_dim, repeat_num=3, device=device, num_memory=4,
                      swin_model_name='swin_tiny_patch4_window7_224', pretrained=False).to(device)
    D = Discriminator(image_size=H, conv_dim=64, c_dim=c_dim, repeat_num=4).to(device)

    G.train()
    out = G(dummy_in, bs=B)  # shape -> (B, 1, H, W)
    print("G out shape:", out.shape)

    # Discriminator input must be single-channel image; feed generated image (detach) or real images.
    h, out_src = D(out.detach())
    print("D feature shape:", h.shape, "D out_src shape:", out_src.shape)


        
