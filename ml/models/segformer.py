import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding using Overlapped Conv"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class SegFormerB0(nn.Module):
    """
    SegFormer-B0 architecture:
    Mix-Transformer (MiT-B0) hierarchical encoder + Lightweight All-MLP Decoder.
    Efficient attention mechanisms for real-time plant disease segmentation.
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()
        # MiT-B0 stages: embed dims: 32, 64, 160, 256
        self.dims = [32, 64, 160, 256]

        # Stage 1
        self.patch_embed1 = OverlapPatchEmbed(7, 4, 3, self.dims[0])
        self.block1 = nn.ModuleList([TransformerBlock(self.dims[0], num_heads=1, sr_ratio=8) for _ in range(2)])
        self.norm1 = nn.LayerNorm(self.dims[0])

        # Stage 2
        self.patch_embed2 = OverlapPatchEmbed(3, 2, self.dims[0], self.dims[1])
        self.block2 = nn.ModuleList([TransformerBlock(self.dims[1], num_heads=2, sr_ratio=4) for _ in range(2)])
        self.norm2 = nn.LayerNorm(self.dims[1])

        # Stage 3
        self.patch_embed3 = OverlapPatchEmbed(3, 2, self.dims[1], self.dims[2])
        self.block3 = nn.ModuleList([TransformerBlock(self.dims[2], num_heads=5, sr_ratio=2) for _ in range(2)])
        self.norm3 = nn.LayerNorm(self.dims[2])

        # Stage 4
        self.patch_embed4 = OverlapPatchEmbed(3, 2, self.dims[2], self.dims[3])
        self.block4 = nn.ModuleList([TransformerBlock(self.dims[3], num_heads=8, sr_ratio=1) for _ in range(2)])
        self.norm4 = nn.LayerNorm(self.dims[3])

        # MLP Decoder
        decoder_dim = 256
        self.linear_c1 = nn.Linear(self.dims[0], decoder_dim)
        self.linear_c2 = nn.Linear(self.dims[1], decoder_dim)
        self.linear_c3 = nn.Linear(self.dims[2], decoder_dim)
        self.linear_c4 = nn.Linear(self.dims[3], decoder_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        B = x.shape[0]

        # Stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2)

        # Stage 2
        x2, H2, W2 = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2, H2, W2)
        x2 = self.norm2(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2)

        # Stage 3
        x3, H3, W3 = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3, H3, W3)
        x3 = self.norm3(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2)

        # Stage 4
        x4, H4, W4 = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4, H4, W4)
        x4 = self.norm4(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2)

        # MLP Decoder
        c1 = self.linear_c1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        c2 = self.linear_c2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        c3 = self.linear_c3(x3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        c4 = self.linear_c4(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        c2 = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        c3 = F.interpolate(c3, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        fused = self.linear_fuse(torch.cat([c1, c2, c3, c4], dim=1))
        logits_low = self.classifier(fused)
        logits = F.interpolate(logits_low, size=input_size, mode="bilinear", align_corners=False)
        return logits
