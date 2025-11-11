
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ----------------------------- Utilities -----------------------------
def _make_divisible(x, m=8):
    return int((x + m - 1) // m) * m

# ----------------------------- Modules -----------------------------
class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.
    Equivalent to a conv with kernel_size=patch_size, stride=patch_size.
    Input: x (B, C, H, W)
    Output: (B, N, embed_dim) where N = (H/patch)*(W/patch)
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # flatten will convert to (B, embed_dim, H_patch, W_patch) -> (B, N, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Expected image size ({self.img_size}x{self.img_size}), got ({H}x{W})."
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x

class Attention(nn.Module):
    """Multi-head self-attention (scaled dot-product)"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # single linear to compute QKV and then split
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv shape -> (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, N, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn_scores.softmax(dim=-1)  # row-wise softmax over keys
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLP(nn.Module):
    """Simple 2-layer MLP with GELU activation"""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer encoder block: Pre-LN, MSA, residual, MLP, residual"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = True,
                 p: float = 0., attn_p: float = 0., drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_p, proj_drop=p)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PreNorm Attention
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # PreNorm MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ----------------------------- DropPath (Stochastic Depth) -----------------------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    This implementation is from timm and is a commonly used variant.
    """
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# ----------------------------- Vision Transformer -----------------------------
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 representation_size: Optional[int] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim  # used by some external code conventions
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # transformer encoder blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, p=drop_rate, attn_p=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # representation / head
        self.representation_size = representation_size
        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
            self.head = nn.Linear(representation_size, num_classes)
        else:
            self.pre_logits = nn.Identity()
            self.head = nn.Linear(embed_dim, num_classes)

        # weight init
        self._init_weights()

    def _init_weights(self):
        # Initialize parameters similar to original ViT and common practice
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: logits (B, num_classes)
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, D)
        x = self.pre_logits(cls_out)
        logits = self.head(x)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # return features before the head (useful for transfer learning)
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

# ----------------------------- Quick smoke test -----------------------------
if __name__ == "__main__":
    # Create a ViT-Base like model (smaller sizes possible)
    model = VisionTransformer(img_size=64, patch_size=16, embed_dim=128, depth=4, num_heads=8, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Forward a random batch
    dummy = torch.randn(2, 3, 64, 64).to(device)
    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, 10)

    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Params: {param_count:,}")

