import torch
import torch.nn as nn


class ExternalAttention(nn.Module):
    def __init__(self, dim: int, S: int = 64):
        super().__init__()
        self.M_k = nn.Linear(dim, S, bias=False)
        self.M_v = nn.Linear(S, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.M_k(x)
        attn = self.softmax(attn)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-9)
        out = self.M_v(attn)
        return out

class MultiHeadExternalAttentionShared(nn.Module):
    """
    Faithful to the paper's literal text: M_k and M_v are a SINGLE pair of
    learnable memories, shared identically across all heads (not one pair
    per head). Each head still operates on its own slice of the embedding
    dimension (head_dim = dim // num_heads), but all heads apply the SAME
    ExternalAttention module to their slice.
    """
 
    def __init__(self, dim: int, num_heads: int = 8, S: int = 64, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
 
        # ONE shared ExternalAttention module for all heads
        self.shared_ea = ExternalAttention(self.head_dim, S=S)
 
        self.W_o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_split = x.chunk(self.num_heads, dim=-1)
 
        # same shared_ea module applied to every head's slice
        head_outputs = [self.shared_ea(x_i) for x_i in x_split]
 
        concat = torch.cat(head_outputs, dim=-1)
        out = self.W_o(concat)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EAViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int,
                 S: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mea = MultiHeadExternalAttentionShared(dim, num_heads=num_heads, S=S, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x = self.mea(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 64,
                 in_channels: int = 3, dim: int = 32):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1, 2)

        class_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        x = x + self.pos_embed
        return x


class EAViT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 64,
        in_channels: int = 3,
        num_classes: int = 15,
        dim: int = 32,
        depth: int = 16,
        num_heads: int = 8,
        S: int = 64,
        mlp_ratio: float = 4.0,
        head_hidden_dims=(2048, 1024),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, dim=dim,
        )

        encoder_mlp_hidden = int(dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            EAViTBlock(
                dim=dim, num_heads=num_heads,
                mlp_hidden_dim=encoder_mlp_hidden, S=S, dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        head_layers = []
        in_dim = dim
        for h in head_hidden_dims:
            head_layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        head_layers += [nn.Linear(in_dim, num_classes)]
        self.mlp_head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        cls_out = x[:, 0]
        y = self.norm(cls_out)

        logits = self.mlp_head(y)
        return logits


if __name__ == "__main__":
    model = EAViT(
        img_size=256, patch_size=64, in_channels=3, num_classes=15,
        dim=32, depth=16, num_heads=8, S=64,
    )

    dummy = torch.randn(4, 3, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")