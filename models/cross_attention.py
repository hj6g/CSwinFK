import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=True)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def _split(self, t):
        B, N, C = t.shape
        return t.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    def forward(self, q_in, k_in):
        Q = self.q_proj(q_in)
        KV = self.kv_proj(k_in)
        K, V = KV.chunk(2, dim=-1)

        Qh = self._split(Q)
        Kh = self._split(K)
        Vh = self._split(V)

        attn = (Qh @ Kh.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ Vh).transpose(1, 2).reshape(
            q_in.size(0), q_in.size(1), self.embed_dim
        )
        out = self.out_drop(self.proj(out))
        return self.norm(q_in + out)


class CrossAttentionModel(nn.Module):
    def __init__(self, d: int = 256, h: int = 4, p: float = 0.1):
        super().__init__()
        self.ca1 = CrossAttentionLayer(d, h, p)  
        self.ca2 = CrossAttentionLayer(d, h, p)  

    def forward(self, cnn_tok, vit_tok):
        cnn_f = self.ca1(cnn_tok, vit_tok)   
        vit_f = self.ca2(vit_tok, cnn_tok)   
        return cnn_f, vit_f
