import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from .cnn_backbone import CNNBackbone_Res64
from .cross_attention import CrossAttentionModel
from .swin_transformer import swin_tiny_patch4_window7_224


class CNN_SwinTiny_CA(nn.Module):
    def __init__(self, n_classes: int = 2, embed_dim: int = 256):
        super().__init__()
        self.cnn = CNNBackbone_Res64()
        self.swin = swin_tiny_patch4_window7_224(num_classes=0)
        self.swin_proj = nn.Linear(768, embed_dim)
        self.cross = CrossAttentionModel(d=embed_dim, h=4, p=0.1)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)                         
        cnn_tok = cnn_feat.flatten(2).transpose(1, 2) 

        swin_tok = self.swin.forward_features(x)      
        swin_tok = self.swin_proj(swin_tok)           

        cnn_tok = F_nn.adaptive_avg_pool1d(
            cnn_tok.transpose(1, 2),
            swin_tok.size(1)
        ).transpose(1, 2)                             

        cnn_f, vit_f = self.cross(cnn_tok, swin_tok)

        vec = torch.cat(
            [cnn_f.mean(1), vit_f.mean(1)],
            dim=-1
        )                                             

        fused = self.mlp(vec)                         
        return self.head(fused)                       
