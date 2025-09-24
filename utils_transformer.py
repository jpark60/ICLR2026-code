import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.):
        super().__init__()
        self.head = head
        self.feats = feats
        # self.sqrt_d = feats ** 0.5 
        self.sqrt_d = (feats // head) ** 0.5 

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        h = self.head
        d = f // h

        # project & split heads
        q = self.q(x).view(b, n, h, d).transpose(1,2)
        k = self.k(x).view(b, n, h, d).transpose(1,2)
        v = self.v(x).view(b, n, h, d).transpose(1,2)

        # scaled dot-product
        attn = torch.softmax((q @ k.transpose(-2,-1)) / self.sqrt_d, dim=-1)  # (b,h,n,n)

        # combine
        out = (attn @ v).transpose(1,2).contiguous().view(b, n, f)  # (b,n,f)
        return self.o(self.dropout(out))

class TransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.):
        super().__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class EmberTransformer(nn.Module):
    def __init__(
        self,
        in_feats: int = 85,
        num_classes: int = 42,
        hidden: int = 384,
        mlp_hidden: int = 384*3,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.feat_emb = nn.Linear(in_feats, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if use_cls_token else None
        n_tokens = 1 + (1 if use_cls_token else 0)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, hidden))

        # Stack of TransformerEncoders
        encoders = [
            TransformerEncoder(hidden, mlp_hidden, head=nhead, dropout=dropout)
            for _ in range(num_layers)
        ]
        self.encoder = nn.Sequential(*encoders)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        """
        x: (B, D) Ember features
        returns logits (B, C), and optionally features for contrastive/loss (B, hidden)
        """
        B = x.size(0)
        # (B, hidden) -> (B, 1, hidden)
        tokens = self.feat_emb(x).unsqueeze(1)

        if self.use_cls:
            # (B, 1, hidden)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # now (B, 2, hidden)

        # add pos embed
        tokens = tokens + self.pos_emb

        # run through transformer stack
        out = self.encoder(tokens)  # (B, n_tokens, hidden)

        if self.use_cls:
            feat = out[:, 0]          # (B, hidden)
        else:
            feat = out[:, 0]          # same shape

        logits = self.classifier(feat)  # (B, num_classes)
        return logits, feat

    def feature_extract(self, x):
        """
        Extract features without classification
        """
        B = x.size(0)
        # (B, hidden) -> (B, 1, hidden)
        tokens = self.feat_emb(x).unsqueeze(1)

        if self.use_cls:
            # (B, 1, hidden)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # now (B, 2, hidden)

        # add pos embed
        tokens = tokens + self.pos_emb

        # run through transformer stack
        out = self.encoder(tokens)  # (B, n_tokens, hidden)

        if self.use_cls:
            feat = out[:, 0]          # (B, hidden)
        else:
            feat = out[:, 0]          # same shape

        return feat

EmberTransformerPyTorch = EmberTransformer
