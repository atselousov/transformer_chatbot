import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, time_size):
        if not hasattr(cls, '_future_mask'):
            cls._future_mask = torch.triu(torch.ones(time_size, time_size, dtype=torch.uint8), 1)

        if cls._future_mask.shape[0] < time_size:
            cls._future_mask = torch.triu(cls._future_mask.resize_(time_size, time_size).fill_(1), 1)

        mask = cls._future_mask[:time_size, :time_size]

        return mask

    def __init__(self, n_features, n_heads, dropout):
        super(MultiheadAttention, self).__init__()

        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        x = x.view(*x.shape[:-1], self.n_heads, self.n_features // self.n_head)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, padding_mask=None):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_head)

        feature_mask = MultiheadAttention._get_future_mask(w.shape[-1]).unsqueeze(0).unsqueeze(0)
        w.masked_fill_(feature_mask, float('-inf'))
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            w.masked_fill_(padding_mask, float('-inf'))

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(*x.shape[:-2], self.n_features)

        return x

    def forward(self, x, padding_mask):
        query, key, value = self.qkv_proj(x).split(self.n_features, dim=-1)

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x


class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, in_features, middle_features, dropout):
        super(FeedForward, self).__init__()

        self.layer1 = nn.Linear(in_features, middle_features)
        self.layer2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer1.weight, std=0.02)
        nn.init.normal_(self.layer2.weight, std=0.02)

    def forward(self, x):
        x = FeedForward.gelu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        a = self.attn(x, padding_mask)
        a = self.dropout(a)
        x = self.attn_norm(x + a)

        f = self.ff(x)
        f = self.dropout(f)
        x = self.ff_norm(x + f)

        return x


class TransformerModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout):
        super(TransformerModel, self).__init__()
        
        self.embeddings = nn.Embedding(n_embeddings, embeddings_size, padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(n_pos_embeddings + 1, embeddings_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList([TransformerBlock(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout) for _ in range(n_layers)])
        self.pre_softmax = nn.Linear(embeddings_size, n_embeddings, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)
        self.pre_softmax.weight = self.embeddings.weight

    def forward(self, x):
        padding_mask = x.eq(self.embeddings.padding_idx)

        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pos_embeddings.padding_idx)

        x = self.embeddings(x) * math.sqrt(self.embeddings.embeddings_size) + self.pos_embeddings(positions)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, padding_mask)

        x = self.pre_softmax(x)

        return x