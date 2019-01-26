#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import checkpoint_sequential


DEBUG = True


def my_print(string, pref=None):
    
    if pref is not None:
        string = pref + string
    if DEBUG:
        print(string)
    


class MultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or cls._future_mask.shape < size:
            cls._future_mask = torch.triu(torch.ones(size[0], size[1], dtype=torch.uint8, device=device), 1)

        mask = cls._future_mask[:size[0], :size[1]]

        return mask

    def __init__(self, n_features, n_heads, dropout, layer_num=None):
        super(MultiheadAttention, self).__init__()
        assert n_features % n_heads == 0
        
        self.layer_num = layer_num

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        my_print(f'_split_heads | input x size: x (batch_size, seq_len, num_features) {x.size()}', self.pref())
        # x (batch_size, seq_len, num_features)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        my_print(f'_split_heads | view x size: x (batch_size, seq_len, self.n_heads, num_features // self.n_heads) {x.size()}', self.pref())
        # x (batch_size, seq_len, self.n_heads, num_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)
        my_print(f'_split_heads | permute x size: {x.size()}', self.pref())

        return x

    def _attn(self, q, k, v, apply_future_mask=True, padding_mask=None, dump=None):
        my_print(f'_attn | q size: {q.size()}', self.pref())
        my_print(f'_attn | k size: {k.size()}', self.pref())
        my_print(f'_attn | v size: {v.size()}', self.pref())
        
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)
        
        my_print(f'_attn | w = q * k: {w.size()}', self.pref())

        if apply_future_mask:
            future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
            w.masked_fill_(future_mask, float('-inf'))
        
        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        if padding_mask is not None:
            w.masked_fill_(padding_mask.all(dim=-1).unsqueeze(1).unsqueeze(2).unsqueeze(3), 0)
           
        if dump is not None:
            my_print('dump is not none', self.pref())
            dump.append((self.layer_num, self.run_i, w.cpu()))
        else:
            my_print('None', self.pref())
            
        my_print(f'w_size{w.size()}', self.pref())
        my_print(f'v_size{v.size()}', self.pref())
            
        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        my_print(f'_merge_heads | input x size: {x.size()}', self.pref())
        x = x.permute(0, 2, 1, 3).contiguous()
        my_print(f'_merge_heads | permute x size: {x.size()}', self.pref())
        x = x.view(x.shape[0], x.shape[1], self.n_features)
        my_print(f'_merge_heads | view x size: {x.size()}', self.pref())
        
        return x
    
    def pref(self):
         return f'layer_num: {self.layer_num} | run_i: {self.run_i} | '

    def forward(self, query, key, value, padding_mask, dump=None, run_i=None):
        self.run_i = run_i
        
        qkv_same = (query.data_ptr() == key.data_ptr() == value.data_ptr())
        kv_same = (key.data_ptr() == value.data_ptr())
        
        my_print(f'- q_size: {query.size()}', self.pref())
        my_print(f'- k_size: {key.size()}', self.pref())
        my_print(f'- v_size: {value.size()}', self.pref())
        my_print(f'- qkv_same: {qkv_same}', self.pref())
        my_print(f'- kv_same: {kv_same}', self.pref())

        if qkv_same:
            query, key, value = self.qkv_proj(query).split(self.n_features, dim=-1)
            apply_future_mask = True  # self-attention
        elif kv_same:
            q_w, q_b = self.qkv_proj.weight[:self.n_features, :], self.qkv_proj.bias[:self.n_features]
            query = F.linear(query, q_w, q_b)
            kv_w, kv_b = self.qkv_proj.weight[self.n_features:, :], self.qkv_proj.bias[self.n_features:]
            key, value = F.linear(key, kv_w, kv_b).split(self.n_features, dim=-1)
            apply_future_mask = False
        else:
            assert False
        
        my_print(f'+ q_size: {query.size()}', self.pref())
        query = self._split_heads(query)
        my_print(f'= q_size: {query.size()}', self.pref())
        
        my_print(f'+ k_size: {key.size()}', self.pref())
        key = self._split_heads(key, is_key=True)
        my_print(f'= k_size: {key.size()}', self.pref())
        
        my_print(f'+ v_size: {value.size()}', self.pref())
        value = self._split_heads(value)
        my_print(f'= v_size: {value.size()}', self.pref())
        
        x = self._attn(query, key, value, apply_future_mask, padding_mask, dump=dump)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x


class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, in_features, middle_features, dropout):
        super(FeedForward, self).__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = FeedForward.gelu(self.layer_1(x))
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout, layer_num=None):
        super(TransformerBlock, self).__init__()
        self.layer_num = layer_num
        
        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout, layer_num=self.layer_num)
        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask, *contexts, dump=None):
        '''contexts = [(context1, padding_mask1), ...]'''
        
#         my_print(f'layer_num: {self.layer_num}')

        inputs = (x, padding_mask) + contexts

        full_attn = 0
        n_attn = len(inputs) // 2
        for i in range(0, len(inputs), 2):
            c, m = inputs[i], inputs[i+1].byte()
            a = self.attn(x, c, c, m, dump=dump, run_i=i//2)
            full_attn += (a / n_attn)

        full_attn = self.dropout(full_attn)
        x = self.attn_norm(x + full_attn)

        f = self.ff(x)
        f = self.dropout(f)
        x = self.ff_norm(x + f)

        return (x, padding_mask) + contexts


class TransformerModule(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout, 
                 n_segments=None):
        super(TransformerModule, self).__init__()

        self.embeddings = nn.Embedding(n_embeddings, embeddings_size, padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(n_pos_embeddings + 1, embeddings_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList([TransformerBlock(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout, layer_num=i) for i in range(n_layers)])
        self.n_segments = n_segments        
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)

    def forward(self, x, enc_contexts=[], dump=None):
        padding_mask = x.eq(self.embeddings.padding_idx)

        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pos_embeddings.padding_idx)
        
        x = self.embeddings(x) * math.sqrt(self.embeddings.embedding_dim) + self.pos_embeddings(positions)
        x = self.embed_dropout(x)

        enc_contexts = sum(enc_contexts, ())

        if self.n_segments is not None:
            padding_mask = padding_mask.float()  # fucking checkpoint_sequential
            padding_mask.requires_grad_()  # fucking checkpoint_sequential
            out = checkpoint_sequential(self.layers, self.n_segments, x, padding_mask, *enc_contexts)
            x = out[0]
        else:
            for layer in self.layers:
                out = layer(x, padding_mask, *enc_contexts, dump=dump)
                x = out[0]
        
        return x, padding_mask
