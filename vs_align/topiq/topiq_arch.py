# Original TOPIQ metric proposed by Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
# https://arxiv.org/abs/2308.03060

# Modifications for more efficient frame matching by pifroggi
# or tepete on the "Enhance Everything!" Discord Server
# https://github.com/pifroggi/vs_align

# Model trained by IQA-PyTorch
# https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_fr_kadid_res50-2c4cc61d.pth

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy


def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2
        src2, self.attn_map = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, self.attn_map = self.multihead_attn(query=tgt2,
                                                  key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, src):
        output = src

        for layer in self.layers:
            output = layer(output)
        
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)
        
        return output


class CFANet(nn.Module):
    def __init__(self,
                 semantic_model,
                 feature_dim_list,
                 num_class=1,
                 inter_dim=256,
                 num_heads=4,
                 num_attn_layers=1,
                 dprate=0.1,
                 activation='gelu',
                 out_act=False,
                 block_pool='weighted_avg',
                 ):
        super().__init__()

        self.num_class = num_class 
        self.block_pool = block_pool

        self.semantic_model = semantic_model
        self.feature_dim_list = feature_dim_list

        # Define self-attention and cross scale attention blocks
        self.fusion_mul = 3 
        ca_layers = sa_layers = num_attn_layers 

        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # Gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3
            self.weight_pool.append(
                nn.Sequential(
                    nn.Conv2d(dim // 3, 64, 1, stride=1),
                    self.act_layer,
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    self.act_layer,
                    nn.Conv2d(64, 1, 3, stride=1, padding=1),
                    nn.Sigmoid()
                )
            )
                
            self.dim_reduce.append(nn.Sequential(
                nn.Conv2d(dim, inter_dim, 1, 1),
                self.act_layer,
                )
            )

            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        # Cross scale attention
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # Attention pooling and MLP layers 
        self.attn_pool = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)

        linear_dim = inter_dim
        self.score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, self.num_class),
        ]

        # Make sure output is positive, useful for 2AFC datasets with probability labels
        if out_act and self.num_class == 1:
            self.score_linear.append(nn.Softplus())
        
        if self.num_class > 1:
            self.score_linear.append(nn.Softmax(dim=-1))

        self.score_linear = nn.Sequential(*self.score_linear)
        
        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        self.eps = 1e-8


    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)


    def dist_func(self, x, y, eps=1e-12):
        return torch.sqrt((x - y) ** 2 + eps)
        

    def compare_features(self, dist_feat_list, ref_feat_list):
        start_level = 0
        end_level = len(dist_feat_list) 

        b, c, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat((
            self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]),
            self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1)
        ), dim=1) 

        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]
            tmp_ref_feat = ref_feat_list[i]
            
            # Gated local pooling
            diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)
            tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
            weight = self.weight_pool[i](diff)
            tmp_feat = tmp_feat * weight

            tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

            # Self-attention
            tmp_pos_emb = F.interpolate(pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False)
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)

            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb

            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)
        
        # Cross-scale attention: coarse to fine
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1] 
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)

        final_feat = self.attn_pool(query)
        out_score = self.score_linear(final_feat.mean(dim=0))

        return out_score


    def forward(self, x, y):
        return dist_to_mos(self.compare_features(x, y))
