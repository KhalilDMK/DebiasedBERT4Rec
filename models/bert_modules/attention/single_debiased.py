import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class DebiasedAttention(nn.Module):
    """
    Compute Scaled Dot Product Attention
    """

    def forward(self, query, key, value, temp_prop_enc, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # Apply temporal propensity encoding
        temp_prop_enc = temp_prop_enc.repeat(1, scores.shape[1] * scores.shape[2]).view(scores.shape)
        scores = scores * temp_prop_enc * torch.transpose(temp_prop_enc, 2, 3)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        #p_attn = p_attn * temp_prop_enc

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
