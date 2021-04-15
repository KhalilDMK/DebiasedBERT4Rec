import torch.nn as nn

from .attention import DebiasedMultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class DebiasedTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.debiased_attention = DebiasedMultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, temp_prop_enc, stat_prop_enc, att_debiasing):
        x = self.input_sublayer(x, lambda _x: self.debiased_attention.forward(_x, _x, _x, temp_prop_enc, stat_prop_enc, att_debiasing, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
