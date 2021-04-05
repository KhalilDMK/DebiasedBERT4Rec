import torch
from torch import nn as nn

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from models.bert_modules.debiased_transformer import DebiasedTransformerBlock
from utils import fix_random_seed_as
import sys


class DebiasedBERT(nn.Module):
    def __init__(self, args, pos_dist, train_popularity_vector_loader):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        self.max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        #self.pos_dist = 1 - torch.cat((pos_dist, torch.zeros(1, self.max_len)), 0)
        #print(pos_dist)
        #print(pos_dist[0])
        #print(pos_dist.shape)
        #print(torch.cat((pos_dist, torch.zeros(1, self.max_len)), 0).shape)
        #self.pos_dist = torch.cat((pos_dist, torch.zeros(1, self.max_len)), 0)
        #self.pos_dist = torch.cat((pos_dist, torch.zeros(1, self.max_len)), 0) + sys.float_info.epsilon
        self.device = args.device
        self.pos_dist = pos_dist.to(self.device)
        self.pos_dist = torch.cat((torch.zeros(1, self.max_len).to(self.device), self.pos_dist, torch.zeros(1, self.max_len).to(self.device)), 0) + sys.float_info.epsilon
        self.train_popularity_vector = train_popularity_vector_loader.popularity_vector.to(self.device)
        self.train_popularity_vector = torch.cat((torch.FloatTensor([0]).to(self.device), self.train_popularity_vector, torch.FloatTensor([0]).to(self.device))) + sys.float_info.epsilon
        self.att_debiasing = args.att_debiasing

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=self.max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [DebiasedTransformerBlock(hidden, heads, hidden * 4, dropout)] + [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers - 1)])

    def forward(self, x):

        # temporal propensity encoding of input
        #print(self.pos_dist.shape)
        #print(x.shape)
        #print([self.pos_dist[x[i].cpu(), range(x.shape[1])].view(-1, self.max_len) for i in range(x.shape[0])][0].shape)
        #temp_prop_enc = torch.cat([self.pos_dist[x[i].cpu(), range(x.shape[1])].view(-1, self.max_len) for i in range(x.shape[0])], 0)
        temp_prop_enc = self.pos_dist[x.flatten().cpu(), list(range(x.shape[1])) * x.shape[0]].view(-1, self.max_len)
        stat_prop_enc = self.train_popularity_vector[x.flatten()].view_as(x) + sys.float_info.epsilon
        #print(temp_prop_enc.shape)

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        block = 0
        for transformer in self.transformer_blocks:
            if block == 0:
                x = transformer.forward(x, mask, temp_prop_enc, stat_prop_enc, self.att_debiasing)
            else:
                x = transformer.forward(x, mask)
            block += 1

        return x

    def init_weights(self):
        pass
