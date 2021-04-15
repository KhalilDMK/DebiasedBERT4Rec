from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args, pos_dist, train_popularity_vector_loader):
        super().__init__(args)
        self.bert = BERT(args, pos_dist, train_popularity_vector_loader)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
