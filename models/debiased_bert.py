from .base import BaseModel
from .bert_modules.debiased_bert import DebiasedBERT

import torch.nn as nn


class DebiasedBERTModel(BaseModel):
    def __init__(self, args, pos_dist):
        super().__init__(args)
        self.bert = DebiasedBERT(args, pos_dist)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'debiased_bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
