from .base import BaseModel
import torch.nn as nn


class TFModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        user_count = args.user_count
        item_count = args.item_count
        num_hidden = args.tf_hidden_units
        num_timesteps = args.bert_max_len

        self.embed_session = nn.Embedding(user_count, num_hidden)
        self.embed_item = nn.Embedding(item_count, num_hidden)
        self.embed_time = nn.Embedding(num_timesteps, num_hidden)

        nn.init.normal_(self.embed_session.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.embed_time.weight, std=0.01)

    @classmethod
    def code(cls):
        return 'tf'

    def forward(self, user_indices, item_indices, timesteps):
        session_latent = self.embed_session(user_indices)
        item_latent = self.embed_item(item_indices)
        time_latent = self.embed_item(timesteps)

        prediction = (session_latent * item_latent * time_latent).sum(dim=-1)
        return prediction
