from .bert import BERTModel
from .debiased_bert import DebiasedBERTModel
from .dae import DAEModel
from .vae import VAEModel

MODELS = {
    BERTModel.code(): BERTModel,
    DebiasedBERTModel.code(): DebiasedBERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel
}


def model_factory(args, pos_dist):
    model = MODELS[args.model_code]
    return model(args, pos_dist)
