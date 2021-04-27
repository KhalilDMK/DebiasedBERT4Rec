from .bert import BERTModel
from .tf import TFModel
from .debiased_bert import DebiasedBERTModel
from .dae import DAEModel
from .vae import VAEModel

MODELS = {
    BERTModel.code(): BERTModel,
    TFModel.code(): TFModel,
    DebiasedBERTModel.code(): DebiasedBERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel
}


def model_factory(args, pos_dist, train_popularity_vector_loader):
    model = MODELS[args.model_code]
    return model(args, pos_dist, train_popularity_vector_loader)
