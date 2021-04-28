from .bert import BERTModel
from .tf import TFModel


MODELS = {
    BERTModel.code(): BERTModel,
    TFModel.code(): TFModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
