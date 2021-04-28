from .bert import BERTTrainer
from .tf import TFTrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    TFTrainer.code(): TFTrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, train_temporal_popularity=[], train_popularity_loader=[], val_popularity_loader=[], test_popularity_loader=[], temporal_propensity=[], temporal_relevance=[], static_propensity=[]):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, train_temporal_popularity, train_popularity_loader, val_popularity_loader, test_popularity_loader, temporal_propensity, temporal_relevance, static_propensity)
