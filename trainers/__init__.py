from .bert import BERTTrainer
from .dae import DAETrainer
from .vae import VAETrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, pos_dist, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root, pos_dist, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader)
