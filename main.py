import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader, position_distributions = dataloader_factory(args)
    model = model_factory(args, position_distributions)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()

    generate_recommendations = (input('Generate recommendations? y/[n]: ') == 'y')
    if generate_recommendations:
        recommendations, recommendation_positions = trainer.recommend()
        trainer.eval_position_bias(recommendations, recommendation_positions, position_distributions)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
