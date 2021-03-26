import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    args.iteration += 1
    train_loader, val_loader, test_loader, position_distributions, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader = dataloader_factory(args, export_root)
    model = model_factory(args, position_distributions)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader)
    trainer.train()

    #test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    #if test_model:
    trainer.test()

    #generate_recommendations = (input('Generate recommendations? y/[n]: ') == 'y')
    #if generate_recommendations:
    recommendations, recommendation_positions = trainer.recommend()
    trainer.eval_position_bias(recommendations, recommendation_positions, position_distributions)


def loop():
    for i in range(args.num_iterations):
        print('#' * 20 + '\nIteration ' + str(i) + '\n' + '#' * 20)
        train()


if __name__ == '__main__':
    args.iteration = -1
    export_root = setup_train(args)
    if args.mode == 'train':
        train()
    if args.mode == 'loop':
        loop()
    else:
        raise ValueError('Invalid mode')
