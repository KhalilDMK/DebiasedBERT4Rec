import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import itertools


def train():
    args.iteration += 1
    train_loader, val_loader, test_loader, position_distributions, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader, user_count, item_count = dataloader_factory(args, export_root)
    model = model_factory(args, position_distributions, train_popularity_vector_loader)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, position_distributions, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader)
    trainer.train()

    #test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    #if test_model:
    trainer.test()

    #generate_recommendations = (input('Generate recommendations? y/[n]: ') == 'y')
    #if generate_recommendations:
    recommendations, recommendation_positions = trainer.recommend()
    trainer.eval_position_bias(recommendations, recommendation_positions)


def loop():
    for i in range(args.num_iterations):
        print('#' * 20 + '\nIteration ' + str(i) + '\n' + '#' * 20)
        torch.cuda.empty_cache()
        train()
    plot_evolution(export_root, args.num_iterations)


def tune():
    num_configurations = args.num_configurations
    num_reps = args.num_reps
    hyperparameters = ['bert_hidden_units', 'bert_num_blocks', 'bert_num_heads', 'train_batch_size', 'bert_dropout', 'bert_mask_prob', 'skew_power']
    hyper_tun_configurations = []
    for hyperparameter in hyperparameters:
        exec('tune_' + hyperparameter + ' = eval(eval("args.tune_" + hyperparameter))')
        hyper_tun_configurations.append(eval('tune_' + hyperparameter))
    hyper_tun_configurations = random.sample(set(itertools.product(*hyper_tun_configurations)), num_configurations)
    for configuration in hyper_tun_configurations:
        print('#' * 50 + '\nconfiguration ' + str(configuration) + '\n' + '#' * 50)
        for i in range(len(hyperparameters)):
            exec('args.' + hyperparameters[i] + ' = configuration[i]')
        for rep in range(num_reps):
            args.rep = rep
            print('\nRep: ' + str(rep) + '\n' + '#' * 20)
            torch.cuda.empty_cache()
            train()
    summarize_tuning_results(export_root, hyperparameters)


if __name__ == '__main__':
    args.iteration = -1
    export_root = setup_train(args)
    if args.mode == 'train':
        train()
    elif args.mode == 'loop':
        loop()
    elif args.mode == 'tune':
        tune()
    else:
        raise ValueError('Invalid mode')
