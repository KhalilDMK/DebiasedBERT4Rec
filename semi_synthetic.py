from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import itertools


def train():
    args.model_code, args.dataloader_code, args.trainer_code = 'tf', 'tf', 'tf'
    if args.tf_target == 'exposure':
        args.best_metric = 'AUC'
    elif args.tf_target == 'relevance':
        args.best_metric = 'MSE'
    train_loader, val_loader, test_loader, position_distributions, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader, user_count, item_count = dataloader_factory(
        args, export_root)
    args.user_count, args.item_count = user_count, item_count
    model = model_factory(args, position_distributions, train_popularity_vector_loader)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, position_distributions,
                              train_popularity_vector_loader, val_popularity_vector_loader,
                              test_popularity_vector_loader)
    trainer.train()
    trainer.test()
    trainer.save_test_performance()

def tune():
    num_configurations = args.num_configurations
    num_reps = args.num_reps
    hyperparameters = ['bert_hidden_units', 'train_batch_size']
    hyper_tun_configurations = []
    for hyperparameter in hyperparameters:
        exec('tune_' + hyperparameter + ' = eval(eval("args.tune_" + hyperparameter))')
        hyper_tun_configurations.append(eval('tune_' + hyperparameter))
    print(hyper_tun_configurations)
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

def generate():
    args.model_code, args.dataloader_code, args.trainer_code = 'tf', 'tf', 'tf'
    if args.tf_target == 'exposure':
        args.best_metric = 'AUC'
    elif args.tf_target == 'relevance':
        args.best_metric = 'MSE'
    train_loader, val_loader, test_loader, gen_loader, position_distributions, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader, user_count, item_count = dataloader_factory(
        args, export_root)
    args.user_count, args.item_count = user_count, item_count
    model = model_factory(args, position_distributions, train_popularity_vector_loader)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, position_distributions,
                              train_popularity_vector_loader, val_popularity_vector_loader,
                              test_popularity_vector_loader)
    trainer.train()
    trainer.reconstruct(gen_loader)

def train_semi_synthetic():
    train_loader, val_loader, test_loader, temporal_propensity_dataloader, temporal_relevance_dataloader, static_propensity_dataloader, user_count, item_count = dataloader_factory(
        args, export_root)
    model = model_factory(args, temporal_propensity_dataloader, static_propensity_dataloader)
    #trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, temporal_propensity_dataloader,
    #                          static_propensity_dataloader, static_propensity_dataloader,
    #                          static_propensity_dataloader)
    #trainer.train()
    #trainer.test()


if __name__ == '__main__':
    args.iteration = -1
    export_root = setup_train(args)
    if args.mode == 'train':
        train()
    elif args.mode == 'tune':
        tune()
    elif args.mode == 'generate':
        generate()
    elif args.mode == 'train_semi_synthetic':
        train_semi_synthetic()
    else:
        raise ValueError('Invalid mode')
