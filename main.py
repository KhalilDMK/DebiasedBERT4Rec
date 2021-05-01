from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import itertools


def train_bert_real():
    args.iteration += 1
    args.model_code, args.dataloader_code, args.trainer_code = 'bert', 'bert', 'bert'
    train_loader, val_loader, test_loader, train_temporal_popularity, train_popularity, val_popularity, test_popularity = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, train_temporal_popularity, train_popularity, val_popularity, test_popularity)
    trainer.train()
    trainer.test()
    recommendations = trainer.recommend()
    trainer.final_data_eval_save_results()


def loop_bert_real():
    for i in range(args.num_iterations):
        print('#' * 20 + '\nIteration ' + str(i) + '\n' + '#' * 20)
        torch.cuda.empty_cache()
        train_bert_real()
    plot_evolution(args.export_root, args.num_iterations)


def tune_bert_real():
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
            train_bert_real()
    summarize_tuning_results(args.export_root, hyperparameters)


def train_tf():
    args.iteration += 1
    args.model_code, args.dataloader_code, args.trainer_code = 'tf', 'tf', 'tf'
    if args.tf_target == 'exposure':
        args.best_metric = 'AUC'
    elif args.tf_target == 'relevance':
        args.best_metric = 'MSE'
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader)
    trainer.train()
    trainer.test()
    trainer.save_test_performance()


def tune_tf():
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
            train_tf()
    summarize_tuning_results(args.export_root, hyperparameters)


def generate_semi_synthetic():
    args.model_code, args.dataloader_code, args.trainer_code = 'tf', 'tf', 'tf'
    if args.tf_target == 'exposure':
        args.best_metric = 'AUC'
    elif args.tf_target == 'relevance':
        args.best_metric = 'MSE'
    train_loader, val_loader, test_loader, gen_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader)
    trainer.train()
    trainer.reconstruct(gen_loader)


def train_bert_semi_synthetic():
    args.iteration += 1
    args.model_code, args.dataloader_code, args.trainer_code = 'bert', 'bert', 'bert'
    train_loader, val_loader, test_loader, temporal_propensity, temporal_relevance, static_propensity = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, temporal_propensity=temporal_propensity, temporal_relevance=temporal_relevance, static_propensity=static_propensity)
    trainer.train()
    trainer.test()
    trainer.final_data_eval_save_results()


if __name__ == '__main__':
    args.iteration = -1
    args.export_root = setup_train(args)
    verify_loss_debiasing(args)
    if args.mode == 'train_bert_real':
        train_bert_real()
    elif args.mode == 'tune_bert_real':
        tune_bert_real()
    elif args.mode == 'loop_bert_real':
        loop_bert_real()
    elif args.mode == 'train_bert_semi_synthetic':
        train_bert_semi_synthetic()
    elif args.mode == 'tune_bert_semi_synthetic':
        print()
    elif args.mode == 'generate_semi_synthetic':
        generate_semi_synthetic()
    elif args.mode == 'train_tf':
        train_tf()
    elif args.mode == 'tune_tf':
        tune_tf()
    else:
        raise ValueError('Invalid mode')
