from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim
import matplotlib.pyplot as plt
import pandas as pd


def setup_train(args):
    set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    create_recommendation_folder(export_root)
    export_experiments_config_as_json(args, export_root)

    #pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root


def create_experiment_export_folder(args):
    experiment_dir, experiment_description = args.experiment_dir, args.experiment_description
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path


def create_recommendation_folder(export_root):
    if not os.path.exists(Path(export_root).joinpath('recommendations')):
        os.mkdir(Path(export_root).joinpath('recommendations'))


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def load_weights(model, path):
    pass


def save_test_result(export_root, result):
    filepath = Path(export_root).joinpath('test_result.txt')
    with filepath.open('w') as f:
        json.dump(result, f, indent=2)


def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def set_up_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))


def load_pretrained_weights(model, path):
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
    model.load_state_dict(model_state_dict)


def setup_to_resume(args, model, optimizer):
    chk_dict = torch.load(os.path.join(os.path.abspath(args.resume_training), 'models/checkpoint-recent.pth'))
    model.load_state_dict(chk_dict[STATE_DICT_KEY])
    optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])


def create_optimizer(model, args):
    if args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


def plot_evolution(export_root, num_iterations):
    create_plots_folder(export_root)
    results = merge_results(export_root, num_iterations)
    generate_plots(results, export_root, num_iterations)


def create_plots_folder(export_root):
    os.mkdir(Path(export_root).joinpath('plots'))


def merge_results(export_root, num_iterations):
    results = None
    for i in range(num_iterations):
        with open(Path(export_root).joinpath('logs', 'test_metrics_iter_' + str(i) + '.json')) as f:
            if not results:
                results = json.load(f)
                results = {k:[v] for k, v in results.items()}
            else:
                intermediate = json.load(f)
                results = {k:v + [intermediate[k]] for k, v in results.items()}
    return results


def generate_plots(results, export_root, num_iterations):
    for metric in results:
        plt.figure()
        plt.plot(range(num_iterations), results[metric])
        plt.title('Evolution of ' + str(metric) + ' wrt feedback loop iterations')
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        #plt.show()
        plt.savefig(Path(export_root).joinpath('plots', metric + '.png'))


def summarize_tuning_results(export_root, hyperparameters):
    log_path = Path(export_root).joinpath('logs')
    files = os.listdir(log_path)
    files = [x for x in files if x.startswith('test')]
    with open(log_path.joinpath(files[0])) as f:
        results = pd.DataFrame(index=range(len(files)), columns=hyperparameters + ['rep'] + list(json.load(f).keys()))
    frame_idx = 0
    for file in files:
        configuration = eval(file.split('_')[3])
        rep = eval(file.split('_')[-1][0:-5])
        for i in range(len(hyperparameters)):
            results[hyperparameters[i]][frame_idx] = configuration[i]
        results['rep'][frame_idx] = rep
        with open(log_path.joinpath(file)) as f:
            result = json.load(f)
        for metric in result:
            results[metric][frame_idx] = result[metric]
        frame_idx += 1
    results.to_csv(log_path.joinpath('results.csv'))


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
