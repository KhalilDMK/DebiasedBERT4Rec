from templates import set_template
from datasets import DATASETS
import argparse


parser = argparse.ArgumentParser(description='Debiased_BERT4Rec')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train_bert_real', choices=['train_bert_real', 'tune_bert_real', 'loop_bert_real', 'train_bert_semi_synthetic', 'tune_bert_semi_synthetic', 'generate_semi_synthetic', 'train_tf', 'tune_tf'])
parser.add_argument('--template', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-1m', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=0, help='Only keep ratings greater than or equal to this value.')
parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings.')
parser.add_argument('--min_sc', type=int, default=5, help='Only keep items with more than min_sc ratings.')
parser.add_argument('--split', type=str, default='leave_one_out', choices=['leave_one_out'], help='How to split the '
                                                                                                  'datasets.')
parser.add_argument('--dataset_split_seed', type=int, default=0)
parser.add_argument('--generate_semi_synthetic_seed', type=int, default=1)

################
# Dataloader
################
parser.add_argument('--dataloader_random_seed', type=float, default=None)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=500)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not needed in BERT')
parser.add_argument('--train_negative_sample_size', type=int, default=0)
parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization.')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum.')
# lr scheduler #
parser.add_argument('--enable_lr_schedule', type=lambda x: (str(x).lower() == 'true'), default=True, help='Set True to enable learning rate decay.')
parser.add_argument('--decay_step', type=int, default=50, help='Decay step for StepLR.')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR.')
# feedback loop iterations #
parser.add_argument('--num_iterations', type=int, default=20, help='Number of feedback loop iterations.')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training.')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10], help='List of cutoffs k in Metrics@k.')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric used for optimizing the model.')
# recommendation #
parser.add_argument('--top_k_recom', type=int, default=10, help='Number of recommended items at each iteration '
                                                                  'from which the user is assumed to randomly select '
                                                                  'one. Used in the feedback loop.')

################
# Model
################
parser.add_argument('--model_init_seed', type=int, default=0)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=20, help='Max sequence length.')
parser.add_argument('--bert_hidden_units', type=int, default=256, help='Size of hidden vectors (d_model) in BERT.')
parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers.')
parser.add_argument('--bert_num_heads', type=int, default=4, help='Number of heads for multi-head attention.')
parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model.')
parser.add_argument('--bert_mask_prob', type=float, default=0.15, help='Probability for masking items in the training sequence.')
parser.add_argument('--skew_power', type=float, default=0.01, help='Skewing power applied on propensities to scale them and avoid numerical overflow.')
parser.add_argument('--loss_debiasing', type=str, default=None,
                    choices=[None, 'static_propensity', 'temporal_propensity', 'relevance', 'static_popularity', 'temporal_popularity'],
                    help="Type of debiasing to apply on the loss: "
                         "None is BERT4Rec, "
                         "'static_propensity' is IPS-BERT4Rec with semi-synthetic propensities, "
                         "'temporal_propensity' is ITPS-BERT4Rec with semi-synthetic propensities, "
                         "'relevance' is Oracle-BERT4Rec with semi-synthetic relevance, "
                         "'static_popularity' is IPS-BERT4Rec with real popularities, "
                         "'temporal_popularity' is ITPS-BERT4Rec with real temporal popularities.")
parser.add_argument('--propensity_clipping', type=float, default=0.2, help='Propensity clipping value used as lower '
                                                                           'threshold to avoid numerical precision '
                                                                           'issues when propensity values are too '
                                                                           'small.')
# TF #
parser.add_argument('--tf_hidden_units', type=int, default=100, help='Number of hidden units in the Tensor Factorization model.')
parser.add_argument('--tf_target', type=str, default='relevance', choices=['exposure', 'relevance'], help='Target to be modeled by the Tensor Factorization model.')
parser.add_argument('--frac_exposure_negatives', type=float, default=3.0, help='Fraction of sampled instances with negative exposure per number of positive instances. If None, all non-interactions will be considered as having negative exposure.')
parser.add_argument('--skewness_parameter', type=float, default=1.0, help='Power applied to the propensity scores to control the exposure bias through the skewness of the distribution. (Power p in the paper.)')
parser.add_argument('--unbiased_eval', type=lambda x: (str(x).lower() == 'true'), default=True, help='Set to True to enable: '
                                                                                                     '1) an unbiased semi-synthetic evaluation on true relevance if mode in ["train_bert_semi_synthetic", "tune_bert_semi_synthetic"]; or '
                                                                                                     '2) an unbiased evaluation on real data using ITPS-based evaluation metrics if mode in ["train_bert_real", "tune_bert_real", "loop_bert_real"].')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')

################
# Hyperparameter tuning
################
parser.add_argument('--num_configurations', type=int, default=20, help='Number of random hyperparameter configurations.')
parser.add_argument('--num_reps', type=int, default=2, help='Number of replicates in hyperparameter tuning.')
parser.add_argument('--tune_bert_hidden_units', type=str, default='[64, 128, 256]', help='Tuning values for bert_hidden_units.')
parser.add_argument('--tune_bert_num_blocks', type=str, default='[1, 2, 3]', help='Tuning values for bert_num_blocks.')
parser.add_argument('--tune_bert_num_heads', type=str, default='[1, 2, 4, 8]', help='Tuning values for bert_num_heads.')
parser.add_argument('--tune_train_batch_size', type=str, default='[64, 128, 256]', help='Tuning values for train_batch_size.')
parser.add_argument('--tune_bert_dropout', type=str, default='[0, 0.01, 0.1, 0.2]', help='Tuning values for bert_dropout.')
parser.add_argument('--tune_bert_mask_prob', type=str, default='[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]', help='Tuning values for bert_mask_prob.')
parser.add_argument('--tune_skew_power', type=str, default='[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]', help='Tuning values for skew_power.')
parser.add_argument('--tune_tf_hidden_units', type=str, default='[50, 100, 200]', help='Tuning values for tf_hidden_units.')

################
args = parser.parse_args()
set_template(args)
