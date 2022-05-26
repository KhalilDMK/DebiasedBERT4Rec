def set_template(args):
    if args.template is None:
        return

    # Train BERT4Rec on semi-synthetic data and evaluate it with the unbiased evaluation process.

    elif args.template.startswith('train_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 16
        args.bert_num_blocks = 1
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.15
        args.bert_dropout = 0.4

        args.loss_debiasing = None
        args.unbiased_eval = True

    # Train IPS-BERT4Rec on semi-synthetic data and evaluate it with the unbiased evaluation process.

    elif args.template.startswith('train_ips_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0.4

        args.loss_debiasing = 'static_propensity'
        args.unbiased_eval = True

    # Train ITPS-BERT4Rec on semi-synthetic data and evaluate it with the unbiased evaluation process.

    elif args.template.startswith('train_itps_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0.4

        args.loss_debiasing = 'temporal_propensity'
        args.unbiased_eval = True

    # Train the Oracle on semi-synthetic data and evaluate it with the unbiased evaluation process.

    elif args.template.startswith('train_oracle_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 8
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0

        args.loss_debiasing = 'relevance'
        args.unbiased_eval = True

    # Train BERT4Rec on semi-synthetic data with a skewed exposure distribution, and evaluate it with the unbiased
    # evaluation process. The exposure skewness power is input by the user.

    elif args.template.startswith('robustness_increasing_bias_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 16
        args.bert_num_blocks = 1
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.15
        args.bert_dropout = 0.4

        args.loss_debiasing = None
        args.unbiased_eval = True

        args.skewness_parameter = float(input('Input the value of the power p (power applied to the propensity scores '
                                              'to control the exposure bias through the skewness of the distribution.'
                                              '): '))

    # Train IPS-BERT4Rec on semi-synthetic data with a skewed exposure distribution, and evaluate it with the unbiased
    # evaluation process. The exposure skewness power is input by the user.

    elif args.template.startswith('robustness_increasing_bias_ips_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0.4

        args.loss_debiasing = 'static_propensity'
        args.unbiased_eval = True

        args.skewness_parameter = float(input('Input the value of the power p (power applied to the propensity scores '
                                              'to control the exposure bias through the skewness of the distribution.'
                                              '): '))

    # Train ITPS-BERT4Rec on semi-synthetic data with a skewed exposure distribution, and evaluate it with the unbiased
    # evaluation process. The exposure skewness power is input by the user.

    elif args.template.startswith('robustness_increasing_bias_itps_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0.4

        args.loss_debiasing = 'temporal_propensity'
        args.unbiased_eval = True

        args.skewness_parameter = float(input('Input the value of the power p (power applied to the propensity scores '
                                              'to control the exposure bias through the skewness of the distribution.'
                                              '): '))

    # Train the Oracle on semi-synthetic data with a skewed exposure distribution, and evaluate it with the unbiased
    # evaluation process. The exposure skewness power is input by the user.

    elif args.template.startswith('robustness_increasing_bias_oracle_semi_synthetic_unbiased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 8
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0

        args.loss_debiasing = 'relevance'
        args.unbiased_eval = True

        args.skewness_parameter = float(input('Input the value of the power p (power applied to the propensity scores '
                                              'to control the exposure bias through the skewness of the distribution.'
                                              '): '))

    # Train BERT4Rec on semi-synthetic data and evaluate it with the standard LOO evaluation process.

    elif args.template.startswith('train_bert_semi_synthetic_biased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 16
        args.bert_num_blocks = 1
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.15
        args.bert_dropout = 0.4

        args.loss_debiasing = None
        args.unbiased_eval = False

    # Train IPS-BERT4Rec on semi-synthetic data and evaluate it with the standard LOO evaluation process.

    elif args.template.startswith('train_ips_bert_semi_synthetic_biased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0.4

        args.loss_debiasing = 'static_propensity'
        args.unbiased_eval = False

    # Train ITPS-BERT4Rec on semi-synthetic data and evaluate it with the standard LOO evaluation process.

    elif args.template.startswith('train_itps_bert_semi_synthetic_biased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0.4

        args.loss_debiasing = 'temporal_propensity'
        args.unbiased_eval = False

    # Train the Oracle on semi-synthetic data and evaluate it with the standard LOO evaluation process.

    elif args.template.startswith('train_oracle_semi_synthetic_biased_eval'):
        args.mode = 'train_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.train_batch_size = 8
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0

        args.loss_debiasing = 'relevance'
        args.unbiased_eval = False

    # Tune BERT4Rec on semi-synthetic data. The hyperparameter tuning is performed on a validation set using the
    # unbiased evaluation process.

    elif args.template.startswith('tune_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'tune_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = None
        args.unbiased_eval = True

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'

    # Tune IPS-BERT4Rec on semi-synthetic data. The hyperparameter tuning is performed on a validation set using the
    # unbiased evaluation process.

    elif args.template.startswith('tune_ips_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'tune_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = 'static_propensity'
        args.unbiased_eval = True

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'

    # Tune ITPS-BERT4Rec on semi-synthetic data. The hyperparameter tuning is performed on a validation set using the
    # unbiased evaluation process.

    elif args.template.startswith('tune_itps_bert_semi_synthetic_unbiased_eval'):
        args.mode = 'tune_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = 'temporal_propensity'
        args.unbiased_eval = True

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'

    # Tune the Oracle on semi-synthetic data. The hyperparameter tuning is performed on a validation set using the
    # unbiased evaluation process.

    elif args.template.startswith('tune_oracle_semi_synthetic_unbiased_eval'):
        args.mode = 'tune_bert_semi_synthetic'

        args.dataset_code = 'ml-100k'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = 'relevance'
        args.unbiased_eval = True

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'

    # Generate semi-synthetic interactions using Tensor Factorization.

    elif args.template.startswith('generate_semi_synthetic_data_interactions'):
        args.mode = 'generate_semi_synthetic'

        args.dataset_code = 'ml-100k'
        args.generate_semi_synthetic_seed = 1

        args.train_batch_size = 256
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100

        args.bert_max_len = 100

        args.tf_hidden_units = 50
        args.tf_target = 'exposure'
        args.frac_exposure_negatives = 3.0

    # Generate semi-synthetic ratings using Tensor Factorization.

    elif args.template.startswith('generate_semi_synthetic_data_ratings'):
        args.mode = 'generate_semi_synthetic'

        args.dataset_code = 'ml-100k'
        args.generate_semi_synthetic_seed = 1

        args.train_batch_size = 256
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100

        args.bert_max_len = 100

        args.tf_hidden_units = 50
        args.tf_target = 'relevance'

    # Tune the Tensor Factorization model for the task of interaction prediction.
    # The hyperparameter tuning is performed on a validation set.

    elif args.template.startswith('tune_tf_semi_synthetic_interactions'):
        args.mode = 'tune_tf'

        args.dataset_code = 'ml-100k'
        args.generate_semi_synthetic_seed = 1

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100

        args.bert_max_len = 100

        args.tf_target = 'exposure'
        args.frac_exposure_negatives = 3.0

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_tf_hidden_units = '[50, 100, 200]'

    # Tune the Tensor Factorization model for the task of rating prediction.
    # The hyperparameter tuning is performed on a validation set.

    elif args.template.startswith('tune_tf_semi_synthetic_ratings'):
        args.mode = 'tune_tf'

        args.dataset_code = 'ml-100k'
        args.generate_semi_synthetic_seed = 1

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100

        args.bert_max_len = 100

        args.tf_target = 'relevance'

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_tf_hidden_units = '[50, 100, 200]'

    # Train BERT4Rec on the Movielens 1M dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'ml-1m'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 1
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.15
        args.bert_dropout = 0.2
        args.skew_power = 0.001

        args.loss_debiasing = None
        args.unbiased_eval = False

    # Train IPS-BERT4Rec on the Movielens 1M dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_ips_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'ml-1m'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0
        args.skew_power = 0.5

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

    # Train ITPS-BERT4Rec on the Movielens 1M dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_itps_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'ml-1m'

        args.train_batch_size = 8
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 1
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0.1
        args.skew_power = 0.2

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

    # Train BERT4Rec on the Movielens 20M dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'ml-20m'

        args.train_batch_size = 256
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 256
        args.bert_num_blocks = 3
        args.bert_num_heads = 4
        args.bert_mask_prob = 0.6
        args.bert_dropout = 0
        args.skew_power = 0.2

        args.loss_debiasing = None
        args.unbiased_eval = False

    # Train IPS-BERT4Rec on the Movielens 20M dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_ips_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'ml-20m'

        args.train_batch_size = 256
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 256
        args.bert_num_blocks = 2
        args.bert_num_heads = 4
        args.bert_mask_prob = 0.5
        args.bert_dropout = 0.1
        args.skew_power = 0.01

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

    # Train ITPS-BERT4Rec on the Movielens 20M dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_itps_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'ml-20m'

        args.train_batch_size = 128
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 256
        args.bert_num_blocks = 3
        args.bert_num_heads = 8
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0.01
        args.skew_power = 0.2

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

    # Train BERT4Rec on the Amazon Beauty dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10
        args.bert_hidden_units = 16
        args.bert_num_blocks = 1
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0
        args.skew_power = 0.01

        args.loss_debiasing = None
        args.unbiased_eval = False

    # Train IPS-BERT4Rec on the Amazon Beauty dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_ips_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.2
        args.bert_dropout = 0.1
        args.skew_power = 0.01

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

    # Train ITPS-BERT4Rec on the Amazon Beauty dataset and evaluate it on the test set where the negative samples are
    # sampled based on their popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('train_itps_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'train_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.6
        args.bert_dropout = 0.1
        args.skew_power = 0.2

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

    # Tune BERT4Rec on the Movielens 1M dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'ml-1m'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = None
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune IPS-BERT4Rec on the Movielens 1M dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_ips_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'ml-1m'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune ITPS-BERT4Rec on the Movielens 1M dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_itps_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'ml-1m'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune BERT4Rec on the Movielens 20M dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'ml-20m'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = None
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[64, 128, 256]'
        args.tune_bert_num_blocks = '[1, 2, 3]'
        args.tune_bert_num_heads = '[1, 2, 4, 8]'
        args.tune_train_batch_size = '[64, 128, 256]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0, 0.01, 0.1, 0.2]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune IPS-BERT4Rec on the Movielens 20M dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_ips_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'ml-20m'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[64, 128, 256]'
        args.tune_bert_num_blocks = '[1, 2, 3]'
        args.tune_bert_num_heads = '[1, 2, 4, 8]'
        args.tune_train_batch_size = '[64, 128, 256]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0, 0.01, 0.1, 0.2]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune ITPS-BERT4Rec on the Movielens 20M dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_itps_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'ml-20m'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[64, 128, 256]'
        args.tune_bert_num_blocks = '[1, 2, 3]'
        args.tune_bert_num_heads = '[1, 2, 4, 8]'
        args.tune_train_batch_size = '[64, 128, 256]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0, 0.01, 0.1, 0.2]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune BERT4Rec on the Amazon Beauty dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10

        args.loss_debiasing = None
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune IPS-BERT4Rec on the Amazon Beauty dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_ips_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Tune ITPS-BERT4Rec on the Amazon Beauty dataset. The tuning is performed on a validation set in which the
    # negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation.

    elif args.template.startswith('tune_itps_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'tune_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

        args.num_configurations = 20
        args.num_reps = 5
        args.tune_bert_hidden_units = '[8, 16, 32, 64]'
        args.tune_bert_num_blocks = '[1, 2]'
        args.tune_bert_num_heads = '[1, 2]'
        args.tune_train_batch_size = '[8, 16, 32]'
        args.tune_bert_dropout = '[0, 0.1, 0.2, 0.4]'
        args.tune_bert_mask_prob = '[0.1, 0.15, 0.2, 0.4, 0.6]'
        args.tune_skew_power = '[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]'

    # Simulate a feedback loop with recommendations from BERT4Rec on the Movielens 1M dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'ml-1m'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 1
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.15
        args.bert_dropout = 0.2
        args.skew_power = 0.001

        args.loss_debiasing = None
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from IPS-BERT4Rec on the Movielens 1M dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_ips_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'ml-1m'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.1
        args.bert_dropout = 0
        args.skew_power = 0.5

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from ITPS-BERT4Rec on the Movielens 1M dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_itps_bert_real_ml_1m_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'ml-1m'

        args.train_batch_size = 8
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 64
        args.bert_num_blocks = 1
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0.1
        args.skew_power = 0.2

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from BERT4Rec on the Movielens 20M dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'ml-20m'

        args.train_batch_size = 256
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 256
        args.bert_num_blocks = 3
        args.bert_num_heads = 4
        args.bert_mask_prob = 0.6
        args.bert_dropout = 0
        args.skew_power = 0.2

        args.loss_debiasing = None
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from IPS-BERT4Rec on the Movielens 20M dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_ips_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'ml-20m'

        args.train_batch_size = 256
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 256
        args.bert_num_blocks = 2
        args.bert_num_heads = 4
        args.bert_mask_prob = 0.5
        args.bert_dropout = 0.1
        args.skew_power = 0.01

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from ITPS-BERT4Rec on the Movielens 20M dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_itps_bert_real_ml_20m_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'ml-20m'

        args.train_batch_size = 128
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 100
        args.bert_hidden_units = 256
        args.bert_num_blocks = 3
        args.bert_num_heads = 8
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0.01
        args.skew_power = 0.2

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from BERT4Rec on the Amazon Beauty dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.train_batch_size = 16
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10
        args.bert_hidden_units = 16
        args.bert_num_blocks = 1
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.4
        args.bert_dropout = 0
        args.skew_power = 0.01

        args.loss_debiasing = None
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from IPS-BERT4Rec on the Amazon Beauty dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_ips_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.2
        args.bert_dropout = 0.1
        args.skew_power = 0.01

        args.loss_debiasing = 'static_popularity'
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10

    # Simulate a feedback loop with recommendations from ITPS-BERT4Rec on the Amazon Beauty dataset.
    # The ranking evaluation is based on popularity negative sampling.

    elif args.template.startswith('loop_itps_bert_real_amazon_beauty_pop_eval_sampling'):
        args.mode = 'loop_bert_real'

        args.dataset_code = 'amazon-beauty'

        args.train_batch_size = 32
        args.val_batch_size = 500
        args.test_batch_size = 500

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 50
        args.gamma = 0.1
        args.num_epochs = 100
        args.metric_ks = [5, 10]
        args.best_metric = 'NDCG@10'

        args.bert_max_len = 10
        args.bert_hidden_units = 64
        args.bert_num_blocks = 2
        args.bert_num_heads = 1
        args.bert_mask_prob = 0.6
        args.bert_dropout = 0.1
        args.skew_power = 0.2

        args.loss_debiasing = 'temporal_popularity'
        args.unbiased_eval = False

        args.num_iterations = 10
        args.top_k_recom = 10
