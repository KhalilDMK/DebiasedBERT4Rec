from datasets import dataset_factory
from .bert import BertDataloader
from .tf import TFDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    TFDataloader.code(): TFDataloader,
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    args.data_root = dataset.data_root
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()
    if args.mode in ['train_bert_semi_synthetic', 'tune_bert_semi_synthetic']:
        temporal_propensity, temporal_relevance, static_propensity = dataloader.get_semi_synthetic_properties()
        return train_loader, val_loader, test_loader, temporal_propensity, temporal_relevance, static_propensity
    elif args.mode in ['train_bert_real', 'tune_bert_real', 'loop_bert_real']:
        train_temporal_popularity, train_popularity_loader, val_popularity_loader, test_popularity_loader = dataloader.get_real_properties()
        return train_loader, val_loader, test_loader, train_temporal_popularity, train_popularity_loader, val_popularity_loader, test_popularity_loader
    elif args.mode == 'generate_semi_synthetic':
        gen_loader = dataloader.get_gen_loader()
        return train_loader, val_loader, test_loader, gen_loader
    else:
        return train_loader, val_loader, test_loader
