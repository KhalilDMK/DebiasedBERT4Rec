from datasets import dataset_factory
from .bert import BertDataloader
from .tf import TFDataloader
from .ae import AEDataloader
import tqdm


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    TFDataloader.code(): TFDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args, export_root):
    dataset = dataset_factory(args, export_root)
    args.data_root = dataset.data_root
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    user_count, item_count = dataloader.user_count, dataloader.item_count
    train, val, test = dataloader.get_pytorch_dataloaders()
    position_distribution, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader = [], [], [], []
    if args.mode == 'train_semi_synthetic':
        temporal_propensity_dataloader = dataloader.get_temporal_propensity_dataloader()
        temporal_relevance_dataloader = dataloader.get_temporal_relevance_dataloader()
        static_propensity_dataloader = dataloader.get_static_propensity_dataloader()
        return train, val, test, temporal_propensity_dataloader, temporal_relevance_dataloader, static_propensity_dataloader, user_count, item_count
    elif args.dataloader_code != 'tf':
        position_distribution = dataloader.get_train_position_distributions_dataloader()
        train_popularity_vector_loader = dataloader.get_popularity_vector_dataloader(include_test=False)
        val_popularity_vector_loader = dataloader.get_popularity_vector_dataloader(include_test=True, mode='val')
        test_popularity_vector_loader = dataloader.get_popularity_vector_dataloader(include_test=True, mode='test')
        return train, val, test, position_distribution, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader, user_count, item_count
    elif args.mode == 'generate':
        gen = dataloader._get_gen_loader()
        return train, val, test, gen, position_distribution, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader, user_count, item_count
    #return train, val, test, position_distribution, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader, user_count, item_count
