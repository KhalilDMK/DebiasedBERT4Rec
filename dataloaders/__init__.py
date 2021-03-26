from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader
import tqdm


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args, export_root):
    dataset = dataset_factory(args, export_root)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    position_distribution = dataloader.get_train_position_distributions_dataloader()
    train_popularity_vector_loader = dataloader.get_popularity_vector_dataloader(include_test=False)
    val_popularity_vector_loader = dataloader.get_popularity_vector_dataloader(include_test=True, mode='val')
    test_popularity_vector_loader = dataloader.get_popularity_vector_dataloader(include_test=True, mode='test')
    return train, val, test, position_distribution, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader
