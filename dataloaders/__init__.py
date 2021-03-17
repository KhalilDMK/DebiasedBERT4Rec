from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader
import tqdm


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    position_distribution = dataloader.get_train_position_distributions_dataloader()
    return train, val, test, position_distribution
