from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .ml_100k import ML100KDataset
from .amazon_beauty import AMAZONBEAUTY

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    ML100KDataset.code(): ML100KDataset,
    AMAZONBEAUTY.code(): AMAZONBEAUTY
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
