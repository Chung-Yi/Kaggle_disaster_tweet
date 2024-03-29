from .base_dataloader import BaseDataLoader
from costom_dataset import DisasterDataset, DisasterTweetDataset

class DisatserDataLoader(BaseDataLoader):
    def __init__(self, config):

        dataset = DisasterTweetDataset(config, 'train')
        super().__init__(dataset, int(config['data_loader']['batch_size']),float(config['data_loader']["split_weight"]), int(config['data_loader']['num_workers']))

        