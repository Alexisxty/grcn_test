from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import torch
import dgl
from dgl.data import CoraGraphDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import torch
from torch.utils.data import Dataset


# 修饰器
class CoraGraphDatasetWrapper(Dataset):
    def __init__(self, graph, mask):
        self.graph = graph
        self.mask = mask

    def __len__(self):
        return torch.sum(self.mask).item()

    def __getitem__(self, idx):
        return self.graph, self.mask


def collate(samples):
    graphs, masks = map(list, zip(*samples))
    return graphs[0], masks[0]

class CoraDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data/',
                 train_val_test_split: tuple[int, int, int] = (140, 500, 1000),
                 batch_size: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 num_features: int = 1433,
                 num_classes: int = 7
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.num_features = num_features

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 7 #Cora数据集有7个类别

    def prepare_data(self) -> None:
        CoraGraphDataset()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load Cora dataset and set up train, validation, and test splits"""
        cora_dataset = CoraGraphDataset()
        graph = cora_dataset[0]

        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        self.data_train = CoraGraphDatasetWrapper(graph, train_mask)
        self.data_val = CoraGraphDatasetWrapper(graph, val_mask)
        self.data_test = CoraGraphDatasetWrapper(graph, test_mask)


    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          collate_fn=collate)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

if __name__ == "__main__":
    #_ = CoraDataModule()

    ######################test#######################
    cora_data_module = CoraDataModule()
    cora_data_module.prepare_data()
    cora_data_module.setup(stage='fit')

    train_loader = cora_data_module.train_dataloader()
    val_loader = cora_data_module.val_dataloader()
    test_loader = cora_data_module.test_dataloader()

    for batch in train_loader:
        print("Train batch:", batch)
        break

    for batch in val_loader:
        print("Validation batch:", batch)
        break
    for batch in test_loader:
        print("Test batch:", batch)
        break