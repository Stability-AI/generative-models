from typing import Optional

import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
import torchvision.transforms as transforms


class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.train_config = train
        assert (
            "datapipeline" in self.train_config and "loader" in self.train_config
        ), "train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using the one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
            ), "test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("USING DUMMY DATASET: HOPE YOU'RE DEBUGGING ;)")
            print("#" * 100)

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset

        self.train_datapipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
        loader = create_loader(self.train_datapipeline, **self.train_config.loader)
        return loader

    def val_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.val_datapipeline, **self.val_config.loader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.test_datapipeline, **self.test_config.loader)


def create_dataset(**kwargs):
    # Implement your dataset creation logic here
    pass


def create_dummy_dataset(**kwargs):
    # Implement your dummy dataset creation logic here
    pass


def create_loader(datapipeline, **loader_kwargs):
    # Implement your loader creation logic here
    pass


# Usage example
train_config = {
    "datapipeline": {
        # Specify the data pipeline configuration
    },
    "loader": {
        # Specify the loader configuration
    }
}

val_config = {
    "datapipeline": {
        # Specify the validation data pipeline configuration
    },
    "loader": {
        # Specify the validation loader configuration
    }
}

test_config = {
    "datapipeline": {
        # Specify the test data pipeline configuration
    },
    "loader": {
        # Specify the test loader configuration
    }
}

data_module = StableDataModuleFromConfig(
    train=train_config,
    validation=val_config,
    test=test_config,
    skip_val_loader=False,
    dummy=False,
)
data_module.prepare_data()
data_module.setup()

train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()

# Use the dataloaders for training, validation, and testing
# ...
