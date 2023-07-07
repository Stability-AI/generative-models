import torchvision
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split


class CIFAR10DataDictWrapper(Dataset):
    def __init__(self, dset):
        super().__init__()
        self.dset = dset

    def __getitem__(self, i):
        x, y = self.dset[i]
        return {"jpg": x, "cls": y}

    def __len__(self):
        return len(self.dset)


class CIFAR10Loader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, shuffle=True, data_augmentation=True):
        super().__init__()

        # Define transformations
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        if data_augmentation:
            transform_list.insert(0, transforms.RandomHorizontalFlip())
            transform_list.insert(0, transforms.RandomCrop(32, padding=4))

        transform = transforms.Compose(transform_list)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation

        self.dataset = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=True,
            transform=transform,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit':
            # Split the dataset into training and validation sets
            train_length = int(len(self.dataset) * 0.8)
            val_length = len(self.dataset) - train_length
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_length, val_length]
            )

        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=".data/",
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ])
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )


# Usage example
data_module = CIFAR10Loader(batch_size=64, num_workers=4, shuffle=True, data_augmentation=True)
data_module.prepare_data()
data_module.setup()

train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()

# Use the dataloaders for training, validation, and testing
# ...