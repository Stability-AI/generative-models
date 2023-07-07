import torchvision
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class MNISTDataDictWrapper(Dataset):
    def __init__(self, dset):
        super().__init__()
        self.dset = dset

    def __getitem__(self, i):
        x, y = self.dset[i]
        return {"jpg": x, "cls": y}

    def __len__(self):
        return len(self.dset)


class MNISTLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers=0,
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
    ):
        super().__init__()

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else 0
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.train_dataset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/", train=True, download=True, transform=transform
            )
        )
        self.test_dataset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/", train=False, download=True, transform=transform
            )
        )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )


if __name__ == "__main__":
    mnist_loader = MNISTLoader(batch_size=64, num_workers=4, prefetch_factor=2)
    mnist_loader.prepare_data()

    train_dataloader = mnist_loader.train_dataloader()
    test_dataloader = mnist_loader.test_dataloader()
    val_dataloader = mnist_loader.val_dataloader()

    # Example usage
    for batch in train_dataloader:
        # Training iteration
        pass

    for batch in val_dataloader:
        # Validation iteration
        pass

    for batch in test_dataloader:
        # Testing iteration
        pass
