from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning.pytorch as pl
import numpy as np


DATA_PATH = 'Data/'


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=8, val_frac=0.3):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.train = None
        self.train_sampler = None
        self.val = None
        self.valid_sampler = None
        self.test = None

    def prepare_data(self):
        train = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=self.train_transform)
        self.test = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=self.test_transform)
        val_amount = int(len(train) * self.val_frac)
        self.train, self.val = random_split(train, [len(train) - val_amount, val_amount])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=8, val_frac=0.3):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.train = None
        self.train_sampler = None
        self.val = None
        self.valid_sampler = None
        self.test = None

    def prepare_data(self):
        train = datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=self.train_transform)
        self.test = datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=self.test_transform)
        val_amount = int(len(train) * self.val_frac)
        self.train, self.val = random_split(train, [len(train) - val_amount, val_amount])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=self.valid_sampler)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
