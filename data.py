from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import pickle

DATA_PATH = 'Data/'


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    return data


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

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

        self.cifar_train = None
        self.cifar_test = None

    def prepare_data(self):
        self.cifar_train = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=self.train_transform)
        self.cifar_test = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

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

        self.cifar_train = None
        self.cifar_test = None

    def prepare_data(self):
        self.cifar_train = datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=self.train_transform)
        self.cifar_test = datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
