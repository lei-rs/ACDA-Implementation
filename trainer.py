from lightning.pytorch.callbacks import ModelCheckpoint
from model import *
from data import *
import lightning.pytorch as pl
import argparse
import os


class Model(pl.LightningModule):
    def __init__(self, cifar='10', ad=False):
        super().__init__()
        self.save_hyperparameters()

        if cifar == '10':
            num_classes = 10
        elif cifar == '100':
            num_classes = 100

        if ad:
            self.model = ADLeNet(num_classes=num_classes)
        else:
            self.model = LeNet(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar', type=str, default='10', help='CIFAR-10 or CIFAR-100')
    parser.add_argument('--ad', type=bool, default=False, help='Use ADLeNet or LeNet')
    parser.add_argument('--cont', type=int, default=None, help='Continue training from checkpoint')
    parser.add_argument('--test', type=bool, default=False, help='Test model')
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=50, callbacks=[checkpoint_callback])

    if args.cifar == '10':
        dm = CIFAR10DataModule()
    elif args.cifar == '100':
        dm = CIFAR100DataModule()

    if args.cont is not None:
        file = os.listdir(f'lightning_logs/version_{args.cont}/checkpoints')[-1]
        model = Model.load_from_checkpoint(f'lightning_logs/version_{args.cont}/checkpoints/{file}')
    else:
        model = Model(cifar=args.cifar, ad=args.ad)

    trainer.fit(model, datamodule=dm)
    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=dm)
