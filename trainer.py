from model import *
from data import *
import lightning.pytorch as pl


class Model(pl.LightningModule):
    def __init__(self, cifar='10', ad=False):
        super().__init__()
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
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=25)
    dm = CIFAR10DataModule()
    model = Model(cifar='10', ad=True)
    trainer.fit(model, dm)
