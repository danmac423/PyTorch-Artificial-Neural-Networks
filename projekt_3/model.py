import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        y: torch.Tensor
        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
    ):
        super().__init__()

        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.has_se = se_ratio is not None and se_ratio > 0
        expanded_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(),
            )
        else:
            self.expand_conv = nn.Identity()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(),
        )

        if self.has_se:
            self.se = SEBlock(expanded_channels, int(1 / se_ratio))

        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand_conv(x)
        y = self.depthwise_conv(y)

        if self.has_se:
            y = self.se(y)

        y = self.project_conv(y)

        if self.use_residual:
            y = x + y

        return y


class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, kernel_size=3, stride=1),
            MBConvBlock(16, 24, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(24, 40, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock(40, 80, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(80, 112, expand_ratio=6, kernel_size=5, stride=2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(112, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x


class LitConvNet(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.lr = lr
        self.wd = weight_decay
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.train_acc(preds, y)

        self.log("train/loss", loss, prog_bar=True)
        self.log(
            "train/acc",
            self.train_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.val_acc.update(preds, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="max", patience=2, factor=0.5
            ),
            "monitor": "val/acc",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
