import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy


class PriceClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        out_faetures: int,
        activation: str = "relu",
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        act_layer = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
        }[activation]

        layers = []
        input_size = in_features
        for h in hidden_units:
            layers.append(nn.Linear(input_size, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer)
            layers.append(nn.Dropout(dropout))

            input_size = h

        layers.append(nn.Linear(input_size, out_faetures))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LitPriceClassifier(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        out_features: int,
        class_weights_tensor=None,
        lr=1e-3,
        optimizer_name="adam",
        activation="relu",
        dropout=0.3,
        use_batchnorm=True,
        momentum=0.9,
        weight_decay=0.01,
        nesterov=False,
    ):
        super().__init__()

        self.classifier = PriceClassifier(
            in_features,
            hidden_units,
            out_features,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        self.train_acc = Accuracy(task="multiclass", num_classes=out_features)
        self.val_acc = Accuracy(task="multiclass", num_classes=out_features)
        self.test_acc = Accuracy(task="multiclass", num_classes=out_features)
        self.train_avg_acc = Accuracy(
            task="multiclass", num_classes=out_features, average="macro"
        )
        self.val_avg_acc = Accuracy(
            task="multiclass", num_classes=out_features, average="macro"
        )
        self.test_avg_acc = Accuracy(
            task="multiclass", num_classes=out_features, average="macro"
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        target = target.type(torch.long)
        logits = self(x)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, target)
        self.train_avg_acc(preds, target)

        self.log("train/loss", loss, prog_bar=True)
        self.log(
            "train/accuracy",
            self.train_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/average_accuracy",
            self.train_avg_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        target = target.type(torch.long)
        logits = self(x)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, target)
        self.val_avg_acc.update(preds, target)

        self.log("val/loss", loss, prog_bar=True)
        self.log(
            "val/accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/average_accuracy",
            self.val_avg_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        name = self.optimizer_name.lower()
        if name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        elif name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return optimizer
