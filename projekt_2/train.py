import wandb
from data import (
    get_dataloaders,
    load_and_preprocess_data,
    get_class_weights,
    get_input_dim,
)
from model import LitPriceClassifier
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer


def train_model():
    with wandb.init():
        config = wandb.config
        X_train, X_val, y_train, y_val = load_and_preprocess_data()
        in_features = get_input_dim(X_train)
        out_features = 3
        class_weights_tensor = (
            get_class_weights(y_train) if config.use_weights else None
        )

        train_loader, val_loader = get_dataloaders(batch_size=64)

        model = LitPriceClassifier(
            in_features=in_features,
            hidden_units=config.hidden_units,
            out_features=out_features,
            class_weights_tensor=class_weights_tensor,
            lr=config.lr,
            optimizer_name=config.optimizer_name,
            activation=config.activation,
            dropout=config.dropout,
            use_batchnorm=config.use_batchnorm,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        wandb_logger = WandbLogger(experiment=wandb.run)
        checkpoint_callback = ModelCheckpoint(
            monitor="val/average_accuracy",
            mode="max",
            save_top_k=1,
            filename="best-model",
        )

        trainer = Trainer(
            max_epochs=10,
            logger=wandb_logger,
            deterministic=True,
            callbacks=[checkpoint_callback],
            accelerator="cpu",
        )

        trainer.fit(model, train_loader, val_loader)
