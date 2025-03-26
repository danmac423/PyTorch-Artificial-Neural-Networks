import wandb
from train import train_model

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/average_accuracy", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
        "optimizer_name": {"values": ["adamw", "adam", "sgd", "rmsprop"]},
        "momentum": {"min": 0.5, "max": 0.99},
        "weight_decay": {"min": 0.0, "max": 0.05},
        "dropout": {"min": 0.1, "max": 0.5},
        "activation": {"values": ["relu", "leaky_relu", "gelu"]},
        "use_batchnorm": {"values": [True, False]},
        "use_weights": {"values": [True, False]},
        "hidden_units": {
            "values": [
                [512],
                [256],
                [128],
                [64],
                [512, 256],
                [256, 128],
                [128, 64],
                [512, 256, 128],
                [256, 128, 64],
                [128, 64, 32],
            ]
        },
    },
}


sweep_id = wandb.sweep(sweep_config, project="price-classification")
wandb.agent(sweep_id, function=train_model, count=1000)
