import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_and_preprocess_data(
    path: str = "train_dataset.csv",
    features_scaler_path="features_scaler.pkl",
    price_scaler_path="sale_price_scaler.pkl",
    test_size=0.2,
):
    data = pd.read_csv(path)
    X = data.loc[:, :"SubwayStation_no_subway_nearby"]
    y = data[["PriceCategoryNum", "SalePrice"]]

    features_scaler = joblib.load(features_scaler_path)
    sale_price_scaler = joblib.load(price_scaler_path)

    features_to_scale = features_scaler.feature_names_in_
    X[features_to_scale] = features_scaler.transform(X[features_to_scale])
    y.loc[:, "SalePrice"] = sale_price_scaler.transform(y[["SalePrice"]]).flatten()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    return X_train, X_val, y_train, y_val


def create_dataset(X: pd.DataFrame, y: pd.Series) -> torch.utils.data.Dataset:
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.long)
    return torch.utils.data.TensorDataset(X_tensor, y_tensor)


def get_dataloaders(batch_size=64):
    X_train, X_val, y_train, y_val = load_and_preprocess_data()
    train_dataset = create_dataset(X_train, y_train["PriceCategoryNum"])
    val_dataset = create_dataset(X_val, y_val["PriceCategoryNum"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def get_class_weights(y_train: pd.DataFrame):
    y_numpy = y_train["PriceCategoryNum"].to_numpy()
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_numpy), y=y_numpy
    )
    return torch.tensor(class_weights, dtype=torch.float32)


def get_input_dim(X_train: pd.DataFrame):
    return X_train.shape[1]
