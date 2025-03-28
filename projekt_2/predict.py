import pandas as pd
import torch
import joblib
from model import PriceClassifier
from sklearn.preprocessing import StandardScaler


TEST_CSV = "test_dataset.csv"
SCALER_PATH = "features_scaler.pkl"
OUPUT_CSV = "pred.csv"

torch.serialization.add_safe_globals([PriceClassifier])


X = pd.read_csv(TEST_CSV)

scaler: StandardScaler = joblib.load(SCALER_PATH)
features_to_scale = scaler.feature_names_in_
X[features_to_scale] = scaler.transform(X[features_to_scale])

X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float)

model = torch.load("best.pt", weights_only=False)

model.eval()
with torch.no_grad():
    logits = model(X_tensor)
    preds = torch.argmax(logits, dim=1).numpy()

assert len(preds) == len(X)

pd.Series(preds).to_csv(OUPUT_CSV, index=False, header=False)
