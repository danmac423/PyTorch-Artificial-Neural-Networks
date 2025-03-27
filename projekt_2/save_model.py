from model import LitPriceClassifier
import torch

CHECKPOINT_PATH = "price-classification/trnphmyp/checkpoints/best-model.ckpt"


model = LitPriceClassifier.load_from_checkpoint(CHECKPOINT_PATH)
torch.save(model.classifier, "best.pt")
