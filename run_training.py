import random
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import wandb

from seed import set_seed

set_seed()

from models import PairwiseDistanceNet
from dataset import DistanceNetDataset
from trainer import RegressionDistanceNetTrainer, ClassificationDistanceNetTrainer

config = {
    "learning_rate": 0.0001,
    "architecture": "DINOv2 Concat LinearHead Regression",
    "dataset": "AdobeIndoorNav",
    "epochs": 100,
    "batch_size": 512,
}

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="valouev-university-of-california-berkeley",
    # Set the wandb project where this run will be logged.
    project="distancenet-v3",
    # Track hyperparameters and run metadata.
    config=config,
)

transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
])

train_scenes = [
    "et07-imagination-lab",
    "et12-corner-2",
    "et12-cr-helsinki",
    "et12-cr-honolulu",
    "et12-cr-hamburg",
    "et12-office-104",
    "et12-office-108",
    "et12-office-110",
    "et12-office-111",
    "et12-office-112",
    "et12-office-113",
    "et12-office-114",
    "et12-office-115",
    "et12-office-117",
    "et12-office-132",
]

test_scenes = [
    "et12-corner-1",
    "et07-cr-galgary",
    "et12-cr-hongkong",
    "et12-kitchen",
    "et07-office-114",
    "et07-office-419",
    "et07-office-420",
    "et07-office-423",
    "et07-office-424",
]

train_dataset = DistanceNetDataset(scenes=train_scenes, transform=transform, horizons=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], classification=False)
val_dataset = DistanceNetDataset(scenes=test_scenes, transform=transform, horizons=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], classification=False)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

model = PairwiseDistanceNet(n_classes=None)
model = model.to(torch.device("cuda"))
trainer = RegressionDistanceNetTrainer(config, train_dataset, model, train_loader, val_loader)

for epoch in range(config["epochs"]):
    if config["architecture"] == "DINOv2 Concat LinearHead Classification":
        train_loss, train_acc = trainer._train_one_epoch()
        val_loss, val_acc = trainer._evaluate()
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })
        torch.save(model.state_dict(), f"checkpoints/model_classification_{epoch}.pth")
    else:
        train_loss, train_per_class = trainer._train_one_epoch()
        val_loss, val_per_class = trainer._evaluate()
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_loss_class_{k}": v for k, v in train_per_class.items()},
            **{f"val_loss_class_{k}": v for k, v in val_per_class.items()},
        })
        torch.save(model.state_dict(), f"checkpoints/model_regression_{epoch}.pth")