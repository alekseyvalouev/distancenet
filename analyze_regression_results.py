import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from models import PairwiseDistanceNet
from dataset import DistanceNetDataset

def main():
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
    ])

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

    horizons = list(range(2, 17))

    print("Loading dataset...")
    val_dataset = DistanceNetDataset(
        scenes=test_scenes, transform=transform,
        horizons=horizons, classification=False, negative_samples=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairwiseDistanceNet(n_classes=None)

    checkpoints = glob.glob("checkpoints/model_regression_*.pth")
    if not checkpoints:
        print("No regression checkpoints found. Please train the model first.")
        exit(1)

    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Loading {latest_checkpoint}...")
    model.load_state_dict(torch.load("checkpoints/model_regression_5.pth", map_location=device))
    model.to(device)
    model.eval()

    # For each true label, track the best (smallest error) and worst (largest error) example
    best = {}   # label -> (error, pred, img1, img2)
    worst = {}  # label -> (error, pred, img1, img2)

    print("Collecting examples...")
    with torch.no_grad():
        for img_1, img_2, y in val_loader:
            img_1_dev, img_2_dev = img_1.to(device), img_2.to(device)
            preds = model(img_1_dev, img_2_dev).squeeze(1)

            for i in range(len(y)):
                true_label = y[i].item()
                pred_val = preds[i].item()
                error = abs(pred_val - true_label)

                if true_label not in best or error < best[true_label][0]:
                    best[true_label] = (error, pred_val, img_1[i].cpu(), img_2[i].cpu())
                if true_label not in worst or error > worst[true_label][0]:
                    worst[true_label] = (error, pred_val, img_1[i].cpu(), img_2[i].cpu())

            if len(best) == len(horizons) and len(worst) == len(horizons):
                # Keep scanning for better/worse examples — only stop after full pass
                pass

    n_labels = len(horizons)
    fig, axes = plt.subplots(n_labels, 2, figsize=(10, 3 * n_labels))

    for row, label in enumerate(horizons):
        for col, (store, title_prefix) in enumerate([(best, "Best"), (worst, "Worst")]):
            ax = axes[row, col]
            if label in store:
                error, pred_val, im1, im2 = store[label]
                im1_np = np.clip(im1.permute(1, 2, 0).numpy(), 0, 1)
                im2_np = np.clip(im2.permute(1, 2, 0).numpy(), 0, 1)
                combined = np.concatenate((im1_np, im2_np), axis=1)
                ax.imshow(combined)
                ax.set_title(f"{title_prefix} | True: {label} | Pred: {pred_val:.2f} | Err: {error:.2f}")
            else:
                ax.text(0.5, 0.5, "No Example Found", ha="center", va="center", fontsize=12)
                ax.set_title(f"{title_prefix} | True: {label}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("regression_examples.png", bbox_inches='tight', dpi=150)
    print("Saved to regression_examples.png")

if __name__ == "__main__":
    main()
