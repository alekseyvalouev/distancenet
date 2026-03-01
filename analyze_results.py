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
    # 1. Setup config and data loader
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

    print("Loading dataset...")
    val_dataset = DistanceNetDataset(scenes=test_scenes, transform=transform, horizons=[2, 4, 8], classification=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True) # Shuffle to get random examples

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairwiseDistanceNet(n_classes=4)
    
    # Find the latest checkpoint
    checkpoints = glob.glob("checkpoints/model_classification_*.pth")
    if not checkpoints:
        print("No checkpoints found. Please train the model first.")
        exit(1)
        
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Loading {latest_checkpoint}...")
    model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # 3. Collect 16 examples (4 true classes x 4 pred classes)
    examples = {}
    class_names = {0: "Dist 2", 1: "Dist 4", 2: "Dist 8", 3: "Diff Scene"}

    print("Collecting examples...")
    with torch.no_grad():
        for img_1, img_2, y in val_loader:
            img_1_dev, img_2_dev, y_dev = img_1.to(device), img_2.to(device), y.to(device)
            out = model(img_1_dev, img_2_dev)
            preds = torch.argmax(out, dim=1)
            
            for i in range(len(y_dev)):
                t = y_dev[i].item()
                p = preds[i].item()
                if (t, p) not in examples:
                    # Store original images as CPU tensors
                    examples[(t, p)] = (img_1[i].cpu(), img_2[i].cpu())
                    
            if len(examples) == 16:
                break

    # 4. Plot
    print("Plotting confusion matrix examples...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for true_c in range(4):
        for pred_c in range(4):
            ax = axes[true_c, pred_c]
            if (true_c, pred_c) in examples:
                im1, im2 = examples[(true_c, pred_c)]
                # Undo transform: to numpy (H, W, C)
                im1_np = im1.permute(1, 2, 0).numpy()
                im2_np = im2.permute(1, 2, 0).numpy()
                
                # Clip values to [0, 1] just in case
                im1_np = np.clip(im1_np, 0, 1)
                im2_np = np.clip(im2_np, 0, 1)
                
                # Concatenate horizontally
                combined = np.concatenate((im1_np, im2_np), axis=1)
                ax.imshow(combined)
            else:
                ax.text(0.5, 0.5, "No Example Found", ha="center", va="center", fontsize=12)
            
            ax.set_title(f"True: {class_names[true_c]} | Pred: {class_names[pred_c]}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("confusion_matrix_examples.png", bbox_inches='tight', dpi=150)
    print("Saved confusion matrix examples to confusion_matrix_examples.png")

if __name__ == "__main__":
    main()