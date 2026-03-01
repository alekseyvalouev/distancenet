"""
Generate classification diagnostic plots from a trained PairwiseDistanceNet checkpoint.

Generates:
  1. Confusion matrix (normalized + raw counts)
  2. Per-class accuracy bar chart
  3. Classification report (console + text file)
  4. Per-class confidence distribution (correct vs incorrect)

Usage:
  python make_classification_plots.py --checkpoint checkpoints/model_classification_9.pth
"""

import argparse
import os

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm

from models import PairwiseDistanceNet
from dataset import DistanceNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Classification diagnostic plots")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "et12-corner-1",
            "et07-cr-galgary",
            "et12-cr-hongkong",
            "et12-kitchen",
            "et07-office-114",
            "et07-office-419",
            "et07-office-420",
            "et07-office-423",
            "et07-office-424",
        ],
        help="Validation scenes",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[2, 4, 8], help="Horizon values")
    parser.add_argument("--negative_samples", action="store_true", default=True, help="Include negative samples")
    parser.add_argument("--n_classes", type=int, default=16, help="Number of output classes in the model")
    parser.add_argument("--valid_classes", nargs="+", type=int, default=[0, 1, 2, 3], help="Classes to keep for plotting (others are discarded)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    return parser.parse_args()


def build_class_labels(horizons, has_negatives):
    """Build human-readable class labels from horizon values."""
    labels = [f"{h}-step" for h in horizons]
    if has_negatives:
        labels.append("negative")
    return labels


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Run inference and collect all predictions, ground truths, and softmax probabilities."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for img_1, img_2, y in tqdm(dataloader, desc="Inference"):
        img_1, img_2 = img_1.to(device), img_2.to(device)
        y = y.long()

        logits = model(img_1, img_2)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = torch.argmax(logits, dim=1).cpu()

        all_preds.append(preds)
        all_targets.append(y)
        all_probs.append(probs)

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_targets).numpy(),
        torch.cat(all_probs).numpy(),
    )


def plot_confusion_matrix(y_true, y_pred, class_labels, valid_classes, output_dir):
    """Plot and save both raw-count and normalized confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    cm_raw = confusion_matrix(y_true, y_pred, labels=valid_classes)
    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=class_labels)
    disp_raw.plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("True", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)

    # Normalized (row-wise → recall per class)
    cm_norm = confusion_matrix(y_true, y_pred, labels=valid_classes, normalize="true")
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_labels)
    disp_norm.plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("True", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_class_accuracy(y_true, y_pred, class_labels, valid_classes, output_dir):
    """Plot a bar chart of per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(class_labels))
    bars = ax.bar(class_labels, per_class_acc, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    overall_acc = np.mean(y_true == y_pred)
    ax.axhline(y=overall_acc, color="red", linestyle="--", linewidth=1.5, label=f"Overall: {overall_acc:.1%}")
    ax.legend(fontsize=11)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path = os.path.join(output_dir, "per_class_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_classification_report(y_true, y_pred, class_labels, valid_classes, output_dir):
    """Print and save the sklearn classification report."""
    report = classification_report(y_true, y_pred, labels=valid_classes, target_names=class_labels, digits=4)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    path = os.path.join(output_dir, "classification_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"Saved: {path}")


def plot_confidence_distributions(y_true, y_pred, probs, class_labels, output_dir):
    """Plot histograms of predicted-class confidence for correct vs. incorrect predictions, per class."""
    n_classes = len(class_labels)
    cols = min(n_classes, 3)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, label in enumerate(class_labels):
        ax = axes[i]
        mask = y_true == i
        if mask.sum() == 0:
            ax.set_title(f"{label} (no samples)")
            continue

        # Confidence = softmax probability of the predicted class
        confidences = probs[mask, y_pred[mask]]
        correct = y_pred[mask] == y_true[mask]

        if correct.any():
            ax.hist(confidences[correct], bins=20, range=(0, 1), alpha=0.7, color="#2ecc71", label="Correct", edgecolor="black", linewidth=0.5)
        if (~correct).any():
            ax.hist(confidences[~correct], bins=20, range=(0, 1), alpha=0.7, color="#e74c3c", label="Incorrect", edgecolor="black", linewidth=0.5)

        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    # Hide unused axes
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confidence Distribution (Correct vs. Incorrect)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "confidence_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Determine classes ---
    has_negatives = args.negative_samples
    n_classes = args.n_classes
    valid_classes = set(args.valid_classes)
    class_labels = build_class_labels(args.horizons, has_negatives)

    print(f"Model output classes: {n_classes}")
    print(f"Valid classes for plots: {sorted(valid_classes)} → {class_labels}")
    print(f"Device: {device}")

    # --- Dataset & DataLoader ---
    transform = T.Compose([T.ToTensor(), T.Resize((224, 224))])
    dataset = DistanceNetDataset(
        scenes=args.scenes,
        horizons=args.horizons,
        negative_samples=has_negatives,
        transform=transform,
        classification=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Load model ---
    model = PairwiseDistanceNet(n_classes=n_classes)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Run inference ---
    print("Running inference...")
    y_pred, y_true, probs = collect_predictions(model, dataloader, device)
    print(f"Total samples before filtering: {len(y_true)}")

    # --- Filter to valid classes ---
    # Keep only samples whose ground-truth label is in valid_classes
    valid_mask = np.isin(y_true, list(valid_classes))
    n_discarded = (~valid_mask).sum()

    if n_discarded > 0:
        discarded_labels = y_true[~valid_mask]
        unique, counts = np.unique(discarded_labels, return_counts=True)
        print(f"\nDiscarded {n_discarded} samples with ground-truth class not in {sorted(valid_classes)}:")
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls}: {cnt} samples")
    else:
        print("\nNo samples discarded (all ground-truth labels are in valid classes).")

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    probs = probs[valid_mask]

    # Also report how many remaining predictions fall outside valid classes
    pred_outside = ~np.isin(y_pred, list(valid_classes))
    if pred_outside.any():
        outside_preds = y_pred[pred_outside]
        unique_p, counts_p = np.unique(outside_preds, return_counts=True)
        print(f"\n{pred_outside.sum()} out of {len(y_pred)} remaining samples were *predicted* outside valid classes:")
        for cls, cnt in zip(unique_p, counts_p):
            print(f"  Predicted class {cls}: {cnt} samples")

    print(f"\nSamples used for plots: {len(y_true)}")

    # --- Generate all plots ---
    valid_classes_list = sorted(valid_classes)
    plot_confusion_matrix(y_true, y_pred, class_labels, valid_classes_list, args.output_dir)
    plot_per_class_accuracy(y_true, y_pred, class_labels, valid_classes_list, args.output_dir)
    save_classification_report(y_true, y_pred, class_labels, valid_classes_list, args.output_dir)
    plot_confidence_distributions(y_true, y_pred, probs, class_labels, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
