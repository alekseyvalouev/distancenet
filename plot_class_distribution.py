"""
Bar plot to visualize the class (horizon/distance) distribution
of a DistanceNetDataset.
"""
import matplotlib.pyplot as plt
from collections import Counter
from dataset import DistanceNetDataset
from torchvision import transforms as T


def plot_class_distribution(dataset):
    # Each entry in dataset.data is (scene_or_scenes, horizon, pair)
    horizon_values = [sample[1] for sample in dataset.data]
    counts = Counter(horizon_values)

    # Sort: negatives (-1) last
    sorted_horizons = sorted(counts.keys(), key=lambda h: (h == -1, h))
    sorted_counts = [counts[h] for h in sorted_horizons]
    labels = ["Negative" if h == -1 else f"Distance {h}" for h in sorted_horizons]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#DA8BC3']
    bars = ax.bar(labels, sorted_counts, color=colors[:len(labels)], edgecolor='white', linewidth=1.2)

    for bar, count in zip(bars, sorted_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sorted_counts) * 0.02,
                str(count), ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Number of Samples', fontsize=13)
    ax.set_title('Dataset Class Distribution', fontsize=15, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    print("Saved plot to class_distribution.png")
    plt.show()

if __name__ == "__main__":
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


    train_dataset = DistanceNetDataset(scenes=train_scenes, transform=transform, horizons=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], classification=False)
    plot_class_distribution(train_dataset)
