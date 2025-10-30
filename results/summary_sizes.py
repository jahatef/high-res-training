import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np


# -------------------------------
# Visualization helpers
# -------------------------------

def plot_size_scatter(ax, datasets, colors, width_max, height_max):
    """Scatter of width vs height for all datasets."""
    for i, (name, (widths, heights, _)) in enumerate(datasets.items()):
        clip_w = np.clip(widths, 0, width_max)
        clip_h = np.clip(heights, 0, height_max)
        ax.scatter(clip_w, clip_h, s=5, alpha=0.3, color=colors[i % 10], label=name)
    ax.set_xlim(0, width_max)
    ax.set_ylim(0, height_max)
    ax.set_title("Width vs Height Scatter (clipped 99th percentile)")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.grid(True)
    ax.legend()
    ax.axis('equal')


def plot_aspect_ratio_hist(ax, datasets, colors):
    """Aspect ratio distributions across datasets."""
    for i, (name, (widths, heights, _)) in enumerate(datasets.items()):
        ratios = widths / heights
        ratios = ratios[np.isfinite(ratios)]
        ax.hist(ratios, bins=100, color=colors[i % 10], alpha=0.5, density=True, label=name, edgecolor='black')
    ax.set_title("Aspect Ratio Distribution (Width / Height)")
    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()


def plot_size_density(ax, datasets, colors, width_max, height_max):
    """2D density plot (hexbin) for image width vs height, safe against shape mismatches."""
    hb = None
    for i, (name, (widths, heights, _)) in enumerate(datasets.items()):
        # Clip and filter valid pairs
        mask = (
            (widths <= width_max) &
            (heights <= height_max) &
            np.isfinite(widths) &
            np.isfinite(heights)
        )
        clip_w = widths[mask]
        clip_h = heights[mask]

        # Ensure arrays are same length and not empty
        n = min(len(clip_w), len(clip_h))
        if n == 0:
            continue
        clip_w, clip_h = clip_w[:n], clip_h[:n]

        hb = ax.hexbin(
            clip_w, clip_h,
            gridsize=70, cmap='viridis',
            alpha=0.4, mincnt=1
        )

    ax.set_xlim(0, width_max)
    ax.set_ylim(0, height_max)
    ax.set_title("2D Density of Image Sizes (Hexbin)")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.axis('equal')

    if hb is not None:
        plt.colorbar(hb, ax=ax, label="Image Count (approx.)")


# -------------------------------
# Combined overlaid comparison
# -------------------------------

def plot_all_datasets_together(datasets, output_dir):
    """Plots all datasets together with multiple visualization methods."""
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    # Compute global 99th percentile limits
    all_widths = np.concatenate([w for w, _, _ in datasets.values()])
    all_heights = np.concatenate([h for _, h, _ in datasets.values()])
    all_areas = np.concatenate([a for _, _, a in datasets.values()])
    width_max = np.percentile(all_widths, 99)
    height_max = np.percentile(all_heights, 99)
    area_max = np.percentile(all_areas, 99)

    # ---- Create figure with multiple visualization types ----
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # 1️⃣ Width distributions
    for i, (name, (widths, _, _)) in enumerate(datasets.items()):
        axes[0, 0].hist(widths, bins=100, color=colors[i % 10], alpha=0.5, density=True, label=name, edgecolor='black')
    axes[0, 0].set_xlim(0, width_max)
    axes[0, 0].set_title("Image Widths")
    axes[0, 0].set_xlabel("Width (pixels)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2️⃣ Height distributions
    for i, (name, (_, heights, _)) in enumerate(datasets.items()):
        axes[0, 1].hist(heights, bins=100, color=colors[i % 10], alpha=0.5, density=True, label=name, edgecolor='black')
    axes[0, 1].set_xlim(0, height_max)
    axes[0, 1].set_title("Image Heights")
    axes[0, 1].set_xlabel("Height (pixels)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3️⃣ Area distributions
    for i, (name, (_, _, areas)) in enumerate(datasets.items()):
        axes[1, 0].hist(areas, bins=100, color=colors[i % 10], alpha=0.5, density=True, label=name, edgecolor='black')
    axes[1, 0].set_xlim(0, area_max)
    axes[1, 0].set_title("Image Areas (Width × Height)")
    axes[1, 0].set_xlabel("Area (pixels²)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4️⃣ Aspect ratio histograms
    plot_aspect_ratio_hist(axes[1, 1], datasets, colors)

    # 5️⃣ Width vs Height scatter
    plot_size_scatter(axes[2, 0], datasets, colors, width_max, height_max)

    # 6️⃣ 2D density (hexbin)
    plot_size_density(axes[2, 1], datasets, colors, width_max, height_max)

    plt.suptitle("Comprehensive Image Size Comparison Across Datasets", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(output_dir, "all_datasets_comprehensive.png")
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Saved comprehensive visualization to {output_path}")


# -------------------------------
# Main entry point
# -------------------------------

'''def main(dataset_dirs, output_dir):
    datasets = {}
    for dataset_dir in dataset_dirs:
        name = os.path.basename(os.path.normpath(dataset_dir))
        print(f"\nProcessing dataset: {name}")
        image_paths = get_image_paths(dataset_dir)
        widths, heights, areas = collect_image_sizes(image_paths)
        datasets[name] = (widths, heights, areas)

    plot_all_datasets_together(datasets, output_dir)
'''

import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd  # ✅ Added for CSV output


# -------------------------------
# Data collection
# -------------------------------

import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd


# -------------------------------
# Data collection
# -------------------------------

def get_image_paths(dataset_root):
    image_paths = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def collect_image_sizes(image_paths):
    valid_paths, widths, heights, areas = [], [], [], []
    for path in tqdm(image_paths, desc=f"Reading image sizes"):
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Couldn't open {path}: {e}")
    return valid_paths, np.array(widths), np.array(heights), np.array(areas)


# -------------------------------
# Main visualization + saving
# -------------------------------

def main(dataset_dirs, output_dir, from_csv=False):
    datasets = {}
    os.makedirs(output_dir, exist_ok=True)

    for dataset_dir in dataset_dirs:
        name = os.path.basename(os.path.normpath(dataset_dir))
        csv_path = os.path.join(output_dir, f"{name}_sizes.csv")

        if from_csv and os.path.exists(csv_path):
            # ✅ Read existing CSV
            print(f"\nReading precomputed CSV for {name}: {csv_path}")
            df = pd.read_csv(csv_path)
            widths = df["width"].to_numpy()
            heights = df["height"].to_numpy()
            areas = df["area"].to_numpy()
        else:
            # ✅ Compute from images and save CSV
            print(f"\nProcessing dataset: {name}")
            image_paths = get_image_paths(dataset_dir)
            valid_paths, widths, heights, areas = collect_image_sizes(image_paths)

            df = pd.DataFrame({
                "image_path": valid_paths,
                "width": widths,
                "height": heights,
                "area": areas
            })
            df.to_csv(csv_path, index=False)
            print(f"Saved image size data for {name} to {csv_path}")

        datasets[name] = (widths, heights, areas)

    # ✅ Combined scatter plot
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Compute percentile clipping across all datasets
    all_widths = np.concatenate([w for w, _, _ in datasets.values()])
    all_heights = np.concatenate([h for _, h, _ in datasets.values()])
    width_max = np.percentile(all_widths, 99)
    height_max = np.percentile(all_heights, 99)
    names = list(datasets.keys())

    for i, ((name, (widths, heights, _)), color) in enumerate(zip(datasets.items(), colors)):
        mask = (widths <= width_max) & (heights <= height_max)
        plt.scatter(widths[mask], heights[mask], s=5, alpha=0.9, color=color, label=names[i])

    plt.xlabel("Width (pixels)", fontsize=20)
    plt.ylabel("Height (pixels)", fontsize=20)
    plt.axis('equal')
    plt.xlim(0, 4500)
    plt.ylim(0, 4500)
    tick_positions = [0, 250, 500, 1000, 2000, 3000, 4000]
    plt.xticks(tick_positions, fontsize=14)
    plt.yticks(tick_positions, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc='upper center',
        ncol=len(datasets)//2 + 1,
        markerscale=3,
        fancybox=True,
        shadow=False,
        fontsize=16
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "width_vs_height_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive multi-dataset image size visualization (overlaid)."
    )
    parser.add_argument("dataset_dirs", nargs="+", help="Paths to dataset directories (names used for CSVs)")
    parser.add_argument("--output_dir", default="histograms_comprehensive",
                        help="Directory to save results (plots + CSV files)")
    parser.add_argument("--from_csv", action="store_true",
                        help="If set, read precomputed CSVs from output_dir instead of rescanning images")
    args = parser.parse_args()
    main(args.dataset_dirs, args.output_dir, args.from_csv)
