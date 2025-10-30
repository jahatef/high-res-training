import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_val_accuracy(csv_path, output_dir="."):
    """
    Plot validation accuracy from a whitespace-separated CSV file.
    Expected columns: epoch val_accuracy
    """
    df = pd.read_csv(csv_path, delim_whitespace=True)
    
    # Ensure correct column names (adjust if your file differs)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Figure 1: Epochs 0–4 ----
    subset1 = df[(df['Epoch'] >= 0) & (df['Epoch'] <= 3)]
    plt.figure(figsize=(6, 4))
    plt.plot(subset1['Epoch'], subset1['ViT-384'], marker='o', color='C0', linestyle='dashed', label="ViT-384")
    plt.plot(subset1['Epoch'], subset1['ViT-4096'], marker='o', color='C1', linestyle='dotted', label="ViT-4096")
    plt.plot(subset1['Epoch'], subset1['ViT-4096-Synth'], marker='o', color='C2', label="ViT-4096-Synth")
    plt.legend()
    #plt.title('Validation Accuracy (Epochs 0–4)')
    plt.xlabel('Epoch')
    plt.xticks([0,1,2,3])
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    fig1_path = os.path.join(output_dir, "val_acc_epochs_0_4.png")
    plt.savefig(fig1_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"✅ Saved: {fig1_path}")

    # ---- Figure 2: Epochs 5–15 ----
    subset2 = df[(df['Epoch'] >=4) & (df['Epoch'] <= 15)]
    plt.figure(figsize=(6, 4))
    plt.plot(subset2['Epoch'], subset2['ViT-384'], marker='o', color='C0', linestyle='dashed', label="ViT-384")
    plt.plot(subset2['Epoch'], subset2['ViT-4096'], marker='o', color='C1', linestyle='dotted', label="ViT-4096")
    plt.plot(subset2['Epoch'], subset2['ViT-4096-Synth'], marker='o', color='C2', label="ViT-4096-Synth")
    #plt.legend()
    #plt.title('Validation Accuracy (Epochs 5–15)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    fig2_path = os.path.join(output_dir, "val_acc_epochs_5_15.png")
    plt.savefig(fig2_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"✅ Saved: {fig2_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validation accuracy from whitespace-separated CSV file.")
    parser.add_argument("--csv", type=str, required=True, help="Path to whitespace-separated CSV file with columns 'epoch' and 'val_accuracy'.")
    args = parser.parse_args()

    plot_val_accuracy(args.csv)
