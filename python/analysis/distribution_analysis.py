"""Statistical comparison between real and generated point clouds."""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def collect_clouds(directory: str, max_n: int = 200) -> List[np.ndarray]:
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))[:max_n]
    return [load_bin(os.path.join(directory, f)) for f in files]


def point_density(clouds: List[np.ndarray]) -> np.ndarray:
    return np.array([len(c) for c in clouds])


def range_histogram(clouds: List[np.ndarray], bins: int = 50):
    ranges = np.concatenate([np.sqrt((c[:, :3] ** 2).sum(1)) for c in clouds])
    return np.histogram(ranges, bins=bins)


def intensity_histogram(clouds: List[np.ndarray], bins: int = 50):
    intensities = np.concatenate([c[:, 3] for c in clouds])
    return np.histogram(intensities, bins=bins)


def plot_comparison(real_dir: str, gen_dir: str, out_dir: str = "analysis_plots"):
    os.makedirs(out_dir, exist_ok=True)
    real = collect_clouds(real_dir)
    gen  = collect_clouds(gen_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(point_density(real), bins=20, alpha=0.6, label="real", color="steelblue")
    axes[0].hist(point_density(gen),  bins=20, alpha=0.6, label="generated", color="coral")
    axes[0].set_title("Point Density")
    axes[0].legend()

    rh_r, be_r = range_histogram(real)
    rh_g, be_g = range_histogram(gen)
    axes[1].stairs(rh_r, be_r, label="real", color="steelblue")
    axes[1].stairs(rh_g, be_g, label="generated", color="coral")
    axes[1].set_title("Range Distribution")
    axes[1].legend()

    ih_r, be_ir = intensity_histogram(real)
    ih_g, be_ig = intensity_histogram(gen)
    axes[2].stairs(ih_r, be_ir, label="real", color="steelblue")
    axes[2].stairs(ih_g, be_ig, label="generated", color="coral")
    axes[2].set_title("Intensity Distribution")
    axes[2].legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "distribution_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir",  required=True)
    parser.add_argument("--out_dir",  default="analysis_plots")
    args = parser.parse_args()
    plot_comparison(args.real_dir, args.gen_dir, args.out_dir)
