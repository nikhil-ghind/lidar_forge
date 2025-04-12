import os
import struct
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def load_bin(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, 4)


class LidarForgeDataset(Dataset):
    def __init__(self, root: str, num_points: int = 16384, split: str = "train"):
        self.root = root
        self.num_points = num_points
        velodyne_dir = os.path.join(root, "velodyne")
        all_ids = sorted(f[:-4] for f in os.listdir(velodyne_dir) if f.endswith(".bin"))
        n = len(all_ids)
        if split == "train":
            self.ids = all_ids[: int(n * 0.8)]
        else:
            self.ids = all_ids[int(n * 0.8):]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        pts = load_bin(os.path.join(self.root, "velodyne", sid + ".bin"))

        if len(pts) >= self.num_points:
            chosen = np.random.choice(len(pts), self.num_points, replace=False)
        else:
            chosen = np.random.choice(len(pts), self.num_points, replace=True)
        pts = pts[chosen]

        label_path = os.path.join(self.root, "labels", sid + ".json")
        with open(label_path) as f:
            ann = json.load(f)

        num_boxes = len(ann.get("bounding_boxes", []))
        condition = torch.zeros(64)
        condition[0] = min(num_boxes, 63)

        return torch.from_numpy(pts).float(), condition
