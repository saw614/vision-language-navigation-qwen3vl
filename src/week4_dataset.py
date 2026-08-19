import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

ACTION_MAP = {
    0: "Stop",
    1: "Move forward 25cm",
    2: "Turn left 15 degree",
    3: "Turn right 15 degree",
}

class VLNVerseActionDataset(Dataset):
    def __init__(self, root_dir, split_file, max_episodes=None):
        self.root_dir = Path(root_dir)
        self.samples = []

        with gzip.open(split_file, "rt", encoding="utf-8") as f:
            split_data = json.load(f)

        episodes = split_data["episodes"]
        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        for ep in episodes:
            scan = ep["scan"]
            episode_id = ep["episode_id"]
            traj_name = "_".join(episode_id.split("_")[-2:])
            instruction = ep["instruction"]["instruction_text"]

            traj_dir = self.root_dir / "traj_data" / "vlnverse" / scan / traj_name
            parquet_path = traj_dir / "data/chunk-000/episode_000000.parquet"
            rgb_path = traj_dir / "videos/chunk-000/observation.images.rgb/rgb.npy"

            if not parquet_path.exists() or not rgb_path.exists():
                continue

            # ── FAST: only read action column, not full parquet ──
            try:
                df = pd.read_parquet(
                    parquet_path,
                    columns=["observation.action"]
                )
            except Exception:
                continue

            for t in range(len(df)):
                action_id = int(df.iloc[t]["observation.action"])
                self.samples.append({
                    "instruction": instruction,
                    "rgb_path": str(rgb_path),
                    "prev_idx": max(t - 1, 0),
                    "curr_idx": t,
                    "answer": ACTION_MAP[action_id],
                    "episode_id": episode_id,
                    "scan": scan,
                })

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # ── LAZY: only load the 2 frames needed ──────────────────
        rgb = np.load(item["rgb_path"], mmap_mode='r')
        img_prev = Image.fromarray(
            rgb[item["prev_idx"]].astype(np.uint8)
        )
        img_curr = Image.fromarray(
            rgb[item["curr_idx"]].astype(np.uint8)
        )
        return {
            "instruction": item["instruction"],
            "images": [img_prev, img_curr],
            "answer": item["answer"],
            "episode_id": item["episode_id"],
            "scan": item["scan"],
        }
