import os, sys, json, gzip, math, argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from week4_prompt import build_infer_messages
from week4_dataset import VLNVerseActionDataset

VALID_ACTIONS = [
    "Move forward 25cm",
    "Turn right 15 degree",
    "Turn left 15 degree",
    "Stop",
]

ACTION_MAP = {
    0: "Stop",
    1: "Move forward 25cm",
    2: "Turn left 15 degree",
    3: "Turn right 15 degree",
}

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", type=str, default="checkpoints/week4_baseline/best")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--steps_per_episode", type=int, default=3)
parser.add_argument("--max_episodes", type=int, default=500)
args = parser.parse_args()

print(f"\n{'='*60}")
print(f"Loading model from: {args.adapter_path}")
print(f"{'='*60}")

processor = AutoProcessor.from_pretrained(
    args.adapter_path,
    trust_remote_code=True,
)

base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, args.adapter_path)
model.eval()

print("Model loaded!")

ds = VLNVerseActionDataset(
    root_dir="data/VLNVerse_data",
    split_file="data/VLNVerse_data/raw_data/final_splits/fine_train.json.gz",
    max_episodes=args.max_episodes,
)

print(f"Dataset length: {len(ds)}")

if len(ds) == 0:
    print("No samples loaded. Try increasing --max_episodes or check dataset path.")
    sys.exit(0)

correct = 0
total = 0

for ep_idx in range(min(args.num_episodes, max(1, len(ds) // args.steps_per_episode))):
    print(f"\n{'─'*60}")
    print(f"Episode {ep_idx + 1}")

    for step in range(args.steps_per_episode):
        idx = ep_idx * args.steps_per_episode + step

        if idx >= len(ds):
            break

        sample = ds[idx]

        messages = build_infer_messages(sample)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )

        raw = processor.tokenizer.decode(
            gen[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        pred = raw if raw in VALID_ACTIONS else "Move forward 25cm"
        gt = sample["answer"]

        ok = pred == gt

        if ok:
            correct += 1

        total += 1

        print(
            f"  Step {step + 1:2d} | "
            f"GT: {gt:<25} | "
            f"Raw: {raw:<25} | "
            f"Pred: {pred:<25} | "
            f"{'✅' if ok else '❌'}"
        )

if total > 0:
    print(f"\n{'='*60}")
    print(f"Overall accuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"{'='*60}")
else:
    print("No samples were evaluated.")
