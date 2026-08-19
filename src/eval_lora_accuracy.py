import argparse
import torch
from collections import Counter
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from week4_dataset import VLNVerseActionDataset
from week4_prompt import build_infer_messages


VALID_ACTIONS = [
    "Move forward 25cm",
    "Turn right 15 degree",
    "Turn left 15 degree",
    "Stop",
]


def parse_output(raw):
    raw = raw.strip()

    # exact match only = valid
    if raw in VALID_ACTIONS:
        return raw, False

    # invalid output rule
    return "Move forward 25cm", True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    ds = VLNVerseActionDataset(
        root_dir="data/VLNVerse_data",
        split_file="data/VLNVerse_data/raw_data/final_splits/fine_train.json.gz",
        max_episodes=args.max_episodes,
    )

    n = min(args.max_samples, len(ds))

    correct = 0
    invalid = 0
    total = 0

    target_counter = Counter()
    pred_counter = Counter()

    for i in range(n):
        sample = ds[i]
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
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=0,
                use_cache=True,
            )

        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        raw_output = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )[0].strip()

        pred, is_invalid = parse_output(raw_output)
        target = sample["answer"]

        total += 1
        correct += int(pred == target)
        invalid += int(is_invalid)

        target_counter[target] += 1
        pred_counter[pred] += 1

        print(f"[{i}] target={target} | raw={raw_output} | pred={pred} | invalid={is_invalid}")

    print("=" * 60)
    print("total samples:", total)
    print("accuracy:", correct / total)
    print("invalid output rate:", invalid / total)
    print("target distribution:", target_counter)
    print("prediction distribution:", pred_counter)


if __name__ == "__main__":
    main()
