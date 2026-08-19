import os
import argparse
import json
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from week4_dataset import VLNVerseActionDataset
from week4_collate import QwenVLNCollator


VALID_ACTIONS = [
    "Move forward 25cm",
    "Turn right 15 degree",
    "Turn left 15 degree",
    "Stop",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=141)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_dir", type=str, default="checkpoints/week4_wandb_run")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--wandb_project", type=str, default="week4_vln")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--eval_json", type=str,
                        default="vlnverse_closed_loop_eval_20episodes.json")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_steps", type=int, default=1000,
                        help="Save mid-epoch checkpoint every N steps (0 to disable)")
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="Reset LR scheduler on resume (use when resuming with new epochs)")
    return parser.parse_args()


def evaluate(model, loader, processor, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            model_batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            outputs = model(**model_batch)
            total_loss += outputs.loss.item()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values.to(device)
            if image_grid_thw is not None:
                gen_kwargs["image_grid_thw"] = image_grid_thw.to(device)

            generated_ids = model.generate(**gen_kwargs)
            input_len = input_ids.shape[1]

            for i in range(len(generated_ids)):
                new_ids = generated_ids[i, input_len:]
                raw = processor.tokenizer.decode(
                    new_ids, skip_special_tokens=True
                ).strip()
                label_ids = batch["labels"][i]
                answer_ids = label_ids[label_ids != -100]
                if len(answer_ids) > 1:
                    answer_ids = answer_ids[:-1]
                target = processor.tokenizer.decode(
                    answer_ids, skip_special_tokens=True
                ).strip()
                if raw == target:
                    correct += 1
                total += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return avg_loss, accuracy


def get_episode_split(ds, val_ratio=0.1, eval_json=None):
    required_val_episodes = set()
    if eval_json and os.path.exists(eval_json):
        with open(eval_json) as f:
            eval_data = json.load(f)
        required_val_episodes = set(
            ep["episode_id"]
            for ep in eval_data["episodes"]
            if ep["split"] == "val"
        )
        print(f"Required val episodes from eval JSON: "
              f"{len(required_val_episodes)}")

    episode_ids = []
    seen = set()
    for sample in ds.samples:
        eid = sample["episode_id"]
        if eid not in seen:
            episode_ids.append(eid)
            seen.add(eid)

    remaining_episodes = [
        e for e in episode_ids
        if e not in required_val_episodes
    ]

    target_val_count = max(
        len(required_val_episodes),
        int(len(episode_ids) * val_ratio)
    )
    n_extra_val = max(
        0, target_val_count - len(required_val_episodes)
    )
    extra_val = set(remaining_episodes[-n_extra_val:]) \
        if n_extra_val > 0 else set()

    val_episodes = required_val_episodes | extra_val
    train_episodes = set(episode_ids) - val_episodes

    train_indices = [
        i for i, s in enumerate(ds.samples)
        if s["episode_id"] in train_episodes
    ]
    val_indices = [
        i for i, s in enumerate(ds.samples)
        if s["episode_id"] in val_episodes
    ]

    found_required = len(required_val_episodes & set(episode_ids))
    print(f"Train episodes: {len(train_episodes)} "
          f"({len(train_indices)} samples)")
    print(f"Val episodes:   {len(val_episodes)} "
          f"({len(val_indices)} samples)")
    print(f"Required val episodes found: "
          f"{found_required}/{len(required_val_episodes)}")

    return train_indices, val_indices


args = parse_args()

wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    config={
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_episodes": args.max_episodes,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": ["q_proj", "v_proj"],
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "dtype": "float16",
        "val_ratio": args.val_ratio,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "checkpoint_steps": args.checkpoint_steps,
        "reset_scheduler": args.reset_scheduler,
    }
)

print("=" * 60)
print(f"batch_size        = {args.batch_size}")
print(f"epochs            = {args.epochs}")
print(f"max_episodes      = {args.max_episodes}")
print(f"lr                = {args.lr}")
print(f"warmup_ratio      = {args.warmup_ratio}")
print(f"save_dir          = {args.save_dir}")
print(f"lora_r            = {args.lora_r}")
print(f"lora_alpha        = {args.lora_alpha}")
print(f"resume            = {args.resume}")
print(f"num_workers       = {args.num_workers}")
print(f"checkpoint_steps  = {args.checkpoint_steps}")
print(f"reset_scheduler   = {args.reset_scheduler}")
print("=" * 60)

model_name = "Qwen/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
)

base_model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.train()
model.print_trainable_parameters()

ds = VLNVerseActionDataset(
    root_dir="data/VLNVerse_data",
    split_file="data/VLNVerse_data/raw_data/final_splits/fine_train.json.gz",
    max_episodes=args.max_episodes,
)

train_indices, val_indices = get_episode_split(
    ds,
    val_ratio=args.val_ratio,
    eval_json=args.eval_json,
)

train_ds = Subset(ds, train_indices)
val_ds = Subset(ds, val_indices)

collator = QwenVLNCollator(processor)

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collator,
    num_workers=args.num_workers,
    pin_memory=True,
    prefetch_factor=2 if args.num_workers > 0 else None,
    persistent_workers=True if args.num_workers > 0 else False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=args.num_workers,
    pin_memory=True,
    prefetch_factor=2 if args.num_workers > 0 else None,
    persistent_workers=True if args.num_workers > 0 else False,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

total_steps = len(train_loader) * args.epochs
warmup_steps = int(total_steps * args.warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"total_steps       = {total_steps}")
print(f"warmup_steps      = {warmup_steps}")

print("=== Verifying label masking ===")
for batch in train_loader:
    for i in range(min(3, len(batch["labels"]))):
        label_ids = batch["labels"][i]
        answer_tokens = label_ids[label_ids != -100]
        decoded = processor.tokenizer.decode(answer_tokens)
        print(f"  sample {i} labels: {repr(decoded)}")
    break
print("=== Done verifying ===")

start_epoch = 0
global_step = 0
best_val_loss = float("inf")

if args.resume is not None:
    print("Loading checkpoint:", args.resume)
    ckpt = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt and not args.reset_scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print("Scheduler state restored from checkpoint")
    else:
        print("Scheduler reset: starting fresh cosine schedule")
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("val_loss", float("inf")) or float("inf")
    print(f"Resumed from epoch {start_epoch}, "
          f"global_step {global_step}")

os.makedirs(args.save_dir, exist_ok=True)

for epoch in range(start_epoch, args.epochs):
    total_loss = 0.0
    model.train()

    for step, batch in enumerate(train_loader):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(model.device)

        try:
            outputs = model(**batch)
            loss = outputs.loss
        except torch.cuda.OutOfMemoryError:
            print(f'WARNING: OOM at step {step}, skipping batch')
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
        if torch.isnan(loss) or torch.isinf(loss):
            print(f'WARNING: NaN/Inf loss at step {step}, skipping batch')
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        global_step += 1
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": current_lr,
            "train/epoch": epoch + 1,
            "train/global_step": global_step,
        }, step=global_step)

        if step % 20 == 0:
            avg = total_loss / (step + 1)
            print(
                f"epoch {epoch+1}/{args.epochs} "
                f"step {step}/{len(train_loader)} "
                f"loss {loss.item():.6f} "
                f"avg {avg:.6f} "
                f"lr {current_lr:.8f}"
            )

        # Mid-epoch checkpoint
        if args.checkpoint_steps > 0 and global_step % args.checkpoint_steps == 0:
            mid_ckpt_path = os.path.join(
                args.save_dir,
                f"checkpoint_step{global_step}.pt"
            )
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }, mid_ckpt_path)
            print(f"Mid-epoch checkpoint saved at step {global_step} "
                  f"-> {mid_ckpt_path}")

    avg_epoch_loss = total_loss / len(train_loader)
    print(f"epoch {epoch+1} finished "
          f"avg_train_loss={avg_epoch_loss:.6f}")

    print(f"Running validation for epoch {epoch+1}...")
    val_loss, val_acc = evaluate(
        model, val_loader, processor, model.device
    )
    print(f"epoch {epoch+1} val_loss={val_loss:.6f} "
          f"val_acc={val_acc:.4f}")

    wandb.log({
        "epoch/train_loss": avg_epoch_loss,
        "epoch/val_loss": val_loss,
        "epoch/val_accuracy": val_acc,
        "epoch/epoch": epoch + 1,
    }, step=global_step)

    epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_dir = os.path.join(args.save_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        print(f"New best model saved (val_loss={val_loss:.6f})")

    # Delete previous mid-epoch checkpoints to save space
    for f in os.listdir(args.save_dir):
        if f.startswith("checkpoint_step") and f.endswith(".pt"):
            os.remove(os.path.join(args.save_dir, f))
            print(f"Deleted mid-epoch checkpoint: {f}")

    checkpoint_path = os.path.join(
        args.save_dir, "checkpoint_latest.pt"
    )
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "avg_epoch_loss": avg_epoch_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"saved epoch model to {epoch_dir}")
    print(f"saved checkpoint to {checkpoint_path}")

wandb.finish()
print("training finished")
