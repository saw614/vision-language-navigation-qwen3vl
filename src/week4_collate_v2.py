import torch
from qwen_vl_utils import process_vision_info
from week4_prompt import build_train_messages
from transformers import AutoProcessor

_worker_processor = None

def worker_init_fn(worker_id):
    global _worker_processor
    _worker_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True,
    )
    _worker_processor.tokenizer.padding_side = "left"


class QwenVLNCollatorV2:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, batch):
        if "input_ids" in batch[0]:
            return self._collate_processed(batch)
        return self._process_and_collate(batch, self.processor)

    def _collate_processed(self, batch):
        all_input_ids = [b["input_ids"] for b in batch]
        all_attention_mask = [b["attention_mask"] for b in batch]
        all_pixel_values = [b["pixel_values"] for b in batch]
        all_image_grid_thw = [b["image_grid_thw"] for b in batch]
        all_labels = [b["labels"] for b in batch]
        # mm_token_type_ids — present when processor returns it
        has_mm = "mm_token_type_ids" in batch[0]
        if has_mm:
            all_mm_token_type_ids = [b["mm_token_type_ids"] for b in batch]

        max_len = max(t.shape[0] for t in all_input_ids)
        pad_id = self.processor.tokenizer.pad_token_id \
            if self.processor.tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        padded_mm_token_type_ids = []

        for i, (input_ids, attn_mask, labels) in enumerate(
            zip(all_input_ids, all_attention_mask, all_labels)
        ):
            pad_len = max_len - input_ids.shape[0]
            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_id, dtype=input_ids.dtype), input_ids])
            )
            padded_attention_mask.append(
                torch.cat([torch.zeros(pad_len, dtype=attn_mask.dtype), attn_mask])
            )
            padded_labels.append(
                torch.cat([torch.full((pad_len,), -100, dtype=labels.dtype), labels])
            )
            if has_mm:
                mm = all_mm_token_type_ids[i]
                padded_mm_token_type_ids.append(
                    torch.cat([torch.zeros(pad_len, dtype=mm.dtype), mm])
                )

        inputs = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "image_grid_thw": torch.cat(all_image_grid_thw, dim=0),
            "labels": torch.stack(padded_labels),
        }
        if has_mm:
            inputs["mm_token_type_ids"] = torch.stack(padded_mm_token_type_ids)
        return inputs

    def _process_and_collate(self, batch, processor):
        all_texts = []
        all_messages = []
        for sample in batch:
            messages = build_train_messages(sample)
            all_messages.append(messages)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            all_texts.append(text)

        image_inputs, video_inputs = process_vision_info(all_messages)
        inputs = processor(
            text=all_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = torch.full_like(inputs["input_ids"], -100)
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_ids = processor.tokenizer(
            "assistant", add_special_tokens=False
        )["input_ids"]

        for i in range(len(batch)):
            input_ids = inputs["input_ids"][i]
            assistant_block_start = None
            for pos in range(len(input_ids) - len(assistant_ids)):
                if input_ids[pos].item() == im_start_id:
                    match = True
                    for j, aid in enumerate(assistant_ids):
                        if input_ids[pos + 1 + j].item() != aid:
                            match = False
                            break
                    if match:
                        assistant_block_start = pos
            if assistant_block_start is None:
                print(f"WARNING: assistant block not found for sample {i}")
                continue
            answer_start = assistant_block_start + 1 + len(assistant_ids) + 1
            im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
            if len(im_end_positions) == 0:
                print(f"WARNING: no <|im_end|> for sample {i}")
                continue
            last_im_end = im_end_positions[-1].item()
            if answer_start > last_im_end:
                print(f"WARNING: answer_start={answer_start} > last_im_end={last_im_end}")
                continue
            labels[i, answer_start:last_im_end + 1] = \
                input_ids[answer_start:last_im_end + 1]

        inputs["labels"] = labels
        return inputs


def preprocess_sample(sample):
    global _worker_processor
    if _worker_processor is None:
        return sample

    processor = _worker_processor
    messages = build_train_messages(sample)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    # Build labels
    labels = torch.full_like(input_ids, -100)
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_ids = processor.tokenizer(
        "assistant", add_special_tokens=False
    )["input_ids"]

    assistant_block_start = None
    for pos in range(len(input_ids) - len(assistant_ids)):
        if input_ids[pos].item() == im_start_id:
            match = True
            for j, aid in enumerate(assistant_ids):
                if input_ids[pos + 1 + j].item() != aid:
                    match = False
                    break
            if match:
                assistant_block_start = pos

    if assistant_block_start is not None:
        answer_start = assistant_block_start + 1 + len(assistant_ids) + 1
        im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
        if len(im_end_positions) > 0:
            last_im_end = im_end_positions[-1].item()
            if answer_start <= last_im_end:
                labels[answer_start:last_im_end + 1] = \
                    input_ids[answer_start:last_im_end + 1]
    else:
        print(f"WARNING: preprocess_sample — assistant block not found")

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,
        "instruction": sample["instruction"],
        "answer": sample["answer"],
        "episode_id": sample["episode_id"],
    }

    # Include mm_token_type_ids if processor returned it
    if "mm_token_type_ids" in inputs:
        result["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]

    return result
