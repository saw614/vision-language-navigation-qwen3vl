import torch
from qwen_vl_utils import process_vision_info
from week4_prompt import build_train_messages


class QwenVLNCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, batch):
        all_texts = []
        all_messages = []

        for sample in batch:
            messages = build_train_messages(sample)
            all_messages.append(messages)

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            all_texts.append(text)

        image_inputs, video_inputs = process_vision_info(all_messages)

        inputs = self.processor(
            text=all_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = torch.full_like(inputs["input_ids"], -100)

        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

        assistant_ids = self.processor.tokenizer(
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

            labels[i, answer_start:last_im_end + 1] = input_ids[answer_start:last_im_end + 1]

        inputs["labels"] = labels
        return inputs
