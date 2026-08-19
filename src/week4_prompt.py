SYSTEM_PROMPT = """You are a navigation action classifier.

Given the full navigation instruction and two egocentric images, predict the next action.

You must output exactly one action and nothing else.

Valid actions:
Move forward 25cm
Turn right 15 degree
Turn left 15 degree
Stop

Do not explain.
Do not output multiple actions.
Do not add extra words."""


def build_train_messages(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["images"][0], "min_pixels": 784, "max_pixels": 50176},
                {"type": "image", "image": sample["images"][1], "min_pixels": 784, "max_pixels": 50176},
                {
                    "type": "text",
                    "text": (
                        "Instruction:\n"
                        f"{sample['instruction']}\n\n"
                        "Answer:"
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answer"]}],
        },
    ]


def build_infer_messages(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["images"][0], "min_pixels": 784, "max_pixels": 50176},
                {"type": "image", "image": sample["images"][1], "min_pixels": 784, "max_pixels": 50176},
                {
                    "type": "text",
                    "text": (
                        "Instruction:\n"
                        f"{sample['instruction']}\n\n"
                        "Answer:"
                    ),
                },
            ],
        },
    ]
