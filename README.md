# Vision-Language Navigation with Qwen3-VL + LoRA

> Fine-tuning and closed-loop evaluation of a Vision-Language Model for embodied robot navigation in NVIDIA IsaacSim.

## Course Project Notice

This project was developed as part of the **Deep Learning course at Korea Aerospace University (KAU), taught by Prof. Young-Sik Choi**.

This repository presents my **Week 4 implementation and experimental work**, including Qwen3-VL LoRA fine-tuning, training and validation, debugging, IsaacSim closed-loop evaluation, and navigation performance analysis.

The VLNVerse dataset and course materials are not distributed through this repository.

---

## Overview

This project implements a **Vision-Language Navigation (VLN)** system using **Qwen3-VL-2B-Instruct** with **LoRA (Low-Rank Adaptation)**.

The objective is to enable a navigation agent to interpret visual observations together with a natural-language instruction and predict the next navigation action.

At each step, the model receives:

- a natural-language navigation instruction,
- the previous RGB observation,
- the current RGB observation,

and predicts one of four discrete navigation actions:

| Action | Command |
|---|---|
| Forward | `Move forward 25cm` |
| Left | `Turn left 15 degree` |
| Right | `Turn right 15 degree` |
| Stop | `Stop` |

The project covers the complete pipeline from dataset preprocessing and multimodal fine-tuning to simulator-based closed-loop navigation evaluation.

---

## System Pipeline

```text
Navigation Instruction
        +
Previous RGB Image
        +
 Current RGB Image
        │
        ▼
   Qwen3-VL Processor
        │
        ▼
Vision + Language Tokens
        │
        ▼
 Qwen3-VL-2B-Instruct
        │
   LoRA Fine-tuning
   (q_proj, v_proj)
        │
        ▼
 Navigation Action
        │
        ▼
   IsaacSim Robot
        │
        ▼
 New Observation
        │
        └──────────────► Next Navigation Step
```

During training, the model learns to predict the ground-truth navigation action from the instruction and two consecutive observations.

During closed-loop evaluation, the predicted action is executed in IsaacSim. A new observation is then captured and passed back to the model, creating a continuous perception-action loop.

---

## Model

### Base Model

**Qwen3-VL-2B-Instruct**

The model combines a vision encoder with a language decoder, allowing visual observations and textual navigation instructions to be processed jointly.

### LoRA Configuration

Instead of fine-tuning the entire model, **LoRA** is applied to selected attention projection layers.

| Parameter | Value |
|---|---:|
| LoRA Rank (`r`) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | `q_proj`, `v_proj` |

This approach significantly reduces the number of trainable parameters while preserving the capabilities of the pretrained vision-language model.

---

## Dataset

The project uses the **VLNVerse Fine-Grained Navigation** dataset.

The dataset is divided using an **episode-level split** to prevent observations from the same trajectory from appearing in both training and validation sets.

| Split | Episodes |
|---|---:|
| Training | 1,350 |
| Validation | 150 |
| Total | 1,500 |

Total training/validation samples:

**6,582**

### Action Distribution

| Action | Samples |
|---|---:|
| Move Forward | 4,356 |
| Turn Left | 1,111 |
| Turn Right | 974 |
| Stop | 141 |

The action distribution is highly imbalanced, with forward movement representing the majority of training samples.

> The dataset itself is **not included in this repository**.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Qwen3-VL-2B-Instruct |
| Learning Rate | `1e-4` |
| Batch Size | 8 |
| Epochs | 3 |
| Optimizer | AdamW |
| Scheduler | Cosine with Warmup |
| Warmup Ratio | 0.03 |
| Precision | bfloat16 |
| GPU | NVIDIA RTX 4090 |

Training was performed using **PyTorch**, **Hugging Face Transformers**, and **PEFT**.

Experiment monitoring was performed with **Weights & Biases (W&B)**.

---

## Training Results

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
|---:|---:|---:|---:|
| 1 | 0.1076 | 0.0914 | 63.20% |
| **2** | **0.0897** | **0.0885** | **63.50%** |
| 3 | 0.0806 | 0.0883 | 62.72% |

The best validation accuracy was obtained at **Epoch 2: 63.50%**.

Although training loss continued to decrease in Epoch 3, validation accuracy slightly decreased, suggesting the beginning of overfitting.

---

# Closed-Loop Navigation Evaluation

Offline prediction accuracy alone is not sufficient for evaluating a navigation agent.

A model can correctly predict actions on prerecorded observations but still fail when its own predictions change the future observations it receives.

For this reason, the trained model was evaluated in a **closed-loop IsaacSim environment**.

### Environment

- NVIDIA IsaacSim 4.5.0
- Egocentric RGB observations
- Dataset-defined navigation start positions
- Model-generated navigation actions

At every simulation step:

```text
Capture Observation
        ↓
Build Multimodal Prompt
        ↓
Qwen3-VL + LoRA Inference
        ↓
Parse Navigation Action
        ↓
Execute Action in IsaacSim
        ↓
Capture Next Observation
        ↓
Repeat
```

---

## Evaluation Metrics

The navigation system was evaluated using:

| Metric | Description |
|---|---|
| SR | Success Rate |
| OSR | Oracle Success Rate |
| SPL | Success weighted by Path Length |
| nDTW | Normalized Dynamic Time Warping |
| Goal Distance | Final distance from the target |
| Invalid Rate | Fraction of invalid model outputs |

---

# Debugging the Simulation-Evaluation Pipeline

One of the most important findings during this project was that poor navigation performance was not caused only by the trained model.

Several inconsistencies between the training observations and the IsaacSim evaluation environment were discovered.

### Problems Identified

#### 1. Incorrect Camera View

The model initially received an incorrect scene viewport rather than the robot's true egocentric observation.

This produced visual inputs that differed significantly from the training data.

#### 2. Spawn Position Mismatch

The robot was not always initialized at the exact start position defined by the navigation episode.

This created a mismatch between the expected trajectory and the actual simulation state.

#### 3. Camera Height

The camera height was adjusted to:

```text
z + 1.2
```

to provide a viewpoint closer to the observations expected by the navigation model.

#### 4. Top-Down Camera Orientation

The camera orientation was corrected so that it faced the robot's forward direction instead of pointing toward the floor.

#### 5. Transform Conflict

An IsaacSim transform error involving:

```text
rotateXYZ already exists
```

was resolved so that camera transforms could be updated reliably during navigation.

---

## Effect of Environment Fixes

Correcting the perception and simulation pipeline produced a substantial improvement in navigation performance.

| Metric | Before Fix | After Fix |
|---|---:|---:|
| SR | 0.0215 | **0.350** |
| OSR | 0.1207 | **0.450** |
| SPL | 0.158 | 0.000 |
| nDTW | 0.2620 | N/A |
| Goal Distance | 25.3 m | **5.39 m** |
| Invalid Output Rate | 0.00014 | **0.00000** |

The large improvement showed that **consistency between the training observations and deployment environment is critical for embodied Vision-Language Navigation**.

In particular, correcting the camera viewpoint, spawn position, camera orientation, and transform logic allowed the model to receive observations much closer to its training distribution.

---

# Final Closed-Loop Evaluation

To study the effect of maximum episode length, the final model was evaluated with two timeout settings.

## 50-Step Timeout

| Metric | Value |
|---|---:|
| SR | **0.395** |
| OSR | 0.479 |
| nDTW | 0.500 |
| Goal Distance | **4.480 m** |
| Invalid Rate | **0.000** |

## 100-Step Timeout

| Metric | Value |
|---|---:|
| SR | **0.393** |
| OSR | **0.579** |
| SPL | **0.287** |
| nDTW | **0.510** |
| Goal Distance | 5.448 m |
| Invalid Rate | **0.000** |

### Analysis

Increasing the timeout from 50 to 100 steps produced almost no change in final Success Rate:

```text
SR: 0.395 → 0.393
```

However, Oracle Success Rate increased substantially:

```text
OSR: 0.479 → 0.579
```

and nDTW slightly improved:

```text
nDTW: 0.500 → 0.510
```

This suggests that the additional navigation steps allowed the robot to reach or pass near the target more frequently, but did not consistently improve the final stopping decision.

The gap between **OSR and SR** indicates that final-stage navigation and stopping behavior remain important areas for improvement.

---

## Ablation Study

A smaller LoRA configuration was also evaluated.

| Setting | LoRA Rank | Alpha |
|---|---:|---:|
| Baseline | 16 | 32 |
| Ablation | 8 | 16 |

The ablation configuration reduced LoRA capacity while showing similar initial training behavior.

This experiment was used to investigate the trade-off between parameter efficiency and model adaptation capacity.

---

# Technical Challenges

A significant part of the project involved diagnosing and resolving implementation and system-level problems.

### Multimodal Label Masking

Qwen3-VL expands image placeholders into many internal vision tokens.

A naive label-masking implementation therefore caused tensor-length mismatches between model inputs and training labels.

The collator was corrected by constructing labels **after multimodal processor expansion** and masking only the prompt portion.

### Validation Accuracy Bug

The initial evaluation pipeline produced incorrect accuracy values because generated outputs contained extra text or tokens.

A stricter action parser was introduced to extract one of the four valid navigation commands before comparison with the target.

### GPU Utilization Bottleneck

Although training used an RTX 4090, GPU utilization sometimes dropped because image loading, preprocessing, and multimodal batch construction occurred on the CPU.

This demonstrated that GPU training performance can be limited by the data pipeline rather than GPU computation itself.

### VRAM Limitations

Qwen3-VL multimodal training consumed significant GPU memory.

The final training setup used:

```text
Batch size = 8
bfloat16 precision
LoRA fine-tuning
```

to remain within available VRAM.

### Checkpoint Recovery

Training interruptions required restoring model, optimizer, and learning-rate scheduler states.

Checkpoint management was therefore an important part of the training workflow.

### IsaacSim Evaluation Debugging

Camera transforms, robot spawn positions, observation viewpoints, and simulator state all had to be aligned with the training environment before meaningful closed-loop results could be obtained.

---

# Repository Structure

```text
vision-language-navigation-qwen3vl/
│
├── src/
│   ├── train_week4_subset.py
│   ├── week4_dataset.py
│   ├── week4_collate.py
│   ├── week4_collate_v2.py
│   ├── week4_prompt.py
│   ├── demo_inference.py
│   ├── eval_lora_accuracy.py
│   ├── week4_closed_loop_eval.py
│   └── evaluate_closed_loop_result.py
│
├── configs/
│   └── week4_config.yaml
│
├── results/
│   └── vlnverse_closed_loop_eval_20episodes.json
│
├── checkpoints/
│   └── week4_baseline/
│       └── best/
│           ├── adapter_model.safetensors
│           ├── adapter_config.json
│           ├── processor_config.json
│           └── tokenizer files
│
├── AI_prompt_log.md
├── .gitignore
└── README.md
```

---

# Key Scripts

### `train_week4_subset.py`

Main training script for Qwen3-VL LoRA fine-tuning.

### `week4_dataset.py`

Loads and prepares VLNVerse navigation samples.

### `week4_collate.py`

Constructs multimodal batches containing text instructions and image observations.

### `week4_collate_v2.py`

Experimental alternative collator developed while investigating preprocessing and data-loading performance.

### `demo_inference.py`

Runs inference using the fine-tuned LoRA adapter.

### `eval_lora_accuracy.py`

Evaluates action prediction accuracy.

### `week4_closed_loop_eval.py`

Runs the trained navigation policy inside IsaacSim.

### `evaluate_closed_loop_result.py`

Calculates closed-loop navigation metrics from evaluation results.

---

# Model Checkpoint

The repository contains the final LoRA adapter under:

```text
checkpoints/week4_baseline/best/
```

The adapter includes:

```text
adapter_model.safetensors
adapter_config.json
processor_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

The full Qwen3-VL base model is **not included**.

---

# Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Qwen3-VL
- NVIDIA IsaacSim
- Weights & Biases
- CUDA
- Git / GitHub

---

# Key Takeaways

This project provided practical experience across the full lifecycle of a multimodal deep-learning navigation system:

- Vision-Language Model fine-tuning
- LoRA parameter-efficient adaptation
- Multimodal dataset preprocessing
- GPU/CPU pipeline optimization
- Training and checkpoint recovery
- Navigation metric implementation
- IsaacSim integration
- Closed-loop robot evaluation
- Simulation and camera debugging
- Experimental analysis

One of the main lessons from the project was that **closed-loop embodied AI performance depends not only on model accuracy, but also on the consistency of the entire perception-simulation-action pipeline**.

The substantial performance improvement after correcting the IsaacSim observation pipeline demonstrated how deployment mismatches can dominate the apparent performance of a trained navigation model.

---

## Acknowledgment

This project was developed as part of the **Deep Learning course at Korea Aerospace University (KAU), taught by Prof. Young-Sik Choi**.

The course provided the academic framework and project environment for this work.

---

## Disclaimer

This repository is maintained as a personal portfolio representation of my Week 4 implementation and experimental work.

The VLNVerse dataset and course materials are not redistributed in this repository.