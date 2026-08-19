# Deep Learning Term Project Week 4 — deep26
See the full README in the submitted report.
# Vision-Language Navigation with Qwen3-VL + LoRA

## Overview

This project implements a Vision-Language Navigation (VLN) system using Qwen3-VL-2B-Instruct and LoRA fine-tuning. The model receives two consecutive RGB images and a natural language navigation instruction, then predicts one of four navigation actions.

Actions:
- Move forward 25cm
- Turn left 15 degree
- Turn right 15 degree
- Stop

---

## Model Architecture

- Base Model: Qwen3-VL-2B-Instruct
- Vision Encoder: Qwen3-VL Vision Transformer
- Language Model: Qwen3-2B Decoder
- LoRA Target Layers:
  - q_proj
  - v_proj
- LoRA Configuration:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05

Input Pipeline:

Instruction + Image(t-1) + Image(t)
↓
Qwen3-VL Processor
↓
Vision + Language Tokens
↓
Qwen3-VL Decoder
↓
LoRA Fine-tuning
↓
Action Prediction

---

## Dataset

Dataset: VLNVerse Fine-Grained Navigation

Split Strategy:
- Episode-level split
- Train: 90%
- Validation: 10%

Statistics:
- Total Episodes: 1,500
- Train Episodes: 1,350
- Validation Episodes: 150
- Total Samples: 6,582

Action Distribution:

| Action | Count |
|----------|----------|
| Move Forward | 4356 |
| Turn Left | 1111 |
| Turn Right | 974 |
| Stop | 141 |

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Epochs | 3 |
| Optimizer | AdamW |
| Scheduler | Cosine with Warmup |
| Warmup Ratio | 0.03 |
| Precision | bfloat16 |

Hardware:
- NVIDIA RTX 4090

---

## Baseline Results

| Epoch | Train Loss | Val Loss | Val Accuracy |
|---------|---------|---------|---------|
| 1 | 0.1076 | 0.0914 | 63.20% |
| 2 | 0.0897 | 0.0885 | **63.50%** |
| 3 | 0.0806 | 0.0883 | 62.72% |

Best Model:
---

## Closed-Loop Evaluation (IsaacSim)

Environment:
- IsaacSim 4.5.0

Metrics:
- SR (Success Rate)
- OSR (Oracle Success Rate)
- SPL
- nDTW
- Goal Distance
- Invalid Output Rate

Results:

| Metric | Value |
|----------|----------|
| SR | 0.0215 |
| OSR | 0.1207 |
| SPL | 0.158 |
| nDTW | 0.2620 |
| Goal Distance | 25.3 m |
| Invalid Output Rate | 0.00014 |

---

## Technical Challenges

- Label masking bug caused by multimodal token expansion
- Validation pipeline bug causing incorrect accuracy
- GPU bottleneck from Qwen3-VL vision preprocessing
- VRAM Out-of-Memory (OOM) during training
- Checkpoint resume issues
- Learning-rate scheduler recovery issues
- Disk space management for large checkpoints

---

## Ablation Study

| Setting | LoRA Rank | Alpha | Trainable Params |
|----------|----------|----------|----------|
| Baseline | 16 | 32 | ~1.6M |
| Ablation | 8 | 16 | ~0.8M |

The ablation experiment showed similar initial training behavior while reducing trainable parameters by approximately 50%.

---

## Tools and Frameworks

- PyTorch
- Transformers
- PEFT (LoRA)
- Weights & Biases (W&B)
- IsaacSim 4.5.0

---

## Team Members

- 최길웅 – Data preprocessing and dataset construction
- 송예원 – Model architecture and LoRA configuration
- 허가연 – Validation and ablation analysis
- 서뗑기툰 – Model training, debugging, IsaacSim closed-loop evaluation, and performance analysis

---

## Project Outcomes

This project demonstrated the complete pipeline of Vision-Language Navigation, including data preprocessing, multimodal input construction, LoRA fine-tuning, model evaluation, debugging, and simulator-based validation.


## Additional Improvements After Evaluation Environment Fixes

During the final stage of the project, several issues in the IsaacSim closed-loop evaluation environment were identified and corrected. These changes significantly improved navigation performance by ensuring that the model received observations consistent with the training data.

### Major Causes of Performance Improvement

| Fix                       | Description                                                                                                                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Viewport / Ego Camera Fix | The model was modified to receive the actual robot egocentric view instead of an incorrect scene viewport. This allowed the model to correctly perceive the surrounding environment and predict appropriate actions. |
| Start Point = Spawn Point | The robot was spawned at the exact start position defined in the dataset, ensuring consistency between the training data and evaluation environment.                                                                 |
| Camera Height Adjustment  | The camera height was changed to **z + 1.2**, providing a viewpoint closer to human/robot eye level. This resolved cases where the camera observed mostly the floor or ceiling.                                      |
| Top-Down View Issue Fix   | The camera orientation was corrected so that it faced the robot's forward direction rather than pointing downward toward the floor.                                                                                  |
| xformOp Conflict Fix      | The `rotateXYZ already exists` transform error was resolved, allowing camera transforms to update reliably at every simulation step and preventing camera-related crashes.                                           |

### Performance Comparison
### The below comparison tables used to run the evaluate_closed_loop_result.py which is the updated version by fixing the above problems.
| Metric              | Before Fix | After Fix               |
| ------------------- | ---------- | ----------------------- |
| SR                  | 0.0215     | **0.350**               |
| OSR                 | 0.1207     | **0.450**               |
| SPL                 | 0.158      | **0.000**               |
| nDTW                | 0.2620     | N/A                     |
| Goal Distance (m)   | 25.3       | **5.39**                |
| Invalid Output Rate | 0.00014    | **0.00000**             |

### Discussion

The results indicate that a substantial portion of the performance degradation originated from issues in the simulation and perception pipeline rather than the navigation model itself. After correcting the camera viewpoint, spawn location, camera orientation, and transform update logic, the model was able to receive observations that more closely matched the training distribution. As a result, Success Rate (SR) increased from **2.5%** to **35.0%**, Oracle Success Rate (OSR) increased from **12.1%** to **45.0%**, and the average Goal Distance decreased from **25.3 m** to **5.39 m**.

These findings highlight the importance of maintaining consistency between the training environment and the deployment/evaluation environment in Vision-Language Navigation systems.
## Closed-Loop Evaluation: Timeout Step Analysis

### Table 1. Closed-Loop Evaluation Results (Timeout = 50 Steps)

| Metric            | Value |
| ----------------- | ----: |
| SR                | 0.395 |
| OSR               | 0.479 |
| nDTW              | 0.500 |
| Goal Distance (m) | 4.480 |
| Invalid Rate      | 0.000 |

### Table 2. Closed-Loop Evaluation Results (Timeout = 100 Steps)

| Metric            | Value |
| ----------------- | ----: |
| SR                | 0.393 |
| OSR               | 0.579 |
| SPL               | 0.287 |
| nDTW              | 0.510 |
| Goal Distance (m) | 5.448 |
| Invalid Rate      | 0.000 |

###Conclusion

To analyze the effect of episode length, the closed-loop evaluation was conducted using timeout limits of 50 and 100 steps. The results show that both settings produced similar Success Rates (SR), while the longer timeout increased Oracle Success Rate (OSR) and slightly improved nDTW. This indicates that additional steps allowed the robot to approach the target more frequently and follow trajectories that were closer to the ground-truth path.

An interesting observation is that OSR increased considerably (0.479 → 0.579) while SR remained almost unchanged. This suggests that the robot was often able to reach the vicinity of the target but was unable to successfully complete the navigation task. In other words, many failures were caused by stopping behavior or final navigation decisions rather than an inability to reach the target area.
## Model Checkpoint

Best LoRA checkpoint:

https://drive.google.com/drive/folders/1X05hp9FwUx2uLjom8P_SHKOmtm8CmkuC?usp=drive_link

Contents:
- adapter_model.safetensors
- adapter_config.json
- tokenizer files
