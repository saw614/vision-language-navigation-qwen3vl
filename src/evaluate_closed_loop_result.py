#!/usr/bin/env python3
"""
week4_closed_loop_eval_robot_view_fixed.py

Closed-loop evaluation for VLNVerse + Isaac Sim.
Fixes:
1. Isaac viewport uses EgoCamera / robot view.
2. Camera starts at episode start_position.
3. Camera looks forward horizontally, not top-down.
4. Avoids xformOp rotateXYZ already exists crash by using one transform op.
"""

import os
import sys
import json
import math
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["OV_GPU_CHECK_SKIP"] = "1"
os.environ["CARB_APP_RTX_VERIFY_DRIVER_VERSION"] = "0"

# Isaac path
ISAAC_ROOT = os.path.expanduser("~/Downloads")
sys.path.insert(0, ISAAC_ROOT)
sys.path.insert(0, os.path.join(ISAAC_ROOT, "kit/python/lib/python3.10/site-packages"))
sys.path.insert(0, os.path.join(ISAAC_ROOT, "exts"))


VALID_ACTIONS = [
    "Move forward 25cm",
    "Turn right 15 degree",
    "Turn left 15 degree",
    "Stop",
]

ACTION_TO_VELOCITY = {
    "Move forward 25cm":    {"vx": 1.0, "vy": 0.0, "vz": 0.0,    "duration": 0.25},
    "Turn right 15 degree": {"vx": 0.0, "vy": 0.0, "vz": -1.047, "duration": 0.25},
    "Turn left 15 degree":  {"vx": 0.0, "vy": 0.0, "vz":  1.047, "duration": 0.25},
    "Stop":                 {"vx": 0.0, "vy": 0.0, "vz": 0.0,    "duration": 0.0},
}

DEFAULT_ACTION = "Move forward 25cm"


# =========================
# Metrics
# =========================

def calc_ndtw(pred_traj, gt_traj, threshold=3.0):
    if len(pred_traj) == 0 or len(gt_traj) == 0:
        return 0.0

    pred_xy = np.array(pred_traj)
    gt_xy = np.array(gt_traj)

    N, M = len(pred_xy), len(gt_xy)
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dist = np.linalg.norm(pred_xy[i - 1] - gt_xy[j - 1])
            dtw[i, j] = dist + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )

    return float(np.exp(-dtw[N, M] / (threshold * len(gt_xy))))


def calc_trajectory_length(coords):
    if len(coords) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(coords)):
        total += math.sqrt(
            (coords[i][0] - coords[i - 1][0]) ** 2 +
            (coords[i][1] - coords[i - 1][1]) ** 2
        )
    return total


def evaluate_episode(pred_traj, gt_traj, goal_pos):
    goal_x, goal_y = goal_pos[0], goal_pos[1]

    if len(pred_traj) == 0:
        return {
            "SR": 0,
            "OSR": 0,
            "SPL": 0.0,
            "nDTW": 0.0,
            "Goal Dist": float("inf"),
            "TL": 0.0,
        }

    filtered = [pred_traj[0]]
    for p in pred_traj[1:]:
        if p != filtered[-1]:
            filtered.append(p)

    last_x, last_y = filtered[-1]

    goal_dist = math.sqrt(
        (goal_x - last_x) ** 2 +
        (goal_y - last_y) ** 2
    )

    sr = 1 if goal_dist <= 3.0 else 0

    osr = 0
    for x, y in filtered:
        d = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
        if d <= 3.0:
            osr = 1
            break

    tl = calc_trajectory_length(filtered)
    tl_gt = calc_trajectory_length(gt_traj)

    spl = sr * (tl_gt / max(tl, tl_gt)) if tl > 0 and tl_gt > 0 else 0.0
    ndtw = calc_ndtw(filtered, gt_traj)

    return {
        "SR": sr,
        "OSR": osr,
        "SPL": round(spl, 4),
        "nDTW": round(ndtw, 4),
        "Goal Dist": round(goal_dist, 4),
        "TL": round(tl, 4),
    }


# =========================
# Model loading
# =========================

def load_model(adapter_path):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    base_model_path = "/home/ad10/2026_deeplearning/deep26_week4/models/Qwen3-VL-2B-Instruct"

    print("Loading processor from base model...", flush=True)
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading base model...", flush=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )
    model.eval()

    print("Model loaded successfully.", flush=True)
    return model, processor


# =========================
# Inference
# =========================

def predict_action(model, processor, instruction, img_prev, img_curr):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "You are a navigation action classifier.\n"
                "Given the full navigation instruction and two egocentric images, "
                "predict the next action.\n"
                "You must output exactly one action and nothing else.\n"
                "Valid actions:\n"
                "Move forward 25cm\n"
                "Turn right 15 degree\n"
                "Turn left 15 degree\n"
                "Stop\n"
                "Do not explain.\n"
                "Do not output multiple actions.\n"
                "Do not add extra words."
            )}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_prev},
                {"type": "image", "image": img_curr},
                {"type": "text", "text": f"Instruction:\n{instruction}\n\nAnswer:"},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info([messages])

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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[0, input_len:]

    raw = processor.tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()

    if raw in VALID_ACTIONS:
        return raw, raw, False

    return DEFAULT_ACTION, raw, True


# =========================
# Episode data
# =========================

def load_episode_data(eval_json_path, fine_val_gz, fine_train_gz):
    with open(eval_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "episodes" in data:
        raw_episodes = data["episodes"]
    else:
        raw_episodes = data

    all_episodes = {}

    with gzip.open(fine_val_gz, "rt", encoding="utf-8") as f:
        val_data = json.load(f)

    for ep in val_data["episodes"]:
        all_episodes[ep["episode_id"]] = ep

    with gzip.open(fine_train_gz, "rt", encoding="utf-8") as f:
        train_data = json.load(f)

    for ep in train_data["episodes"]:
        all_episodes[ep["episode_id"]] = ep

    print(f"Total episodes in both splits: {len(all_episodes)}", flush=True)

    episodes = []

    for ep in raw_episodes:
        episode_id = ep["episode_id"]
        ep_data = all_episodes.get(episode_id)

        if ep_data is None:
            print(f"WARNING: Episode {episode_id} not found in any split", flush=True)
            continue

        episodes.append({
            "episode_id": episode_id,
            "scene_name": ep.get("scene_name", ep.get("scan", ep.get("scene_id", "unknown_scene"))),
            "split": ep.get("split", "val"),
            "instruction": ep_data["instruction"]["instruction_text"],
            "start_position": ep_data["start_position"],
            "start_rotation": ep_data["start_rotation"],
            "reference_path": ep_data["reference_path"],
            "goal_position": ep_data["goals"]["position"],
        })

    return episodes


# =========================
# Isaac Sim runner
# =========================

def run_episode_in_isaac(
    model,
    processor,
    episode,
    scene_dir,
    work_dir,
    timeout_steps=400,
    headless=False,
):
    if "--/rtx/verifyDriverVersion/enabled=false" not in sys.argv:
        sys.argv.append("--/rtx/verifyDriverVersion/enabled=false")
    if "--/rtx-defaults/verifyDriverVersion/enabled=false" not in sys.argv:
        sys.argv.append("--/rtx-defaults/verifyDriverVersion/enabled=false")

    from isaacsim import SimulationApp

    config = {
        "renderer": "RayTracedLighting",
        "headless": headless,
        "/omni/client/cloud/enabled": False,
        "/persistent/app/omniverse/content_browser/show_cloud_assets": False,
        "/rtx/verifyDriverVersion/enabled": False,
        "/rtx-defaults/verifyDriverVersion/enabled": False,
    }

    simulation_app = SimulationApp(config)

    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import define_prim
    from PIL import Image, ImageDraw
    from pxr import UsdGeom, Gf

    import omni.replicator.core as rep

    scene_name = episode["scene_name"]

    scene_root = os.path.abspath(os.path.join(scene_dir, scene_name))
    usd_path = os.path.join(scene_root, "start_result_navigation.usd")

    print(f"Scene root: {scene_root}", flush=True)
    print(f"USD path: {usd_path}", flush=True)
    print(f"USD exists: {os.path.exists(usd_path)}", flush=True)

    start_pos = episode["start_position"]
    start_rot = episode["start_rotation"]
    instruction = episode["instruction"]
    goal_pos = episode["goal_position"]
    gt_traj = [[p[0], p[1]] for p in episode["reference_path"]]

    print("\n" + "=" * 50, flush=True)
    print(f"Episode: {episode['episode_id']}", flush=True)
    print(f"Scene:   {scene_name}", flush=True)
    print(f"Instruction: {instruction[:100]}...", flush=True)
    print("=" * 50, flush=True)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 200.0,
        rendering_dt=8.0 / 200.0,
    )

    old_cwd = os.getcwd()
    os.chdir(scene_root)
    print(f"Changed cwd to: {os.getcwd()}", flush=True)

    prim = define_prim("/World/Ground", "Xform")
    prim.GetReferences().AddReference(usd_path)
    print("USD loaded", flush=True)

    os.chdir(old_cwd)
    print(f"Restored cwd: {os.getcwd()}", flush=True)

    stage = simulation_app.context.get_stage()

    # -------------------------
    # Ego Camera
    # -------------------------
    camera_path = "/World/EgoCamera"
    cam_prim = UsdGeom.Camera.Define(stage, camera_path)
    cam_prim.CreateFocalLengthAttr(10.0)
    cam_prim.CreateHorizontalApertureAttr(20.0)
    cam_prim.CreateVerticalApertureAttr(20.0)

    camera_xform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))

    # IMPORTANT:
    # Use only one TransformOp.
    # This avoids:
    # xformOp 'rotateXYZ' already exists
    camera_xform.ClearXformOpOrder()
    camera_transform_op = camera_xform.AddTransformOp()

    # Make Isaac viewport show robot view
    if not headless:
        try:
            import omni.kit.viewport.utility as vp_utils
            viewport = vp_utils.get_active_viewport()
            viewport.camera_path = camera_path
            print(f"Viewport attached to EgoCamera: {camera_path}", flush=True)
        except Exception as e:
            print(f"WARNING: Could not attach viewport to EgoCamera: {e}", flush=True)

    world.reset()

    # -------------------------
    # Start point = spawn point
    # -------------------------
    robot_pos = np.array(start_pos, dtype=np.float64)

    # keep z from dataset
    robot_pos[2] = float(start_pos[2])

    # For now, use yaw = 0 as safe default.
    # The camera will face +Y initially.
    # Turn actions will update this yaw.
    robot_yaw = 0.0

    print("start_pos:", start_pos, flush=True)
    print("start_rot:", start_rot, flush=True)
    print("robot_pos:", robot_pos.tolist(), flush=True)
    print("robot_yaw_deg:", math.degrees(robot_yaw), flush=True)

    pred_traj = [[float(robot_pos[0]), float(robot_pos[1])]]
    full_traj = [[float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])]]

    action_records = []
    invalid_count = 0
    step = 0

    # Replicator render product from EgoCamera
    render_product = rep.create.render_product(camera_path, (256, 256))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    prev_image = None
    curr_image = None

    def update_camera():
        """
        First-person robot camera.

        Uses SetLookAt matrix:
        - eye = robot position + camera height
        - target = eye + forward direction
        - up = world Z

        This prevents top-down camera problem.
        """
        cam_height = 1.2

        eye = Gf.Vec3d(
            float(robot_pos[0]),
            float(robot_pos[1]),
            float(robot_pos[2] + cam_height),
        )

        forward = Gf.Vec3d(
            float(math.sin(-robot_yaw)),
            float(math.cos(-robot_yaw)),
            0.0,
        )

        target = Gf.Vec3d(
            float(eye[0] + forward[0]),
            float(eye[1] + forward[1]),
            float(eye[2] + forward[2]),
        )

        up = Gf.Vec3d(0.0, 0.0, 1.0)

        # SetLookAt creates view matrix.
        # Camera prim needs world transform, so use inverse.
        view_matrix = Gf.Matrix4d().SetLookAt(eye, target, up)
        camera_world_matrix = view_matrix.GetInverse()

        camera_transform_op.Set(camera_world_matrix)

    def capture_image():
        world.step(render=True)
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False)

        data = rgb_annotator.get_data()

        if data is None or len(data) == 0:
            return None

        img_array = data[:, :, :3]
        return Image.fromarray(img_array.astype(np.uint8))

    def execute_action(action):
        nonlocal robot_pos, robot_yaw

        vel = ACTION_TO_VELOCITY[action]
        duration = vel["duration"]
        physics_steps = int(duration * 200)

        for _ in range(max(physics_steps, 1)):
            robot_pos[0] += vel["vx"] * math.sin(-robot_yaw) * (1.0 / 200.0)
            robot_pos[1] += vel["vx"] * math.cos(-robot_yaw) * (1.0 / 200.0)
            robot_yaw += vel["vz"] * (1.0 / 200.0)
            world.step(render=False)

        update_camera()

        pred_traj.append([float(robot_pos[0]), float(robot_pos[1])])
        full_traj.append([float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])])

    # Warmup
    update_camera()
    for _ in range(150):
        world.step(render=True)

    curr_image = capture_image()
    prev_image = curr_image

    # Save first debug image
    if curr_image is not None:
        debug_path = os.path.join(work_dir, f"debug_first_view_{episode['episode_id']}.png")
        os.makedirs(work_dir, exist_ok=True)
        curr_image.save(debug_path)
        print(f"Saved first robot view: {debug_path}", flush=True)

    done = False
    stop_requested = False

    while step < timeout_steps and not done:
        if prev_image is not None and curr_image is not None:
            action, raw_output, is_invalid = predict_action(
                model,
                processor,
                instruction,
                prev_image,
                curr_image,
            )
        else:
            action = DEFAULT_ACTION
            raw_output = ""
            is_invalid = True

        if is_invalid:
            invalid_count += 1

        print(
            f"  step {step:3d} | action: {action} | invalid: {is_invalid}",
            flush=True,
        )

        action_records.append({
            "step": step,
            "action": action,
            "raw_output": raw_output,
            "invalid": bool(is_invalid),
            "root": [
                float(robot_pos[0]),
                float(robot_pos[1]),
                float(robot_pos[2]),
            ],
            "yaw_deg": float(math.degrees(robot_yaw)),
        })

        if action == "Stop":
            done = True
            stop_requested = True
            break

        prev_image = curr_image

        execute_action(action)

        curr_image = capture_image()

        # Save annotated frame
        if curr_image is not None:
            frame_dir = os.path.join(work_dir, "frames_" + episode["episode_id"])
            os.makedirs(frame_dir, exist_ok=True)

            annotated = curr_image.copy().resize((512, 512))
            draw = ImageDraw.Draw(annotated)

            draw.rectangle([0, 0, 512, 55], fill=(0, 0, 0))
            draw.text((8, 8), "TASK: " + instruction[:65], fill=(255, 255, 0))

            draw.rectangle([0, 462, 512, 512], fill=(0, 0, 0))
            draw.text((8, 468), f"Step {step:03d} | {action}", fill=(0, 255, 0))

            annotated.save(os.path.join(frame_dir, f"step_{step:04d}.png"))

        step += 1

    # Save action log / trajectory
    episode_work_dir = os.path.join(work_dir, episode["episode_id"])
    os.makedirs(episode_work_dir, exist_ok=True)

    action_log_path = os.path.join(episode_work_dir, "week4_smoke_action_log.json")

    action_log = {
        "episode_id": episode["episode_id"],
        "scene_name": episode["scene_name"],
        "split": episode["split"],
        "instruction": instruction,
        "start_position": start_pos,
        "start_rotation": start_rot,
        "goal_position": goal_pos,
        "actions": action_records,
        "trajectory": [
            {
                "frame": i,
                "root": p,
            }
            for i, p in enumerate(full_traj)
        ],
        "stop_requested": stop_requested,
    }

    with open(action_log_path, "w") as f:
        json.dump(action_log, f, indent=2)

    print(f"Saved action log: {action_log_path}", flush=True)

    # Evaluate
    metrics = evaluate_episode(pred_traj, gt_traj, goal_pos)

    metrics["episode_id"] = episode["episode_id"]
    metrics["scene_name"] = episode["scene_name"]
    metrics["split"] = episode["split"]
    metrics["steps"] = step
    metrics["invalid_count"] = invalid_count
    metrics["invalid_rate"] = round(invalid_count / max(step, 1), 4)
    metrics["stop_requested"] = stop_requested
    metrics["action_log_path"] = action_log_path

    print(f"\nResults for {episode['episode_id']}:", flush=True)
    print(
        f"  SR={metrics['SR']} OSR={metrics['OSR']} "
        f"SPL={metrics['SPL']} nDTW={metrics['nDTW']} "
        f"Goal Dist={metrics['Goal Dist']}",
        flush=True,
    )

    simulation_app.close()

    return metrics


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--adapter_path", type=str, required=True)

    parser.add_argument(
        "--eval_json",
        type=str,
        default="vlnverse_closed_loop_eval_20episodes.json",
    )

    parser.add_argument(
        "--scene_dir",
        type=str,
        default="data/VLNVerse_scene",
    )

    parser.add_argument(
        "--fine_val_gz",
        type=str,
        default="data/VLNVerse_data/raw_data/final_splits/fine_val.json.gz",
    )

    parser.add_argument(
        "--fine_train_gz",
        type=str,
        default="data/VLNVerse_data/raw_data/final_splits/fine_train.json.gz",
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default="results/robot_view_eval",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--timeout_steps",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--episode_idx",
        type=int,
        default=None,
        help="Run only one episode by index.",
    )

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    print("Loading episode data...", flush=True)

    episodes = load_episode_data(
        args.eval_json,
        args.fine_val_gz,
        args.fine_train_gz,
    )

    print(f"Total episodes to evaluate: {len(episodes)}", flush=True)

    if args.episode_idx is not None:
        if args.episode_idx < 0 or args.episode_idx >= len(episodes):
            raise ValueError(
                f"episode_idx {args.episode_idx} is out of range. "
                f"Valid range: 0 to {len(episodes) - 1}"
            )

        episodes = [episodes[args.episode_idx]]
        print(f"Running only episode index: {args.episode_idx}", flush=True)

    print("First episodes:", flush=True)
    for ep in episodes[:5]:
        print(ep["episode_id"], ep["scene_name"], ep["split"], flush=True)

    print("Before load_model", flush=True)
    model, processor = load_model(args.adapter_path)
    print("After load_model", flush=True)

    print("Starting evaluation loop...", flush=True)

    all_metrics = []

    for i, episode in enumerate(episodes):
        scene_path = os.path.join(
            args.scene_dir,
            episode["scene_name"],
        )

        if not os.path.exists(scene_path):
            print(f"WARNING: Scene not found: {scene_path}, skipping...", flush=True)
            continue

        print(f"\n[{i + 1}/{len(episodes)}] Running: {episode['episode_id']}", flush=True)

        try:
            metrics = run_episode_in_isaac(
                model,
                processor,
                episode,
                args.scene_dir,
                args.work_dir,
                timeout_steps=args.timeout_steps,
                headless=args.headless,
            )

            all_metrics.append(metrics)

            results_path = os.path.join(args.work_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(all_metrics, f, indent=2)

            print(f"Saved results to {results_path}", flush=True)

        except Exception as e:
            print(f"ERROR in {episode['episode_id']}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    if all_metrics:
        df = pd.DataFrame(all_metrics)

        print("\n" + "=" * 70)
        print("CLOSED-LOOP EVALUATION RESULTS")
        print("=" * 70)

        print("\n--- Overall ---")
        for metric in ["SR", "OSR", "SPL", "nDTW", "Goal Dist"]:
            print(f"  {metric}: {df[metric].mean():.4f}")

        print(f"  Invalid Rate: {df['invalid_rate'].mean():.4f}")

        val_df = df[df["split"] == "val"]
        if len(val_df) > 0:
            print(f"\n--- Val episodes ({len(val_df)}) ---")
            for metric in ["SR", "OSR", "SPL", "nDTW", "Goal Dist"]:
                print(f"  {metric}: {val_df[metric].mean():.4f}")

        train_df = df[df["split"] == "train"]
        if len(train_df) > 0:
            print(f"\n--- Train episodes ({len(train_df)}) ---")
            for metric in ["SR", "OSR", "SPL", "nDTW", "Goal Dist"]:
                print(f"  {metric}: {train_df[metric].mean():.4f}")

        print("\n--- Per Episode ---")
        print(
            df[
                [
                    "episode_id",
                    "split",
                    "SR",
                    "OSR",
                    "SPL",
                    "nDTW",
                    "Goal Dist",
                    "invalid_rate",
                    "stop_requested",
                ]
            ].to_string(index=False)
        )

        csv_path = os.path.join(args.work_dir, "eval_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}", flush=True)

    print("\nEvaluation complete!", flush=True)


if __name__ == "__main__":
    main()
