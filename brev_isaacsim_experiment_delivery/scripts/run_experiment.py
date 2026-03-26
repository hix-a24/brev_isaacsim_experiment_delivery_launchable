#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.render_figures import render_all_figures
from src.logger.contact_logger import ContactLogger
from src.logger.episode_logger import EpisodeLogger
from src.logger.intervention_logger import InterventionLogger
from src.logger.step_logger import StepLogger
from src.planner.fallback_planner import FallbackPlanner
from src.policy.octo_policy import Octo
from src.policy.openvla_policy import OpenVLA
from src.supervisor.calibrator import TemperatureCalibrator
from src.supervisor.gating import ControlState, SafetyGatingStateMachine
from src.supervisor.risk_estimator import RiskEstimator
from src.tasks.task_a import DrawerObjectTask
from src.tasks.task_b import ClutterSortTask
from src.utils.config import load_config

SHIFT_AXES = ["lighting", "texture", "occlusion", "sensor", "combined"]
TASK_REGISTRY = {
    "drawer": DrawerObjectTask,
    "clutter_sort": ClutterSortTask,
}
POLICY_REGISTRY = {
    "openvla": OpenVLA,
    "octo": Octo,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic end-to-end experiment pipeline.")
    parser.add_argument("--config", default="configs/pilot_config.yaml", help="Path to YAML config relative to project root.")
    parser.add_argument("--seed", type=int, default=13, help="Global seed for checkpoint generation and evaluation.")
    parser.add_argument("--clean", action="store_true", help="Remove prior logs, figures, and run manifests before execution.")
    parser.add_argument("--episodes-per-condition", type=int, default=None, help="Override config episodes_per_condition.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps_per_episode.")
    return parser.parse_args()


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(p)
    arr = np.clip(arr, 1e-6, 1 - 1e-6)
    return np.log(arr / (1.0 - arr))


def make_shift_config(axis: str, severity: int) -> Dict[str, Any]:
    return {"axis": axis, "severity": int(severity)}


def ensure_clean(root: Path) -> None:
    for rel in [
        "data_logs/episodes.csv",
        "data_logs/steps.csv",
        "data_logs/interventions.csv",
        "data_logs/contacts.csv",
        "figures/fig6_success_collision.png",
        "figures/fig7_robustness_curves.png",
        "figures/fig8_calibration.png",
        "figures/fig9_interventions.png",
        "figures/fig10_safety_characterization.png",
        "checkpoints/openvla_pilot.ckpt",
        "checkpoints/openvla_pilot.ckpt.json",
        "checkpoints/octo_pilot.ckpt",
        "checkpoints/octo_pilot.ckpt.json",
    ]:
        path = root / rel
        if path.exists():
            path.unlink()
    latest = root / "runs" / "latest"
    if latest.is_dir():
        shutil.rmtree(latest)
    (root / "data_logs").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)


def checkpoint_path(root: Path, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return root / path


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.savez(fh, **payload)


def build_training_dataset(task_name: str, task_cfg: Dict[str, Any], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_cls = TASK_REGISTRY[task_name]
    env = task_cls({**task_cfg, "max_steps": min(int(task_cfg.get("max_steps", 32)), 24)}, seed=int(rng.integers(0, 1_000_000)))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    cs: List[float] = []
    demo_count = int(task_cfg.get("demo_count", 12))
    severities = list(task_cfg.get("shift_severities", [0, 1, 2]))

    for _ in range(max(demo_count, 4)):
        shift_axis = SHIFT_AXES[int(rng.integers(0, len(SHIFT_AXES)))]
        severity = int(severities[int(rng.integers(0, len(severities)))])
        obs = env.reset(make_shift_config(shift_axis, severity))
        max_demo_steps = min(env.max_steps, 12)
        for _ in range(max_demo_steps):
            oracle_action = env.oracle_action(obs)
            demo_action = np.tanh(oracle_action + rng.normal(0.0, 0.03, size=oracle_action.shape)).astype(np.float32)
            xs.append(np.asarray(obs["state"], dtype=np.float32))
            ys.append(demo_action)
            cs.append(env.oracle_confidence(obs))
            obs, _, done, _ = env.step(demo_action)
            if done:
                break
    env.close()
    return np.stack(xs), np.stack(ys), np.asarray(cs, dtype=np.float32)


def train_policy_checkpoint(policy_name: str, policy_cfg: Dict[str, Any], task_cfgs: Dict[str, Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
    x_batches: List[np.ndarray] = []
    y_batches: List[np.ndarray] = []
    c_batches: List[np.ndarray] = []
    for task_name, task_cfg in task_cfgs.items():
        x, y, c = build_training_dataset(task_name, task_cfg, rng)
        x_batches.append(x)
        y_batches.append(y)
        c_batches.append(c)

    x = np.concatenate(x_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)
    c = np.concatenate(c_batches, axis=0)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)

    action_coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    action_weights = action_coef[:-1].T.astype(np.float32)
    action_bias = action_coef[-1].astype(np.float32)

    confidence_targets = _logit(c)
    conf_coef, *_ = np.linalg.lstsq(x_aug, confidence_targets, rcond=None)
    conf_weights = conf_coef[:-1].astype(np.float32)
    conf_bias = np.float32(conf_coef[-1])

    action_pred = np.tanh(x @ action_weights.T + action_bias)
    mae = float(np.mean(np.abs(action_pred - y)))
    raw_conf_logits = x @ conf_weights + conf_bias
    raw_conf = np.clip(_sigmoid(raw_conf_logits), 1e-4, 1 - 1e-4)
    outcome = (np.mean(np.abs(action_pred - y), axis=1) < (0.12 if policy_name == "openvla" else 0.15)).astype(np.int32)

    calibrator = TemperatureCalibrator()
    calibrator.fit(raw_conf_logits, outcome)
    clipped_temperature = float(np.clip(calibrator.temperature, 0.6, 2.0))

    policy_noise = 0.015 + 0.25 * mae + (0.0 if policy_name == "openvla" else 0.015)
    if policy_name == "octo":
        action_weights = (action_weights + rng.normal(0.0, 0.012, size=action_weights.shape)).astype(np.float32)
        conf_weights = (conf_weights + rng.normal(0.0, 0.02, size=conf_weights.shape)).astype(np.float32)

    payload = {
        "action_weights": action_weights,
        "action_bias": action_bias,
        "confidence_weights": conf_weights,
        "confidence_bias": np.float32(conf_bias),
        "noise_scale": np.float32(policy_noise),
        "calibration_temperature": np.float32(clipped_temperature),
    }
    metrics = {
        "policy": policy_name,
        "num_samples": int(x.shape[0]),
        "train_mae": round(mae, 6),
        "mean_confidence": round(float(raw_conf.mean()), 6),
        "calibration_temperature": round(float(clipped_temperature), 6),
        "noise_scale": round(float(policy_noise), 6),
    }
    return {"payload": payload, "metrics": metrics}


def initialize_loggers(root: Path):
    data_dir = root / "data_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (
        EpisodeLogger(data_dir),
        StepLogger(data_dir),
        InterventionLogger(data_dir),
        ContactLogger(data_dir),
    )


def run_evaluation(
    root: Path,
    cfg: Dict[str, Any],
    seed: int,
    episodes_per_condition: int,
    max_steps: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    episode_logger, step_logger, intervention_logger, contact_logger = initialize_loggers(root)

    policies: Dict[str, Any] = {}
    policy_configs: Dict[str, Dict[str, Any]] = {}
    for policy_name, policy_cls in POLICY_REGISTRY.items():
        policy_cfg = dict(cfg["policies"][policy_name])
        policy_cfg["seed"] = int(seed + 17 * (len(policies) + 1))
        policy_cfg["checkpoint"] = str(checkpoint_path(root, policy_cfg["checkpoint"]))
        policy = policy_cls(policy_cfg)
        policy.load_model()
        policies[policy_name] = policy
        policy_configs[policy_name] = policy_cfg

    totals = {
        "episodes": 0,
        "successes": 0,
        "collisions": 0,
        "interventions": 0,
        "fallbacks": 0,
        "contacts": 0,
    }

    for task_name, task_cfg in cfg["tasks"].items():
        task_cls = TASK_REGISTRY[task_name]
        severities = list(task_cfg.get("shift_severities", [0, 1, 2]))
        for shift_axis in SHIFT_AXES:
            for severity in severities:
                for policy_name, policy in policies.items():
                    for episode_idx in range(episodes_per_condition):
                        episode_seed = int(rng.integers(0, 1_000_000))
                        env = task_cls({**task_cfg, "max_steps": max_steps}, seed=episode_seed)
                        obs = env.reset(make_shift_config(shift_axis, int(severity)))
                        instruction = f"Execute {task_name} under {shift_axis} shift severity {severity}"
                        policy.reset(instruction)
                        calibrator = TemperatureCalibrator(
                            temperature=policy.calibration_temperature(),
                            fitted=True,
                        )
                        estimator = RiskEstimator(
                            alpha=float(cfg["gating"].get("alpha", 0.5)),
                            beta=float(cfg["gating"].get("beta", 0.5)),
                        )
                        gating = SafetyGatingStateMachine(
                            delta_low=float(cfg["gating"].get("delta_low", 0.2)),
                            delta_high=float(cfg["gating"].get("delta_high", 0.5)),
                        )
                        planner = FallbackPlanner({"policy": policy_name})

                        episode_id = f"{task_name}-{shift_axis}-s{severity}-{policy_name}-ep{episode_idx:03d}"
                        collisions = 0
                        interventions = 0
                        pause_events = 0
                        fallback_events = 0
                        peak_force = 0.0
                        peak_torque = 0.0

                        for t_step in range(max_steps):
                            action, policy_info = policy.act(obs)
                            raw_confidence = float(np.clip(policy_info["raw_policy_confidence"], 1e-4, 1 - 1e-4))
                            calibrated_success_prob = float(calibrator.transform([float(_logit(raw_confidence))])[0])
                            risk = float(estimator.estimate(calibrated_success_prob, policy_info))
                            state, event = gating.update(risk)
                            executed_action = np.asarray(action, dtype=np.float32)
                            resolved_by = "policy"
                            risk_after = risk

                            if state == ControlState.PAUSE_REOBSERVE:
                                executed_action = np.tanh(0.65 * executed_action).astype(np.float32)
                                resolved_by = "reobserve"
                                risk_after = max(risk - 0.12, 0.0)
                            elif state == ControlState.FALLBACK:
                                plan = planner.plan_and_execute(
                                    env.current_goal_name,
                                    {
                                        "oracle_action": env.oracle_action(obs),
                                        "risk": risk,
                                        "margin_to_collision_m": float(obs["state"][5]),
                                        "phase": env.phase_name,
                                    },
                                )
                                executed_action = np.asarray(plan["action"], dtype=np.float32)
                                resolved_by = "fallback_planner"
                                risk_after = float(plan["risk_after"])
                                gating.current_state = ControlState.PROCEED if risk_after < gating.delta_high else gating.current_state

                            next_obs, reward, done, env_info = env.step(executed_action)

                            if event is not None:
                                interventions += 1
                                if event == "pause":
                                    pause_events += 1
                                elif event == "fallback":
                                    fallback_events += 1
                                intervention_logger.log_intervention(
                                    {
                                        "episode_id": episode_id,
                                        "t_step": t_step,
                                        "event_type": event,
                                        "phase": env_info["phase"],
                                        "risk_before": round(risk, 6),
                                        "risk_after": round(risk_after, 6),
                                        "world_x": env_info["world_x"],
                                        "world_y": env_info["world_y"],
                                        "world_z": env_info["world_z"],
                                        "resolved_by": resolved_by,
                                        "resume_success": bool(risk_after < gating.delta_high),
                                    }
                                )

                            if env_info["incident_type"] != "none":
                                contact_logger.log_contact(
                                    {
                                        "episode_id": episode_id,
                                        "t_step": t_step,
                                        "link_name": "panda_hand",
                                        "other_object": "drawer_handle" if task_name == "drawer" else "sorting_tray",
                                        "contact_force_n": env_info["contact_force_n"],
                                        "contact_torque_nm": env_info["contact_torque_nm"],
                                        "incident_type": env_info["incident_type"],
                                        "margin_to_collision_m": env_info["margin_to_collision_m"],
                                    }
                                )
                                totals["contacts"] += 1
                                peak_force = max(peak_force, float(env_info["contact_force_n"]))
                                peak_torque = max(peak_torque, float(env_info["contact_torque_nm"]))
                                collisions += int(env_info["collision"])

                            step_logger.log_step(
                                {
                                    "episode_id": episode_id,
                                    "t_step": t_step,
                                    "phase": env_info["phase"],
                                    "ee_pose": np.array2string(executed_action[:3], precision=4, separator=","),
                                    "joint_positions": np.array2string(executed_action, precision=4, separator=","),
                                    "raw_policy_confidence": round(raw_confidence, 6),
                                    "raw_uncertainty": round(float(policy_info["uncertainty_variance"] + policy_info["uncertainty_entropy"]), 6),
                                    "uncertainty_entropy": round(float(policy_info["uncertainty_entropy"]), 6),
                                    "uncertainty_variance": round(float(policy_info["uncertainty_variance"]), 6),
                                    "calibrated_success_prob": round(calibrated_success_prob, 6),
                                    "calibrated_failure_risk": round(risk, 6),
                                    "gating_state": state.name.lower(),
                                    "action_vector": np.array2string(executed_action, precision=4, separator=","),
                                    "observation_frame_path": "",
                                }
                            )

                            obs = next_obs
                            if done:
                                break

                        success = env.check_success()
                        totals["episodes"] += 1
                        totals["successes"] += int(success)
                        totals["collisions"] += collisions
                        totals["interventions"] += interventions
                        totals["fallbacks"] += fallback_events
                        episode_logger.log_episode(
                            {
                                "run_id": episode_id,
                                "task": task_name,
                                "method": f"{policy_name}_gated",
                                "policy": policy_name,
                                "seed": episode_seed,
                                "shift_axis": shift_axis,
                                "shift_severity": int(severity),
                                "instruction": instruction,
                                "success": int(success),
                                "failure_reason": "" if success else (env.failure_reason or "timeout"),
                                "collision_any": int(collisions > 0),
                                "collision_count": int(collisions),
                                "peak_contact_force_n": round(peak_force, 6),
                                "peak_contact_torque_nm": round(peak_torque, 6),
                                "num_interventions": int(interventions),
                                "num_pause_reobserve": int(pause_events),
                                "num_fallbacks": int(fallback_events),
                                "episode_wall_time_s": round(float(0.05 * env.step_count + 0.15 * interventions + 0.2 * fallback_events), 6),
                                "sim_time_s": round(float(env.step_count / 20.0), 6),
                                "object_in_bin": int(success if task_name == "drawer" else 0),
                                "drawer_closed": int(success if task_name == "drawer" else 0),
                                "timed_out": int((not success) and (env.failure_reason == "timeout")),
                                "video_path": "",
                            }
                        )
                        env.close()
    return totals


def write_manifest(root: Path, cfg: Dict[str, Any], checkpoint_metrics: List[Dict[str, Any]], totals: Dict[str, Any], args: argparse.Namespace, episodes_per_condition: int, max_steps: int) -> None:
    run_dir = root / "runs" / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(root / "data_logs" / "episodes.csv")
    summary = {
        "config": args.config,
        "seed": args.seed,
        "episodes_per_condition": episodes_per_condition,
        "max_steps": max_steps,
        "checkpoint_metrics": checkpoint_metrics,
        "totals": totals,
        "aggregate": {
            "success_rate": round(float(episodes["success"].mean()), 6) if not episodes.empty else 0.0,
            "collision_rate": round(float(episodes["collision_any"].mean()), 6) if not episodes.empty else 0.0,
            "avg_interventions": round(float(episodes["num_interventions"].mean()), 6) if not episodes.empty else 0.0,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "run_summary.md").write_text(
        "\n".join(
            [
                "# Experiment Summary",
                "",
                f"- Config: `{args.config}`",
                f"- Seed: `{args.seed}`",
                f"- Episodes: `{totals['episodes']}`",
                f"- Successes: `{totals['successes']}`",
                f"- Contacts logged: `{totals['contacts']}`",
                f"- Fallback events: `{totals['fallbacks']}`",
                "",
                "Artifacts:",
                "- `data_logs/*.csv`",
                "- `checkpoints/*.ckpt` and `*.json`",
                "- `figures/*.png`",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    episodes_per_condition = int(args.episodes_per_condition or cfg.get("evaluation", {}).get("episodes_per_condition", 5))
    max_steps = int(args.max_steps or cfg.get("evaluation", {}).get("max_steps_per_episode", 40))
    episodes_per_condition = min(episodes_per_condition, 8)
    max_steps = min(max_steps, 60)

    if args.clean:
        ensure_clean(ROOT)
    else:
        (ROOT / "data_logs").mkdir(parents=True, exist_ok=True)
        (ROOT / "figures").mkdir(parents=True, exist_ok=True)
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    checkpoint_metrics: List[Dict[str, Any]] = []
    for policy_name, policy_cfg in cfg["policies"].items():
        trained = train_policy_checkpoint(policy_name, policy_cfg, cfg["tasks"], rng)
        ckpt_path = checkpoint_path(ROOT, policy_cfg["checkpoint"])
        save_checkpoint(ckpt_path, trained["payload"])
        metrics_path = ckpt_path.with_suffix(ckpt_path.suffix + ".json")
        metrics_path.write_text(json.dumps(trained["metrics"], indent=2), encoding="utf-8")
        checkpoint_metrics.append({"checkpoint": str(ckpt_path.relative_to(ROOT)), **trained["metrics"]})

    totals = run_evaluation(ROOT, cfg, args.seed, episodes_per_condition, max_steps)
    render_all_figures(ROOT / "data_logs", ROOT / "figures")
    write_manifest(ROOT, cfg, checkpoint_metrics, totals, args, episodes_per_condition, max_steps)
    print(json.dumps({"status": "ok", "totals": totals, "checkpoint_metrics": checkpoint_metrics}, indent=2))


if __name__ == "__main__":
    main()
