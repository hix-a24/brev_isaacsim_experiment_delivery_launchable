"""Robust figure rendering for the synthetic experiment pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _placeholder(fig_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def render_figure6(data_dir: Path, figures_dir: Path) -> None:
    _ensure_dir(figures_dir)
    episodes_path = data_dir / "episodes.csv"
    fig_path = figures_dir / "fig6_success_collision.png"
    if not episodes_path.exists():
        _placeholder(fig_path, "No episode data available")
        return

    df = pd.read_csv(episodes_path)
    if df.empty:
        _placeholder(fig_path, "Episode log is empty")
        return

    grouped = (
        df.groupby(["task", "shift_axis", "policy"]).agg(
            success_rate=("success", "mean"),
            collision_rate=("collision_any", "mean"),
        )
    ).reset_index()
    grouped["row"] = grouped["task"] + " / " + grouped["shift_axis"]
    pivot_success = grouped.pivot(index="row", columns="policy", values="success_rate").fillna(0.0)
    pivot_collision = grouped.pivot(index="row", columns="policy", values="collision_rate").fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.45 * len(pivot_success.index))))
    for ax, pivot, title in [
        (axes[0], pivot_success, "Success Rate"),
        (axes[1], pivot_collision, "Collision Rate"),
    ]:
        im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)), labels=list(pivot.columns))
        ax.set_yticks(range(len(pivot.index)), labels=list(pivot.index))
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def render_figure7(data_dir: Path, figures_dir: Path) -> None:
    _ensure_dir(figures_dir)
    episodes_path = data_dir / "episodes.csv"
    fig_path = figures_dir / "fig7_robustness_curves.png"
    if not episodes_path.exists():
        _placeholder(fig_path, "No episode data available")
        return

    df = pd.read_csv(episodes_path)
    if df.empty:
        _placeholder(fig_path, "Episode log is empty")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = df.groupby(["task", "shift_axis", "policy", "shift_severity"])["success"].agg(["mean", "count"]).reset_index()
    for (task, axis, policy), sub_df in grouped.groupby(["task", "shift_axis", "policy"]):
        p = sub_df["mean"].to_numpy(dtype=float)
        n = sub_df["count"].to_numpy(dtype=float)
        severities = sub_df["shift_severity"].to_numpy(dtype=float)
        z = 1.96
        denom = 1 + z**2 / np.maximum(n, 1.0)
        centre = (p + z**2 / (2 * np.maximum(n, 1.0))) / denom
        error = z * np.sqrt(np.maximum(p * (1 - p) / np.maximum(n, 1.0) + z**2 / (4 * np.maximum(n, 1.0) ** 2), 0.0)) / denom
        ax.plot(severities, p, marker="o", label=f"{task}/{axis}/{policy}")
        ax.fill_between(severities, np.clip(centre - error, 0, 1), np.clip(centre + error, 0, 1), alpha=0.15)
    ax.set_xlabel("Shift Severity")
    ax.set_ylabel("Success Rate")
    ax.set_title("Robustness Curves")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def render_figure8(data_dir: Path, figures_dir: Path) -> None:
    _ensure_dir(figures_dir)
    steps_path = data_dir / "steps.csv"
    episodes_path = data_dir / "episodes.csv"
    fig_path = figures_dir / "fig8_calibration.png"
    if not steps_path.exists() or not episodes_path.exists():
        _placeholder(fig_path, "Missing step or episode data")
        return

    steps = pd.read_csv(steps_path)
    episodes = pd.read_csv(episodes_path)[["run_id", "success"]].rename(columns={"run_id": "episode_id"})
    if steps.empty or episodes.empty:
        _placeholder(fig_path, "Calibration data is empty")
        return

    df = steps.merge(episodes, on="episode_id", how="left")
    if df["success"].isna().all():
        _placeholder(fig_path, "Missing success labels for calibration")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(0.0, 1.0, 11)
    for ax, label, col in [
        (axes[0], "Raw", "raw_policy_confidence"),
        (axes[1], "Calibrated", "calibrated_success_prob"),
    ]:
        prob = df[col].clip(0.0, 1.0).to_numpy(dtype=float)
        true = df["success"].fillna(0).to_numpy(dtype=float)
        bin_ids = np.clip(np.digitize(prob, bins) - 1, 0, 9)
        bin_prob = []
        bin_true = []
        for i in range(10):
            mask = bin_ids == i
            if mask.any():
                bin_prob.append(prob[mask].mean())
                bin_true.append(true[mask].mean())
            else:
                bin_prob.append(np.nan)
                bin_true.append(np.nan)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.scatter(bin_prob, bin_true)
        ax.set_title(f"Reliability Diagram ({label})")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def render_figure9(data_dir: Path, figures_dir: Path) -> None:
    _ensure_dir(figures_dir)
    interventions_path = data_dir / "interventions.csv"
    fig_path = figures_dir / "fig9_interventions.png"
    if not interventions_path.exists():
        _placeholder(fig_path, "No intervention data available")
        return

    df = pd.read_csv(interventions_path)
    if df.empty:
        _placeholder(fig_path, "Intervention log is empty")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    counts = df.groupby("episode_id").size()
    axes[0].hist(counts, bins=range(1, int(counts.max()) + 2), align="left", rwidth=0.85)
    axes[0].set_xlabel("Interventions per Episode")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Episode Intervention Counts")

    resolution_counts = df.groupby(["event_type", "resolved_by"]).size().unstack(fill_value=0)
    bottoms = np.zeros(len(resolution_counts.index))
    for col in resolution_counts.columns:
        values = resolution_counts[col].to_numpy(dtype=float)
        axes[1].bar(resolution_counts.index, values, bottom=bottoms, label=col)
        bottoms += values
    axes[1].set_title("Intervention Resolutions")
    axes[1].set_xlabel("Event Type")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize="small")

    hist, xedges, yedges = np.histogram2d(df["world_x"], df["world_y"], bins=20)
    mesh = axes[2].pcolormesh(xedges, yedges, hist.T)
    axes[2].set_xlabel("World X")
    axes[2].set_ylabel("World Y")
    axes[2].set_title("Intervention Density")
    fig.colorbar(mesh, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def render_figure10(data_dir: Path, figures_dir: Path) -> None:
    _ensure_dir(figures_dir)
    contacts_path = data_dir / "contacts.csv"
    fig_path = figures_dir / "fig10_safety_characterization.png"
    if not contacts_path.exists():
        _placeholder(fig_path, "No contact data available")
        return

    df = pd.read_csv(contacts_path)
    if df.empty:
        _placeholder(fig_path, "Contact log is empty")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df["margin_to_collision_m"], bins=25)
    axes[0].set_title("Margin to Collision")
    axes[0].set_xlabel("Margin (m)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["contact_force_n"], bins=25)
    axes[1].set_title("Contact Force Distribution")
    axes[1].set_xlabel("Force (N)")
    axes[1].set_ylabel("Count")

    incident_counts = df.groupby("incident_type").size()
    axes[2].bar(incident_counts.index, incident_counts.values)
    axes[2].set_title("Incident Type Counts")
    axes[2].set_xlabel("Incident Type")
    axes[2].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def render_all_figures(data_dir: str | Path = "data_logs", figures_dir: str | Path = "figures") -> None:
    data_dir = Path(data_dir)
    figures_dir = Path(figures_dir)
    render_figure6(data_dir, figures_dir)
    render_figure7(data_dir, figures_dir)
    render_figure8(data_dir, figures_dir)
    render_figure9(data_dir, figures_dir)
    render_figure10(data_dir, figures_dir)
