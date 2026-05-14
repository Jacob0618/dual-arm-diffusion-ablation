"""
Student 4: Automated Ablation Study for Diffusion Policy
=========================================================
Usage on AutoDL:
    python run_ablation.py

Runs 12 ablation experiments automatically:
  - 5x demo count ablation (50/100/200/300/500 demos)
  - 4x prediction horizon ablation (4/8/16/32)
  - 3x observation horizon ablation (1/2/4)

Results saved to ./ablation_results/
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import json
import time
import csv
from torch.utils.data import DataLoader, Dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from model import ConditionalUnet1D


# ============================================================
# Dataset
# ============================================================
def normalize_data(data, stats):
    return 2 * (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-6) - 1


class DualArmDiffusionDataset(Dataset):
    def __init__(self, h5_path, obs_horizon=2, pred_horizon=16, max_demos=None):
        self.h5_path = h5_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.max_demos = max_demos
        self.slices = []
        self.stats = {}
        self.file = None
        self._load_and_index()

    def _load_and_index(self):
        all_obs = []
        all_actions = []

        with h5py.File(self.h5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())
            if self.max_demos is not None:
                demo_keys = demo_keys[:self.max_demos]

            for demo_name in demo_keys:
                demo = f["data"][demo_name]
                obs = np.concatenate([
                    demo["obs/cube_pos"][:],
                    demo["obs/robot0_eef_pos"][:],
                    demo["obs/robot1_eef_pos"][:],
                    demo["obs/target_pos"][:],
                ], axis=-1)
                all_obs.append(obs)
                all_actions.append(demo["actions"][:])

                num_samples = int(demo.attrs.get("num_samples", obs.shape[0]))
                for i in range(num_samples - self.pred_horizon - self.obs_horizon + 1):
                    self.slices.append((demo_name, i))

        all_obs = np.concatenate(all_obs, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        self.stats["obs"] = {"min": all_obs.min(axis=0), "max": all_obs.max(axis=0)}
        self.stats["action"] = {"min": all_actions.min(axis=0), "max": all_actions.max(axis=0)}

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        demo_name, start_idx = self.slices[idx]
        demo = self.file["data"][demo_name]

        obs_raw = np.concatenate([
            demo["obs/cube_pos"][start_idx:start_idx + self.obs_horizon],
            demo["obs/robot0_eef_pos"][start_idx:start_idx + self.obs_horizon],
            demo["obs/robot1_eef_pos"][start_idx:start_idx + self.obs_horizon],
            demo["obs/target_pos"][start_idx:start_idx + self.obs_horizon],
        ], axis=-1)
        act_raw = demo["actions"][
            start_idx + self.obs_horizon:start_idx + self.obs_horizon + self.pred_horizon
        ]

        return {
            "obs": torch.from_numpy(normalize_data(obs_raw, self.stats["obs"])).float(),
            "action": torch.from_numpy(normalize_data(act_raw, self.stats["action"])).float(),
        }


# ============================================================
# Single Training Run
# ============================================================
def train_single(
    h5_path,
    exp_name,
    num_epochs=100,
    batch_size=256,
    obs_horizon=2,
    pred_horizon=16,
    max_demos=None,
    save_dir="ablation_results",
):
    """Run one training experiment and return loss history."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 12
    action_dim = 14

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  obs_horizon={obs_horizon}, pred_horizon={pred_horizon}, "
          f"max_demos={max_demos}, epochs={num_epochs}, batch={batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Dataset
    dataset = DualArmDiffusionDataset(
        h5_path, obs_horizon=obs_horizon, pred_horizon=pred_horizon, max_demos=max_demos
    )
    print(f"  Dataset samples: {len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = ConditionalUnet1D(
        action_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count/1e6:.1f}M")

    ema = EMAModel(model.parameters(), decay=0.999)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs,
    )

    # Training loop
    loss_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            obs_cond = batch["obs"].flatten(start_dim=1).to(device)
            action = batch["action"].to(device)

            noise = torch.randn(action.shape, device=device)
            timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

            noise_pred = model(noisy_action, timesteps, global_cond=obs_cond)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(model.parameters())

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (num_epochs - epoch - 1)
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.6f} | "
                  f"Time: {elapsed:.0f}s | ETA: {eta:.0f}s")

    total_time = time.time() - start_time
    final_loss = loss_history[-1]
    print(f"  Done! Final loss: {final_loss:.6f} | Total time: {total_time:.0f}s")

    # Save results
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save loss CSV
    csv_path = os.path.join(exp_dir, "loss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, l in enumerate(loss_history):
            writer.writerow([i + 1, l])

    # Save config
    config = {
        "exp_name": exp_name,
        "obs_horizon": obs_horizon,
        "pred_horizon": pred_horizon,
        "max_demos": max_demos,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "dataset_size": len(dataset),
        "param_count": param_count,
        "final_loss": final_loss,
        "total_time_seconds": total_time,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save checkpoint
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "stats": dataset.stats,
    }, os.path.join(exp_dir, "checkpoint.ckpt"))

    return loss_history, config


# ============================================================
# Plotting
# ============================================================
def plot_ablation_group(group_name, experiments, save_dir="ablation_results"):
    """Plot loss curves for a group of ablation experiments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3,
    })

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']

    # --- Loss curve comparison ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, losses, config) in enumerate(experiments):
        label = name.replace("_", " ")
        ax.semilogy(range(1, len(losses)+1), losses,
                     color=colors[i % len(colors)], linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(f"Ablation: {group_name}")
    ax.legend()
    plt.savefig(os.path.join(save_dir, f"ablation_{group_name.lower().replace(' ', '_')}.png"))
    plt.close()

    # --- Final loss bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [e[0].replace("_", " ") for e in experiments]
    final_losses = [e[1][-1] for e in experiments]
    bars = ax.bar(names, final_losses, color=colors[:len(experiments)], width=0.5)
    for bar, val in zip(bars, final_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.5f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Final MSE Loss")
    ax.set_title(f"Final Loss Comparison: {group_name}")
    plt.xticks(rotation=15, ha='right')
    plt.savefig(os.path.join(save_dir, f"ablation_{group_name.lower().replace(' ', '_')}_bar.png"))
    plt.close()

    print(f"  [Plot] {group_name} saved.")


def plot_summary(all_configs, save_dir="ablation_results"):
    """Generate a master summary table image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, max(4, len(all_configs) * 0.5 + 1.5)))
    ax.axis("off")

    columns = ["Experiment", "Demos", "Obs Horizon", "Pred Horizon",
               "Dataset Size", "Final Loss", "Time (min)"]
    data = []
    for c in all_configs:
        data.append([
            c["exp_name"],
            str(c["max_demos"] or 500),
            str(c["obs_horizon"]),
            str(c["pred_horizon"]),
            str(c["dataset_size"]),
            f'{c["final_loss"]:.6f}',
            f'{c["total_time_seconds"]/60:.1f}',
        ])

    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(columns)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Ablation Study: Complete Results Summary", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(os.path.join(save_dir, "ablation_summary_table.png"))
    plt.close()
    print("[Plot] Summary table saved.")


# ============================================================
# Main: Run All Ablations
# ============================================================
def main():
    H5_PATH = "data/demo_robomimic_compatible.hdf5"
    SAVE_DIR = "ablation_results"
    NUM_EPOCHS = 100       # 100 epochs enough to see trends
    BATCH_SIZE = 256       # 96GB VRAM, go big
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(H5_PATH):
        print(f"ERROR: Dataset not found at {H5_PATH}")
        print("Please place demo_robomimic_compatible.hdf5 in ./data/")
        return

    all_configs = []
    total_start = time.time()

    # -------------------------------------------------------
    # Ablation 1: Number of Demonstrations (50/100/200/300/500)
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ABLATION 1: Number of Demonstrations")
    print("=" * 60)

    demo_experiments = []
    for n_demos in [50, 100, 200, 300, 500]:
        name = f"demos_{n_demos}"
        losses, config = train_single(
            H5_PATH, name,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            obs_horizon=2, pred_horizon=16,
            max_demos=n_demos, save_dir=SAVE_DIR,
        )
        demo_experiments.append((name, losses, config))
        all_configs.append(config)

    plot_ablation_group("Demo Count", demo_experiments, SAVE_DIR)

    # -------------------------------------------------------
    # Ablation 2: Prediction Horizon (4/8/16/32)
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ABLATION 2: Prediction Horizon")
    print("=" * 60)

    pred_experiments = []
    for pred_h in [4, 8, 16, 32]:
        name = f"pred_horizon_{pred_h}"
        losses, config = train_single(
            H5_PATH, name,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            obs_horizon=2, pred_horizon=pred_h,
            max_demos=None, save_dir=SAVE_DIR,
        )
        pred_experiments.append((name, losses, config))
        all_configs.append(config)

    plot_ablation_group("Prediction Horizon", pred_experiments, SAVE_DIR)

    # -------------------------------------------------------
    # Ablation 3: Observation Horizon (1/2/4)
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ABLATION 3: Observation Horizon")
    print("=" * 60)

    obs_experiments = []
    for obs_h in [1, 2, 4]:
        name = f"obs_horizon_{obs_h}"
        losses, config = train_single(
            H5_PATH, name,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            obs_horizon=obs_h, pred_horizon=16,
            max_demos=None, save_dir=SAVE_DIR,
        )
        obs_experiments.append((name, losses, config))
        all_configs.append(config)

    plot_ablation_group("Observation Horizon", obs_experiments, SAVE_DIR)

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL ABLATIONS COMPLETE!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results saved to: {SAVE_DIR}/")
    print(f"{'='*60}")

    # Save master config
    with open(os.path.join(SAVE_DIR, "all_results.json"), "w") as f:
        json.dump(all_configs, f, indent=2)

    plot_summary(all_configs, SAVE_DIR)

    # Print final summary
    print("\n  Final Loss Summary:")
    print(f"  {'Experiment':<25} {'Final Loss':<12} {'Time (min)':<10}")
    print(f"  {'-'*47}")
    for c in all_configs:
        print(f"  {c['exp_name']:<25} {c['final_loss']:<12.6f} {c['total_time_seconds']/60:<10.1f}")


if __name__ == "__main__":
    main()
