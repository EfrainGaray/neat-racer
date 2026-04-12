"""
train.py — PPO training on GPU for the racing environment.
Uses Stable Baselines3 with PyTorch CUDA backend.

Run on Fedora: python3 training/train.py
"""
import sys
import os
import torch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from game.racing_env import RacingEnv


class LogCallback(BaseCallback):
    """Log training progress to stdout and /tmp/racer_state.json."""
    def __init__(self):
        super().__init__()
        self._episode_count = 0

    def _on_step(self) -> bool:
        # Log every 2048 steps
        if self.num_timesteps % 2048 == 0:
            import json
            try:
                infos = self.locals.get("infos", [])
                best_dist = max((i.get("distance", 0) for i in infos), default=0)
                best_laps = max((i.get("laps", 0) for i in infos), default=0)
                state = {
                    "timesteps": self.num_timesteps,
                    "episodes": self._episode_count,
                    "best_distance": round(best_dist, 1),
                    "best_laps": best_laps,
                    "device": str(next(self.model.policy.parameters()).device),
                }
                with open("/tmp/racer_state.json", "w") as f:
                    json.dump(state, f)
                print(f"[TRAIN] Steps: {self.num_timesteps:,} | Dist: {best_dist:.0f} | Laps: {best_laps}", flush=True)
            except Exception:
                pass
        return True

    def _on_rollout_end(self):
        self._episode_count += 1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Device: {device}", flush=True)
    if device == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name()}", flush=True)
        print(f"[TRAIN] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # Create vectorized environment (parallel cars)
    n_envs = 16 if device == "cuda" else 4
    print(f"[TRAIN] Creating {n_envs} parallel environments...", flush=True)

    env = make_vec_env(RacingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # PPO with GPU
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "net_arch": [256, 256],  # 2 hidden layers, 256 neurons each
        },
        tensorboard_log="./logs/",
    )

    print(f"[TRAIN] Policy network: {model.policy}", flush=True)
    print(f"[TRAIN] Starting training...", flush=True)

    # Callbacks
    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="racer",
    )

    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[LogCallback(), checkpoint],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted — saving model...", flush=True)

    model.save("racer_final")
    print("[TRAIN] Model saved to racer_final.zip", flush=True)
    env.close()


if __name__ == "__main__":
    main()
