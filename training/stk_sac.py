"""
stk_sac.py — SAC training with CNN (image input) on GPU for SuperTuxKart.
The AI sees what a human sees — learns from pixels.

Run on Fedora: python3 training/stk_sac.py
"""
import sys, os, math, time, json, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pystk2
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from PIL import Image as PILImage
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
IMG_W, IMG_H = 160, 90      # observation image size (downscaled for CNN speed)
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9998


# ── Custom CNN Feature Extractor ──────────────────────────────────────────────
class STKCnn(BaseFeaturesExtractor):
    """CNN that processes the game image into features for SAC."""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # channels first
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, IMG_H, IMG_W)
            n_flat = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


# ── STK Environment with image observations ──────────────────────────────────
class STKImageEnv(gym.Env):
    """SuperTuxKart env — observation is the game image (channels first for CNN)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, track="lighthouse", num_karts=1, laps=3):
        super().__init__()
        self.track_name = track
        self.num_karts = num_karts
        self.laps = laps
        self._race = None
        self._world = None
        self._steps = 0
        self._max_steps = 3000
        self._prev_distance = 0
        self._last_image_full = None  # full res for stream

        # Action: [steer, accel_brake] continuous
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation: image (channels first for PyTorch CNN)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, IMG_H, IMG_W),
            dtype=np.uint8,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._race is not None:
            try:
                self._race.stop()
            except Exception:
                pass
            del self._race

        race_config = pystk2.RaceConfig()
        race_config.track = self.track_name
        race_config.num_kart = self.num_karts
        race_config.laps = self.laps
        race_config.players[0].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL

        self._race = pystk2.Race(race_config)
        self._race.start()
        self._race.step()

        self._world = pystk2.WorldState()
        self._world.update()

        self._steps = 0
        self._prev_distance = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._steps += 1
        steer = float(action[0])
        accel_brake = float(action[1])

        pystk_action = pystk2.Action()
        pystk_action.steer = np.clip(steer, -1, 1)
        if accel_brake >= 0:
            pystk_action.acceleration = accel_brake
            pystk_action.brake = False
        else:
            pystk_action.acceleration = 0
            pystk_action.brake = True
        pystk_action.drift = abs(steer) > 0.8
        pystk_action.nitro = accel_brake > 0.9

        self._race.step(pystk_action)
        self._world.update()

        obs = self._get_obs()

        # Reward
        kart = self._world.karts[0]
        distance = kart.distance_down_track
        velocity = kart.speed

        progress = distance - self._prev_distance
        if progress < -100:
            progress += 1000
        reward = progress * 0.1
        reward += velocity * 0.005

        if kart.is_on_road:
            reward += 0.05
        else:
            reward -= 0.5

        self._prev_distance = distance

        terminated = False
        truncated = False

        if kart.has_finished_race:
            reward += 100
            terminated = True

        if self._steps >= self._max_steps:
            truncated = True

        if velocity < 0.3 and self._steps > 50:
            reward -= 0.1

        info = {
            "distance": distance,
            "velocity": velocity,
            "laps": kart.finished_laps,
            "position": kart.position,
            "on_road": kart.is_on_road,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get downscaled image as observation (channels first)."""
        try:
            raw = np.array(self._race.render_data[0].image)
            self._last_image_full = raw  # save full res for stream
            # Resize to CNN input size
            img = PILImage.fromarray(raw).resize((IMG_W, IMG_H), PILImage.BILINEAR)
            # Channels first (H,W,C) → (C,H,W)
            return np.transpose(np.array(img, dtype=np.uint8), (2, 0, 1))
        except Exception:
            self._last_image_full = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            return np.zeros((3, IMG_H, IMG_W), dtype=np.uint8)

    def render(self):
        return self._last_image_full

    def close(self):
        if self._race:
            try:
                self._race.stop()
            except Exception:
                pass


# ── Stream Callback ───────────────────────────────────────────────────────────
def connect_proxy(retries=10, delay=2.0):
    for attempt in range(retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((PROXY_HOST, PROXY_PORT))
            print(f"[STREAM] Proxy connected on attempt {attempt + 1}", flush=True)
            return s
        except OSError:
            if attempt < retries - 1:
                time.sleep(delay)
    print("[STREAM] Could not connect to proxy", flush=True)
    return None


class StreamCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.proxy_sock = None
        self.start_time = time.time()
        self.best_distance = 0
        self.best_laps = 0
        self.frame_count = 0

    def _on_training_start(self):
        pass

    def _on_step(self) -> bool:
        self.frame_count += 1

        # Update stats
        infos = self.locals.get("infos", [])
        for info in infos:
            if info:
                d = info.get("distance", 0)
                l = info.get("laps", 0)
                if d > self.best_distance:
                    self.best_distance = d
                if l > self.best_laps:
                    self.best_laps = l

        # Render every 2 steps
        if self.frame_count % 2 != 0:
            return True

        try:
            env = self.training_env.envs[0]
            raw = env.unwrapped.render() if hasattr(env, 'unwrapped') else env.render()
            if raw is not None and raw.size > 0:
                img = PILImage.fromarray(raw)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if img.size != (WIDTH, HEIGHT):
                    img = img.resize((WIDTH, HEIGHT), PILImage.BILINEAR)
                image = np.array(img, dtype=np.uint8)

                expected = WIDTH * HEIGHT * 3
                if image.nbytes == expected and self.proxy_sock:
                    try:
                        self.proxy_sock.sendall(image.tobytes())
                    except OSError:
                        self.proxy_sock = connect_proxy(retries=2, delay=1)

                if self.frame_count % 30 == 0:
                    try:
                        PILImage.fromarray(image).save("/tmp/game_frame.jpg")
                    except Exception:
                        pass
        except Exception as e:
            if self.frame_count % 200 == 0:
                print(f"[STREAM] Render error: {e}", flush=True)

        # Write state
        if self.frame_count % 60 == 0:
            elapsed = time.time() - self.start_time
            try:
                with open("/tmp/racer_state.json", "w") as f:
                    json.dump({
                        "timesteps": self.num_timesteps,
                        "best_distance": round(self.best_distance, 1),
                        "best_laps": self.best_laps,
                        "elapsed": round(elapsed),
                        "algorithm": "SAC + CNN (GPU)",
                        "game": "SuperTuxKart",
                    }, f)
            except Exception:
                pass

        return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STK] SAC + CNN Training on {device.upper()}", flush=True)
    if device == "cuda":
        print(f"[STK] GPU: {torch.cuda.get_device_name()}", flush=True)

    proxy_sock = connect_proxy()

    # Init pystk2
    gfx = pystk2.GraphicsConfig.sd()
    gfx.screen_width = 640
    gfx.screen_height = 360
    pystk2.init(gfx)

    tracks = pystk2.list_tracks()
    track = "lighthouse" if "lighthouse" in tracks else tracks[0]
    print(f"[STK] Track: {track} | Obs: {IMG_W}x{IMG_H} image → CNN → GPU", flush=True)

    env = STKImageEnv(track=track, num_karts=1, laps=3)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=3000)

    model = SAC(
        "CnnPolicy", env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        learning_starts=1000,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs={
            "features_extractor_class": STKCnn,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [256, 256],
            "normalize_images": False,
        },
    )

    callback = StreamCallback()
    callback.proxy_sock = proxy_sock

    print("[STK] Training live with SAC + CNN on GPU...", flush=True)
    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=callback,
        )
    except KeyboardInterrupt:
        print("\n[STK] Stopped", flush=True)

    model.save("/tmp/stk_sac_model")
    env.close()
    print("[STK] Done", flush=True)


if __name__ == "__main__":
    main()
