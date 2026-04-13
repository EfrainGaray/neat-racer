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
IMG_W, IMG_H = 84, 84       # standard RL observation size (square for CNN)
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9998
CHECKPOINT_PATH = "/home/clawadmin/neat-racer/checkpoints/stk_sac"


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

    def __init__(self, track="lighthouse", num_karts=1, laps=99):
        super().__init__()
        self.track_name = track
        self.num_karts = num_karts
        self.laps = laps
        self._race = None
        self._world = None
        self._steps = 0
        self._max_steps = 2000     # episodes end — forces learning from resets
        self._prev_distance = 0
        self._stuck_count = 0      # consecutive steps with near-zero speed
        self._last_image_full = None

        # Action: [steer (-1 to 1), accel_brake (-1 to 1)]
        # negative = brake/reverse (to unstick from walls), positive = accelerate
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
            # Race already running — just restart it (avoids segfault from stop+start)
            self._race.restart()
            self._race.step()
            self._world.update()
            self._steps = 0
            self._prev_distance = 0
            self._stuck_count = 0
            return self._get_obs(), {}

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
            pystk_action.acceleration = max(0.15, accel_brake)  # minimum 15% throttle
            pystk_action.brake = False
        else:
            pystk_action.acceleration = 0
            pystk_action.brake = True  # can reverse to unstick from walls
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
            progress += 1000  # crossed finish line

        # Reward — kept in [-1, 1] range for SAC stability
        if progress > 0:
            reward = min(progress * 0.05, 1.0)  # forward progress capped at 1.0
            reward += min(velocity * 0.02, 0.3)  # speed bonus
        else:
            reward = max(progress * 0.1, -1.0)  # backward penalty capped at -1.0

        if kart.is_on_road:
            reward += 0.02
        else:
            reward -= 0.3

        # Stuck detection — based on actual progress, not velocity
        # (kart can spin wheels against wall with velocity > 0 but zero progress)
        if abs(progress) < 0.1 or not kart.is_on_road:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        if self._stuck_count > 30:  # ~1 second no progress
            reward -= 0.5
        if self._stuck_count > 60:  # ~2 seconds
            reward -= 1.0

        # Clamp total reward
        reward = np.clip(reward, -1.5, 1.5)

        self._prev_distance = distance

        terminated = self._stuck_count > 90   # 3 seconds no progress = episode over
        truncated = self._steps >= self._max_steps

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

    def _draw_hud(self, img, infos):
        """Draw training stats overlay on PIL image."""
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)

        elapsed = time.time() - self.start_time
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        hours = int(elapsed // 3600)

        # Current speed/distance from infos
        cur_speed = 0
        cur_dist = 0
        cur_laps = 0
        on_road = True
        for info in (infos or []):
            if info:
                cur_speed = info.get("velocity", 0)
                cur_dist = info.get("distance", 0)
                cur_laps = info.get("laps", 0)
                on_road = info.get("on_road", True)

        # Try loading a monospace font
        try:
            font = ImageFont.truetype("/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf", 16)
            font_sm = ImageFont.truetype("/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf", 13)
            font_title = ImageFont.truetype("/usr/share/fonts/google-noto/NotoSansMono-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            font_sm = font
            font_title = font

        # Panel background
        panel_w, panel_h = 260, 250
        panel_x, panel_y = 16, 16
        overlay = PILImage.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
            radius=8, fill=(8, 6, 18, 200)
        )
        # Border
        overlay_draw.rounded_rectangle(
            [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
            radius=8, outline=(0, 255, 255, 50), width=1
        )
        img = PILImage.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        cx = panel_x + 14
        cy = panel_y + 12

        # Title
        draw.text((cx, cy), "SAC + CNN  |  GPU", fill=(140, 0, 255), font=font_title)
        cy += 22

        # Separator
        draw.line([(cx, cy), (panel_x + panel_w - 14, cy)], fill=(255, 255, 255, 30), width=1)
        cy += 8

        lines = [
            ("STEPS", f"{self.num_timesteps:,}", (0, 255, 255)),
            ("BEST DIST", f"{self.best_distance:.0f}", (255, 200, 50)),
            ("BEST LAPS", f"{self.best_laps}", (255, 0, 128) if self.best_laps > 0 else (100, 110, 120)),
            ("SPEED", f"{cur_speed:.1f}", (30, 215, 96) if cur_speed > 2 else (239, 68, 68)),
            ("DISTANCE", f"{cur_dist:.0f}", (250, 249, 246)),
            ("ON ROAD", "YES" if on_road else "OFF", (30, 215, 96) if on_road else (239, 68, 68)),
            ("TIME", f"{hours:02d}:{mins%60:02d}:{secs:02d}", (100, 110, 120)),
        ]

        for label, value, color in lines:
            draw.text((cx, cy), label, fill=(138, 143, 152), font=font_sm)
            val_w = draw.textlength(value, font=font)
            draw.text((panel_x + panel_w - 14 - val_w, cy - 1), value, fill=color, font=font)
            cy += 26

        # Bottom tag
        tag = "NEAT RACER — AI Learns from Pixels"
        draw.text((16, HEIGHT - 24), tag, fill=(60, 65, 75), font=font_sm)

        # FPS indicator
        fps_text = f"{self.frame_count / max(elapsed, 1):.0f} fps"
        fw = draw.textlength(fps_text, font=font_sm)
        draw.text((WIDTH - fw - 16, HEIGHT - 24), fps_text, fill=(60, 65, 75), font=font_sm)

        return img

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

        # Real-time sync — render every step, throttle to ~30fps
        now = time.time()
        if hasattr(self, '_last_render_time'):
            elapsed = now - self._last_render_time
            target = 1.0 / 30  # 30fps = 1 game step per frame
            if elapsed < target:
                time.sleep(target - elapsed)
        self._last_render_time = time.time()

        try:
            env = self.training_env.envs[0]
            raw = env.unwrapped.render() if hasattr(env, 'unwrapped') else env.render()
            if raw is not None and raw.size > 0:
                img = PILImage.fromarray(raw)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if img.size != (WIDTH, HEIGHT):
                    img = img.resize((WIDTH, HEIGHT), PILImage.BILINEAR)

                # Draw HUD overlay
                img = self._draw_hud(img, infos)

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

        # Autosave model every 10k steps
        if self.num_timesteps > 0 and self.num_timesteps % 5_000 == 0:
            try:
                self.model.save(CHECKPOINT_PATH)
                self.model.save_replay_buffer(CHECKPOINT_PATH + "_buffer")
                print(f"[SAVE] Checkpoint + buffer at {self.num_timesteps:,} steps", flush=True)
            except Exception as e:
                print(f"[SAVE] Error: {e}", flush=True)

        return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STK] SAC + CNN Training on {device.upper()}", flush=True)
    if device == "cuda":
        print(f"[STK] GPU: {torch.cuda.get_device_name()}", flush=True)

    proxy_sock = connect_proxy()

    # Init pystk2
    gfx = pystk2.GraphicsConfig.hd()
    gfx.screen_width = 1280
    gfx.screen_height = 720
    pystk2.init(gfx)

    tracks = pystk2.list_tracks()
    track = "lighthouse" if "lighthouse" in tracks else tracks[0]
    print(f"[STK] Track: {track} | Obs: {IMG_W}x{IMG_H} image → CNN → GPU", flush=True)

    env = STKImageEnv(track=track, num_karts=1, laps=99)

    # Try loading checkpoint, otherwise create new model
    checkpoint_file = CHECKPOINT_PATH + ".zip"
    if os.path.exists(checkpoint_file):
        print(f"[STK] Restoring from checkpoint: {checkpoint_file}", flush=True)
        model = SAC.load(
            CHECKPOINT_PATH, env=env, device=device,
            custom_objects={
                "policy_kwargs": {
                    "features_extractor_class": STKCnn,
                    "features_extractor_kwargs": {"features_dim": 256},
                    "net_arch": [256, 256],
                    "normalize_images": False,
                },
            },
        )
        # Restore replay buffer if exists
        buffer_file = CHECKPOINT_PATH + "_buffer.pkl"
        if os.path.exists(buffer_file):
            model.load_replay_buffer(buffer_file)
            print(f"[STK] Checkpoint + replay buffer loaded ({model.replay_buffer.size()} experiences)", flush=True)
        else:
            print(f"[STK] Checkpoint loaded (no buffer — will re-explore)", flush=True)
    else:
        print(f"[STK] No checkpoint found — starting fresh", flush=True)
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        model = SAC(
            "CnnPolicy", env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            buffer_size=100_000,       # ~4.2GB RAM with 84x84 images (26GB available)
            batch_size=256,
            learning_starts=5000,      # explore more before training
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            gradient_steps=2,          # 2 gradient steps per env step
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
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[STK] Stopped", flush=True)

    model.save(CHECKPOINT_PATH)
    try:
        model.save_replay_buffer(CHECKPOINT_PATH + "_buffer")
    except Exception:
        pass
    print(f"[STK] Final checkpoint + buffer saved", flush=True)
    env.close()
    print("[STK] Done", flush=True)


if __name__ == "__main__":
    main()
