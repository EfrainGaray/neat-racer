"""
stk_sac.py — SAC+CNN v3 for SuperTuxKart
Optimized: low-res render, frame skip, 2x gradient steps, decoupled stream.
"""
import sys, os, time, json, socket, collections
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

# ── Config ─────────────────────────────────────────────────────────────
RENDER_W, RENDER_H = 400, 300      # pystk2 internal render (fast)
STREAM_W, STREAM_H = 1280, 720     # upscaled for stream
OBS_W, OBS_H = 84, 84              # CNN input
FRAME_STACK = 4                     # temporal context (grayscale)
FRAME_SKIP = 2                      # repeat action N frames
RENDER_FPS = 30                     # stream fps (does NOT throttle training)
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9998
CHECKPOINT_PATH = "/home/clawadmin/neat-racer/checkpoints/stk_sac"
SAVE_INTERVAL = 10_000


# ── CNN Feature Extractor ──────────────────────────────────────────────
class STKCnn(BaseFeaturesExtractor):
    """4-frame grayscale stacked → CNN → 256 features."""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, n_ch, OBS_H, OBS_W)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x.float() / 255.0))


# ── Environment ────────────────────────────────────────────────────────
class STKImageEnv(gym.Env):
    """SuperTuxKart with frame stacking + frame skip."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, track="lighthouse"):
        super().__init__()
        self.track_name = track
        self._race = None
        self._world = None
        self._steps = 0
        self._max_steps = 4000
        self._prev_distance = 0
        self._stuck_count = 0
        self._last_image_full = None
        self._frames = collections.deque(maxlen=FRAME_STACK)
        self._total_progress = 0
        self._best_distance = 0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(FRAME_STACK, OBS_H, OBS_W),
            dtype=np.uint8,
        )

    def _grab_frame(self):
        try:
            raw = np.array(self._race.render_data[0].image)
            self._last_image_full = raw
            img = PILImage.fromarray(raw).convert('L').resize(
                (OBS_W, OBS_H), PILImage.BILINEAR
            )
            return np.array(img, dtype=np.uint8)
        except Exception:
            self._last_image_full = np.zeros(
                (RENDER_H, RENDER_W, 3), dtype=np.uint8)
            return np.zeros((OBS_H, OBS_W), dtype=np.uint8)

    def _stacked_obs(self):
        while len(self._frames) < FRAME_STACK:
            self._frames.appendleft(np.zeros((OBS_H, OBS_W), dtype=np.uint8))
        return np.array(self._frames, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._race is not None:
            self._race.restart()
            self._race.step()
            self._world.update()
        else:
            rc = pystk2.RaceConfig()
            rc.track = self.track_name
            rc.num_kart = 1
            rc.laps = 99
            rc.players[0].controller = \
                pystk2.PlayerConfig.Controller.PLAYER_CONTROL
            self._race = pystk2.Race(rc)
            self._race.start()
            self._race.step()
            self._world = pystk2.WorldState()
            self._world.update()

        self._steps = 0
        self._prev_distance = 0
        self._stuck_count = 0
        self._total_progress = 0
        self._best_distance = 0
        self._frames.clear()
        f = self._grab_frame()
        for _ in range(FRAME_STACK):
            self._frames.append(f)
        return self._stacked_obs(), {}

    def step(self, action):
        self._steps += 1
        steer = float(np.clip(action[0], -1, 1))
        ab = float(action[1])

        a = pystk2.Action()
        a.steer = steer
        if ab >= 0:
            a.acceleration = max(0.15, ab)
            a.brake = False
        else:
            a.acceleration = 0
            a.brake = True
        a.drift = abs(steer) > 0.8
        a.nitro = ab > 0.9

        # ── Frame skip: repeat action N times ──
        total_reward = 0.0
        terminated = False
        for _ in range(FRAME_SKIP):
            self._race.step(a)
            self._world.update()

            kart = self._world.karts[0]
            dist = kart.distance_down_track
            vel = kart.speed

            progress = dist - self._prev_distance
            if progress < -100:
                progress += 1000

            if dist > self._best_distance:
                self._best_distance = dist

            # ── Reward v2 ──
            reward = 0.0
            if progress > 0.5:
                self._total_progress += progress
                reward = min(progress * 0.1, 1.0)
                reward += min(vel * 0.01, 0.15)
            elif progress > 0:
                self._total_progress += progress
                reward = progress * 0.05
            elif progress < -0.5:
                reward = -0.15
            else:
                reward = -0.02

            if kart.is_on_road:
                reward += 0.03
            else:
                reward -= 0.08

            if abs(progress) < 0.05:
                self._stuck_count += 1
            else:
                self._stuck_count = max(0, self._stuck_count - 3)

            if self._stuck_count > 60:
                reward -= 0.1
            if self._stuck_count > 120:
                reward -= 0.2

            self._prev_distance = dist
            total_reward += reward

            if self._stuck_count > 200:
                terminated = True
                break

        # Grab frame after skip (last frame of the action)
        self._frames.append(self._grab_frame())
        obs = self._stacked_obs()

        truncated = self._steps >= self._max_steps

        info = {
            "distance": dist,
            "velocity": vel,
            "laps": kart.finished_laps,
            "on_road": kart.is_on_road,
            "total_progress": self._total_progress,
            "best_distance": self._best_distance,
        }
        return obs, total_reward, terminated, truncated, info

    def render(self):
        return self._last_image_full

    def close(self):
        if self._race:
            try:
                self._race.stop()
            except Exception:
                pass


# ── Stream callback ────────────────────────────────────────────────────
def connect_proxy(retries=10, delay=2.0):
    for i in range(retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((PROXY_HOST, PROXY_PORT))
            print(f"[STREAM] Connected attempt {i+1}", flush=True)
            return s
        except OSError:
            if i < retries - 1:
                time.sleep(delay)
    print("[STREAM] Could not connect", flush=True)
    return None


class StreamCallback(BaseCallback):
    """Renders to stream at RENDER_FPS without throttling training."""

    def __init__(self):
        super().__init__()
        self.proxy_sock = None
        self.start_time = time.time()
        self.best_distance = 0
        self.best_laps = 0
        self.total_frames = 0
        self._last_render = 0.0
        self._last_save_step = 0
        self._ep_rewards = collections.deque(maxlen=100)
        self._current_ep_reward = 0

    def _draw_hud(self, img, infos):
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        elapsed = time.time() - self.start_time
        h = int(elapsed // 3600)
        m = int(elapsed % 3600 // 60)
        s = int(elapsed % 60)

        spd = dist = laps = progress = 0
        on_road = True
        for info in (infos or []):
            if info:
                spd = info.get("velocity", 0)
                dist = info.get("distance", 0)
                laps = info.get("laps", 0)
                on_road = info.get("on_road", True)
                progress = info.get("total_progress", 0)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf", 16)
            font_sm = ImageFont.truetype(
                "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf", 13)
            font_title = ImageFont.truetype(
                "/usr/share/fonts/google-noto/NotoSansMono-Bold.ttf", 14)
        except Exception:
            font = font_sm = font_title = ImageFont.load_default()

        pw, ph = 280, 310
        px, py = 16, 16
        ov = PILImage.new('RGBA', img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rounded_rectangle(
            [px, py, px + pw, py + ph], radius=8, fill=(8, 6, 18, 200))
        od.rounded_rectangle(
            [px, py, px + pw, py + ph], radius=8,
            outline=(0, 255, 255, 50), width=1)
        img = PILImage.alpha_composite(
            img.convert('RGBA'), ov).convert('RGB')
        draw = ImageDraw.Draw(img)

        cx, cy = px + 14, py + 12
        draw.text((cx, cy), "SAC+CNN v3 | GPU",
                  fill=(140, 0, 255), font=font_title)
        cy += 22
        draw.line([(cx, cy), (px + pw - 14, cy)],
                  fill=(255, 255, 255, 30), width=1)
        cy += 8

        fps_train = self.total_frames / max(elapsed, 1)
        avg_rew = (sum(self._ep_rewards) / len(self._ep_rewards)
                   if self._ep_rewards else 0)

        lines = [
            ("STEPS", f"{self.num_timesteps:,}", (0, 255, 255)),
            ("TRAIN FPS", f"{fps_train:.0f}", (0, 255, 255)),
            ("AVG REWARD", f"{avg_rew:.1f}",
             (30, 215, 96) if avg_rew > 0 else (239, 68, 68)),
            ("BEST DIST", f"{self.best_distance:.0f}", (255, 200, 50)),
            ("BEST LAPS", f"{self.best_laps}",
             (255, 0, 128) if self.best_laps > 0 else (100, 110, 120)),
            ("SPEED", f"{spd:.1f}",
             (30, 215, 96) if spd > 2 else (239, 68, 68)),
            ("DISTANCE", f"{dist:.0f}", (250, 249, 246)),
            ("PROGRESS", f"{progress:.0f}", (30, 215, 96)),
            ("ON ROAD", "YES" if on_road else "OFF",
             (30, 215, 96) if on_road else (239, 68, 68)),
            ("TIME", f"{h:02d}:{m:02d}:{s:02d}", (100, 110, 120)),
        ]
        for label, val, color in lines:
            draw.text((cx, cy), label, fill=(138, 143, 152), font=font_sm)
            vw = draw.textlength(val, font=font)
            draw.text((px + pw - 14 - vw, cy - 1), val, fill=color,
                      font=font)
            cy += 26

        draw.text((16, STREAM_H - 24),
                  "NEAT RACER v3 \u2014 AI Learns from Pixels",
                  fill=(60, 65, 75), font=font_sm)
        return img

    def _on_step(self) -> bool:
        self.total_frames += 1

        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for r in rewards:
            self._current_ep_reward += float(r)
        for d in dones:
            if d:
                self._ep_rewards.append(self._current_ep_reward)
                self._current_ep_reward = 0

        infos = self.locals.get("infos", [])
        for info in infos:
            if info:
                d = info.get("distance", 0)
                l = info.get("laps", 0)
                if d > self.best_distance:
                    self.best_distance = d
                if l > self.best_laps:
                    self.best_laps = l

        # ── Stream at RENDER_FPS — no sleep ──
        now = time.time()
        if now - self._last_render >= 1.0 / RENDER_FPS:
            self._last_render = now
            try:
                env = self.training_env.envs[0]
                raw = (env.unwrapped.render()
                       if hasattr(env, 'unwrapped') else env.render())
                if raw is not None and raw.size > 0:
                    img = PILImage.fromarray(raw)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Upscale from render res to stream res
                    if img.size != (STREAM_W, STREAM_H):
                        img = img.resize(
                            (STREAM_W, STREAM_H), PILImage.BILINEAR)
                    img = self._draw_hud(img, infos)
                    data = np.array(img, dtype=np.uint8)
                    expected = STREAM_W * STREAM_H * 3
                    if data.nbytes == expected and self.proxy_sock:
                        try:
                            self.proxy_sock.sendall(data.tobytes())
                        except OSError:
                            self.proxy_sock = connect_proxy(
                                retries=2, delay=1)
                    if self.total_frames % 60 == 0:
                        try:
                            PILImage.fromarray(data).save(
                                "/tmp/game_frame.jpg")
                        except Exception:
                            pass
            except Exception as e:
                if self.total_frames % 1000 == 0:
                    print(f"[STREAM] Error: {e}", flush=True)

        # State file
        if self.total_frames % 200 == 0:
            elapsed = time.time() - self.start_time
            fps_train = self.total_frames / max(elapsed, 1)
            avg_rew = (sum(self._ep_rewards) / len(self._ep_rewards)
                       if self._ep_rewards else 0)
            try:
                with open("/tmp/racer_state.json", "w") as f:
                    json.dump({
                        "timesteps": self.num_timesteps,
                        "best_distance": round(self.best_distance, 1),
                        "best_laps": self.best_laps,
                        "elapsed": round(elapsed),
                        "fps": round(fps_train, 1),
                        "avg_reward": round(avg_rew, 2),
                        "algorithm": "SAC+CNN v3 (GPU)",
                        "game": "SuperTuxKart",
                    }, f)
            except Exception:
                pass

        # Autosave
        if (self.num_timesteps > 0
                and self.num_timesteps % SAVE_INTERVAL == 0
                and self.num_timesteps != self._last_save_step):
            self._last_save_step = self.num_timesteps
            try:
                self.model.save(CHECKPOINT_PATH)
                self.model.save_replay_buffer(CHECKPOINT_PATH + "_buffer")
                avg = (sum(self._ep_rewards) / len(self._ep_rewards)
                       if self._ep_rewards else 0)
                print(
                    f"[SAVE] {self.num_timesteps:,} steps | "
                    f"best={self.best_distance:.0f} | "
                    f"avg_rew={avg:.1f}",
                    flush=True)
            except Exception as e:
                print(f"[SAVE] Error: {e}", flush=True)

        return True


# ── Main ───────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STK] SAC+CNN v3 on {device.upper()}", flush=True)
    if device == "cuda":
        print(f"[STK] GPU: {torch.cuda.get_device_name()}", flush=True)

    proxy_sock = connect_proxy()

    # Low-res render for speed — upscaled to 1280x720 for stream
    gfx = pystk2.GraphicsConfig.hd()
    gfx.screen_width = RENDER_W
    gfx.screen_height = RENDER_H
    pystk2.init(gfx)

    tracks = pystk2.list_tracks()
    track = "lighthouse" if "lighthouse" in tracks else tracks[0]
    print(f"[STK] Track: {track} | render={RENDER_W}x{RENDER_H} | "
          f"obs={FRAME_STACK}x{OBS_W}x{OBS_H} gray | skip={FRAME_SKIP}",
          flush=True)

    env = STKImageEnv(track=track)

    checkpoint_file = CHECKPOINT_PATH + ".zip"
    if os.path.exists(checkpoint_file):
        print("[STK] Loading checkpoint...", flush=True)
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
        buffer_file = CHECKPOINT_PATH + "_buffer.pkl"
        if os.path.exists(buffer_file):
            model.load_replay_buffer(buffer_file)
            print(f"[STK] Restored {model.replay_buffer.size()} experiences",
                  flush=True)
        else:
            print("[STK] No buffer found", flush=True)
    else:
        print("[STK] Fresh start", flush=True)
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        model = SAC(
            "CnnPolicy", env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            learning_starts=1000,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            target_entropy=-1.0,
            policy_kwargs={
                "features_extractor_class": STKCnn,
                "features_extractor_kwargs": {"features_dim": 256},
                "net_arch": [256, 256],
                "normalize_images": False,
            },
        )

    cb = StreamCallback()
    cb.proxy_sock = proxy_sock

    print("[STK] Training started...", flush=True)
    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=cb,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[STK] Interrupted", flush=True)

    model.save(CHECKPOINT_PATH)
    try:
        model.save_replay_buffer(CHECKPOINT_PATH + "_buffer")
    except Exception:
        pass
    print("[STK] Final save done", flush=True)
    env.close()


if __name__ == "__main__":
    main()
