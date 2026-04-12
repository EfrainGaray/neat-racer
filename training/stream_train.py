"""
stream_train.py — PPO training with LIVE stream rendering.
Trains AND renders simultaneously — viewers see cars learning in real-time.
Outputs frames to stream_proxy via TCP socket (same pipeline as flappy-neat).

Run on Fedora: SDL_VIDEODRIVER=offscreen python3 training/stream_train.py
"""
import sys, os, math, time, json, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from game.racing_env import RacingEnv
from game.track import make_oval_track
from game.car import Car

# ── Stream config ─────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 60
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9998
N_DISPLAY_CARS = 16  # cars visible on screen

# ── Colors (cyberpunk palette) ────────────────────────────────────────────────
BG_COLOR    = (10, 8, 20)
TRACK_INNER = (40, 45, 60)
TRACK_OUTER = (40, 45, 60)
TRACK_FILL  = (25, 28, 35)
NEON_CYAN   = (0, 255, 255)
NEON_PINK   = (255, 0, 128)
NEON_PURPLE = (180, 0, 255)
WHITE       = (255, 255, 255)
GRAY        = (120, 130, 140)
GREEN       = (30, 215, 96)
RED         = (239, 68, 68)
YELLOW      = (255, 200, 50)

CAR_COLORS = [
    (0, 255, 255), (255, 0, 128), (180, 0, 255), (255, 200, 50),
    (30, 215, 96), (255, 100, 50), (100, 200, 255), (255, 150, 200),
    (150, 255, 100), (255, 80, 80), (80, 255, 200), (200, 100, 255),
    (255, 255, 100), (100, 255, 255), (255, 100, 255), (200, 200, 200),
]


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
    print("[STREAM] Could not connect to proxy — rendering without stream", flush=True)
    return None


class StreamCallback(BaseCallback):
    """Callback that renders the training to stream at 60fps."""

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.proxy_sock = None
        self.surface = None
        self.font = None
        self.font_small = None
        self.clock = None
        self.frame = 0
        self.best_distance = 0
        self.best_laps = 0
        self.episode_count = 0
        self.start_time = time.time()

    def _on_training_start(self):
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        pygame.init()
        self.surface = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.clock = pygame.time.Clock()
        self.proxy_sock = connect_proxy()

    def _on_step(self) -> bool:
        self.frame += 1

        # Get current car states from vectorized env
        infos = self.locals.get("infos", [])

        # Update best stats
        for info in infos:
            d = info.get("distance", 0)
            l = info.get("laps", 0)
            if d > self.best_distance:
                self.best_distance = d
            if l > self.best_laps:
                self.best_laps = l

        # Render every frame for smooth stream (but env steps are batched)
        if self.frame % 1 == 0:
            self._render(infos)

        # Write state for dashboard
        if self.frame % 60 == 0:
            self._write_state()

        return True

    def _on_rollout_end(self):
        self.episode_count += 1

    def _render(self, infos):
        surf = self.surface
        surf.fill(BG_COLOR)

        # Draw track
        self._draw_track(surf)

        # Draw cars from env observations
        try:
            # Get current observations to infer car positions
            # We render based on info dict which has positions
            for i, info in enumerate(infos[:N_DISPLAY_CARS]):
                if not info:
                    continue
                color = CAR_COLORS[i % len(CAR_COLORS)]
                # We need car positions — they're in the env internals
                # For now, draw based on segment position on track
                seg = info.get("segment", 0)
                speed = info.get("speed", 0)
                on_track = info.get("on_track", True)

                if seg < len(self.track["centers"]):
                    cx, cy = self.track["centers"][seg]
                    # Car dot
                    r = 6 if on_track else 4
                    alpha = 255 if on_track else 100
                    pygame.draw.circle(surf, color, (int(cx), int(cy)), r)
                    # Speed indicator
                    if speed > 0 and on_track:
                        # Direction from track tangent
                        next_seg = (seg + 1) % len(self.track["centers"])
                        nx, ny = self.track["centers"][next_seg]
                        angle = math.atan2(ny - cy, nx - cx)
                        ex = cx + math.cos(angle) * (10 + speed * 3)
                        ey = cy + math.sin(angle) * (10 + speed * 3)
                        pygame.draw.line(surf, color, (int(cx), int(cy)), (int(ex), int(ey)), 2)
        except Exception:
            pass

        # Draw overlay
        self._draw_overlay(surf)

        # Send to stream proxy
        if self.proxy_sock:
            try:
                frame_data = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
                self.proxy_sock.sendall(frame_data.tobytes())
            except OSError:
                self.proxy_sock = None

        # Save preview frame for web dashboard
        if self.frame % 15 == 0:
            try:
                pygame.image.save(surf, "/tmp/game_frame.jpg")
            except Exception:
                pass

        self.clock.tick(FPS)

    def _draw_track(self, surf):
        track = self.track

        # Fill track area
        inner_pts = track["inner"].astype(int).tolist()
        outer_pts = track["outer"].astype(int).tolist()

        # Draw track surface (filled polygon between inner and outer)
        all_pts = outer_pts + inner_pts[::-1]
        try:
            pygame.draw.polygon(surf, TRACK_FILL, all_pts)
        except Exception:
            pass

        # Track edges with neon glow
        pygame.draw.lines(surf, NEON_CYAN, True, inner_pts, 2)
        pygame.draw.lines(surf, NEON_PINK, True, outer_pts, 2)

        # Start/finish line
        s_in = track["inner"][0]
        s_out = track["outer"][0]
        pygame.draw.line(surf, WHITE, (int(s_in[0]), int(s_in[1])),
                         (int(s_out[0]), int(s_out[1])), 3)

    def _draw_overlay(self, surf):
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        lines = [
            ("TIMESTEPS", f"{self.num_timesteps:,}", NEON_CYAN),
            ("EPISODES", f"{self.episode_count}", WHITE),
            ("BEST DIST", f"{self.best_distance:.0f}", GREEN),
            ("BEST LAPS", f"{self.best_laps}", YELLOW),
            ("TIME", f"{mins:02d}:{secs:02d}", GRAY),
        ]

        pad = 12
        bw, bh = 220, len(lines) * 32 + pad * 2 + 30
        box = pygame.Surface((bw, bh), pygame.SRCALPHA)
        box.fill((10, 10, 15, 200))
        surf.blit(box, (16, 16))
        pygame.draw.rect(surf, (255, 255, 255, 20), (16, 16, bw, bh), 1)

        # Title
        title = self.font_small.render("PPO TRAINING", True, NEON_PURPLE)
        surf.blit(title, (16 + pad, 16 + pad))

        for i, (label, value, color) in enumerate(lines):
            y = 16 + pad + 24 + i * 32
            lbl = self.font_small.render(label, True, (138, 143, 152))
            val = self.font.render(value, True, color)
            surf.blit(lbl, (16 + pad, y))
            surf.blit(val, (16 + bw - pad - val.get_width(), y))

        # Bottom: "NEAT RACER — AI Learning Live"
        tag = self.font_small.render("NEAT RACER — AI Learning Live", True, (98, 102, 109))
        surf.blit(tag, (16, HEIGHT - 28))

    def _write_state(self):
        try:
            state = {
                "timesteps": self.num_timesteps,
                "episodes": self.episode_count,
                "best_distance": round(self.best_distance, 1),
                "best_laps": self.best_laps,
                "elapsed": round(time.time() - self.start_time, 0),
                "fps": round(self.clock.get_fps(), 1),
            }
            with open("/tmp/racer_state.json", "w") as f:
                json.dump(state, f)
        except Exception:
            pass


def main():
    print("[STREAM] NEAT Racer — PPO Training + Live Stream", flush=True)
    print(f"[STREAM] Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps", flush=True)

    track = make_oval_track()

    n_envs = 16
    print(f"[STREAM] Creating {n_envs} parallel environments...", flush=True)
    env = make_vec_env(RacingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
    )

    callback = StreamCallback(track)

    print("[STREAM] Starting training + rendering...", flush=True)
    try:
        model.learn(
            total_timesteps=50_000_000,
            callback=callback,
        )
    except KeyboardInterrupt:
        print("\n[STREAM] Interrupted", flush=True)

    model.save("/tmp/racer_model")
    print("[STREAM] Model saved", flush=True)
    env.close()


if __name__ == "__main__":
    main()
