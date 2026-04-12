"""
stream_train.py — PPO training with LIVE stream rendering.
Trains AND renders simultaneously — viewers see cars learning in real-time.
Outputs frames to stream_proxy via TCP socket (same pipeline as flappy-neat).

Run on Fedora: SDL_VIDEODRIVER=offscreen python3 training/stream_train.py
"""
import sys, os, math, time, json, socket, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from game.racing_env import RacingEnv
from game.track import make_oval_track

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 60
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9998
N_ENVS = 16

# ── Cyberpunk palette ─────────────────────────────────────────────────────────
BG          = (8, 6, 18)
TRACK_ASPHALT = (30, 32, 40)
TRACK_EDGE_IN  = (0, 200, 255)
TRACK_EDGE_OUT = (255, 0, 128)
LANE_MARK   = (60, 65, 80)
NEON_CYAN   = (0, 255, 255)
NEON_PINK   = (255, 0, 128)
NEON_PURPLE = (140, 0, 255)
WHITE       = (255, 255, 255)
GRAY        = (100, 110, 120)
GREEN       = (30, 215, 96)
RED         = (239, 68, 68)
YELLOW      = (255, 200, 50)
ORANGE      = (255, 120, 30)

CAR_COLORS = [
    (0, 255, 255), (255, 0, 128), (140, 0, 255), (255, 200, 50),
    (30, 215, 96), (255, 100, 50), (100, 200, 255), (255, 150, 200),
    (150, 255, 100), (255, 80, 80), (80, 255, 200), (200, 100, 255),
    (255, 255, 100), (100, 255, 255), (255, 100, 255), (200, 200, 200),
]

# ── Particles ─────────────────────────────────────────────────────────────────
class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'color', 'r')
    def __init__(self, x, y, color, speed=2):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(angle) * random.uniform(0.5, speed)
        self.vy = math.sin(angle) * random.uniform(0.5, speed)
        self.life = random.randint(15, 35)
        self.color = color
        self.r = random.uniform(1.5, 3)
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.r *= 0.96
    @property
    def alive(self):
        return self.life > 0

# ── Stars background ──────────────────────────────────────────────────────────
def make_stars(n=120):
    return [(random.randint(0, WIDTH), random.randint(0, HEIGHT),
             random.uniform(0.3, 1.0), random.randint(1, 2)) for _ in range(n)]

# ── Connection ────────────────────────────────────────────────────────────────
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


# ── Renderer ──────────────────────────────────────────────────────────────────
class StreamCallback(BaseCallback):
    """SB3 callback that renders training to stream on every step."""

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.proxy_sock = None
        self.surface = None
        self.font = None
        self.font_sm = None
        self.font_xs = None
        self.clock = None
        self.frame = 0
        self.best_distance = 0
        self.best_laps = 0
        self.episodes = 0
        self.start_time = time.time()
        self.particles = []
        self.stars = []
        self.prev_alive = [True] * N_ENVS
        self.trail_history = [[] for _ in range(N_ENVS)]

        self._track_surface = None
        self._inited = False

    def _init_render(self):
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        pygame.init()
        self.surface = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_sm = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_xs = pygame.font.SysFont("monospace", 11)
        self.clock = pygame.time.Clock()
        self.proxy_sock = connect_proxy()
        self.stars = make_stars()
        self._build_track_surface()

    def _on_training_start(self):
        self._init_render()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.render(infos)
        if self.num_timesteps % 60 == 0:
            self.write_state()
        return True

    def _on_rollout_end(self):
        self.episodes += 1

    def _build_track_surface(self):
        """Pre-render track to a surface (expensive, do once)."""
        self._track_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        track = self.track
        inner = track["inner"].astype(int).tolist()
        outer = track["outer"].astype(int).tolist()

        # Track fill
        all_pts = outer + inner[::-1]
        try:
            pygame.draw.polygon(self._track_surface, (*TRACK_ASPHALT, 255), all_pts)
        except Exception:
            pass

        # Lane markings (dashed center line)
        centers = track["centers"]
        for i in range(0, len(centers), 6):
            if i % 12 < 6:
                j = (i + 3) % len(centers)
                p1, p2 = centers[i].astype(int), centers[j].astype(int)
                pygame.draw.line(self._track_surface, (*LANE_MARK, 180),
                                 tuple(p1), tuple(p2), 1)

        # Outer glow (thick translucent lines)
        glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.lines(glow, (*TRACK_EDGE_OUT, 30), True, outer, 8)
        pygame.draw.lines(glow, (*TRACK_EDGE_IN, 30), True, inner, 8)
        self._track_surface.blit(glow, (0, 0))

        # Sharp edge lines
        pygame.draw.lines(self._track_surface, TRACK_EDGE_IN, True, inner, 2)
        pygame.draw.lines(self._track_surface, TRACK_EDGE_OUT, True, outer, 2)

        # Start/finish checkered pattern
        s_in = track["inner"][0]
        s_out = track["outer"][0]
        pygame.draw.line(self._track_surface, WHITE,
                         (int(s_in[0]), int(s_in[1])),
                         (int(s_out[0]), int(s_out[1])), 4)

    def render(self, infos):
        self.frame += 1
        surf = self.surface
        surf.fill(BG)

        # Stars
        for sx, sy, bright, sr in self.stars:
            a = int(bright * (140 + 40 * math.sin(self.frame * 0.02 + sx)))
            c = min(255, a)
            pygame.draw.circle(surf, (c, c, int(c * 0.9)), (sx, sy), sr)

        # Track
        surf.blit(self._track_surface, (0, 0))

        # Update trails and draw cars
        for i, info in enumerate(infos[:N_ENVS]):
            if not info:
                continue
            x, y = info.get("x", 0), info.get("y", 0)
            angle = info.get("angle", 0)
            speed = info.get("speed", 0)
            alive = info.get("alive", True)
            color = CAR_COLORS[i % len(CAR_COLORS)]

            # Trail
            trail = self.trail_history[i]
            if alive and speed > 0.5:
                trail.append((x, y))
                if len(trail) > 30:
                    trail.pop(0)
            elif not alive:
                trail.clear()

            # Draw trail (fading)
            for j in range(1, len(trail)):
                alpha = int(j / len(trail) * 120)
                t_col = (color[0], color[1], color[2], alpha)
                t_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(t_surf, t_col, (2, 2), 2)
                surf.blit(t_surf, (int(trail[j][0]) - 2, int(trail[j][1]) - 2))

            if not alive:
                # Crash effect
                if self.prev_alive[i]:
                    for _ in range(12):
                        self.particles.append(Particle(x, y, color, speed=3))
                    self.prev_alive[i] = False
                continue

            self.prev_alive[i] = True

            # Draw car body (rotated rectangle)
            car_w, car_h = 20, 10
            car_surf = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
            # Body
            pygame.draw.rect(car_surf, color, (0, 0, car_w, car_h), border_radius=3)
            # Windshield
            pygame.draw.rect(car_surf, (255, 255, 255, 100), (car_w - 7, 2, 5, car_h - 4), border_radius=1)
            # Headlight
            pygame.draw.circle(car_surf, (255, 255, 200), (car_w - 2, car_h // 2), 2)

            rotated = pygame.transform.rotate(car_surf, -angle)
            rect = rotated.get_rect(center=(int(x), int(y)))
            surf.blit(rotated, rect)

            # Car glow
            glow_surf = pygame.Surface((36, 36), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 25), (18, 18), 18)
            surf.blit(glow_surf, (int(x) - 18, int(y) - 18))

            # Speed bar next to car
            if speed > 1:
                bar_w = int(speed / 8 * 20)
                bar_surf = pygame.Surface((bar_w, 3), pygame.SRCALPHA)
                bar_surf.fill((*GREEN, 150) if speed < 5 else (*ORANGE, 150))
                surf.blit(bar_surf, (int(x) - bar_w // 2, int(y) + 12))

        # Particles
        for p in self.particles:
            p.update()
            if p.alive:
                a = int(p.life / 35 * 200)
                ps = pygame.Surface((int(p.r * 4), int(p.r * 4)), pygame.SRCALPHA)
                pygame.draw.circle(ps, (*p.color, a), (int(p.r * 2), int(p.r * 2)), max(1, int(p.r)))
                surf.blit(ps, (int(p.x) - int(p.r * 2), int(p.y) - int(p.r * 2)))
        self.particles = [p for p in self.particles if p.alive]

        # Stats overlay
        self._draw_overlay(surf, infos)

        # Send to proxy
        if self.proxy_sock:
            try:
                frame_data = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
                self.proxy_sock.sendall(frame_data.tobytes())
            except OSError:
                self.proxy_sock = connect_proxy(retries=3, delay=1)

        # Web preview
        if self.frame % 15 == 0:
            try:
                pygame.image.save(surf, "/tmp/game_frame.jpg")
            except Exception:
                pass

        self.clock.tick(FPS)

    def _draw_overlay(self, surf, infos):
        elapsed = time.time() - self.start_time
        mins, secs = int(elapsed // 60), int(elapsed % 60)

        # Count alive
        alive_count = sum(1 for i in infos[:N_ENVS] if i and i.get("alive", False))
        best_speed = max((i.get("speed", 0) for i in infos[:N_ENVS] if i), default=0)

        # Update bests
        for info in infos[:N_ENVS]:
            if info:
                d = info.get("distance", 0)
                l = info.get("laps", 0)
                if d > self.best_distance:
                    self.best_distance = d
                if l > self.best_laps:
                    self.best_laps = l

        lines = [
            ("STEPS",    f"{self.timesteps:,}",       NEON_CYAN),
            ("ALIVE",    f"{alive_count}/{N_ENVS}",   GREEN if alive_count > 8 else RED),
            ("BEST DIST",f"{self.best_distance:.0f}", YELLOW),
            ("LAPS",     f"{self.best_laps}",         NEON_PINK if self.best_laps > 0 else GRAY),
            ("TOP SPEED",f"{best_speed:.1f}",         ORANGE),
            ("TIME",     f"{mins:02d}:{secs:02d}",    GRAY),
        ]

        pad = 12
        bw, bh = 220, len(lines) * 30 + pad * 2 + 36
        box = pygame.Surface((bw, bh), pygame.SRCALPHA)
        box.fill((8, 6, 18, 210))
        surf.blit(box, (16, 16))

        # Border glow
        border = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(border, (*NEON_CYAN, 40), (0, 0, bw, bh), 1)
        surf.blit(border, (16, 16))

        # Title pill
        title = self.font_sm.render("PPO TRAINING", True, NEON_PURPLE)
        pill_w = title.get_width() + 12
        pill = pygame.Surface((pill_w, 20), pygame.SRCALPHA)
        pill.fill((*NEON_PURPLE, 30))
        surf.blit(pill, (16 + pad, 16 + pad))
        surf.blit(title, (16 + pad + 6, 16 + pad + 2))

        for i, (label, value, color) in enumerate(lines):
            y = 16 + pad + 30 + i * 30
            lbl = self.font_xs.render(label, True, (138, 143, 152))
            val = self.font.render(value, True, color)
            surf.blit(lbl, (16 + pad, y + 2))
            surf.blit(val, (16 + bw - pad - val.get_width(), y))

        # Bottom tag
        tag = self.font_xs.render("NEAT RACER — AI Learning Live", True, (70, 75, 85))
        surf.blit(tag, (16, HEIGHT - 24))

        # FPS
        fps_txt = self.font_xs.render(f"{self.clock.get_fps():.0f} fps", True, (50, 55, 65))
        surf.blit(fps_txt, (WIDTH - fps_txt.get_width() - 16, HEIGHT - 24))

    def write_state(self):
        try:
            with open("/tmp/racer_state.json", "w") as f:
                json.dump({
                    "timesteps": self.timesteps,
                    "episodes": self.episodes,
                    "best_distance": round(self.best_distance, 1),
                    "best_laps": self.best_laps,
                    "elapsed": round(time.time() - self.start_time, 0),
                    "fps": round(self.clock.get_fps(), 1),
                }, f)
        except Exception:
            pass


def main():
    print("[STREAM] NEAT Racer — PPO Training + Live Stream", flush=True)

    track = make_oval_track()

    print(f"[STREAM] Creating {N_ENVS} parallel environments...", flush=True)
    env = make_vec_env(RacingEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy", env, verbose=0, device="cpu",
        learning_rate=3e-4, n_steps=2048, batch_size=256,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
    )

    callback = StreamCallback(track)
    callback._init_render()

    print("[STREAM] Training + rendering live (callback mode)...", flush=True)
    try:
        model.learn(
            total_timesteps=100_000_000,
            callback=callback,
        )
    except KeyboardInterrupt:
        print("\n[STREAM] Stopped", flush=True)

    model.save("/tmp/racer_model")
    env.close()


if __name__ == "__main__":
    main()
