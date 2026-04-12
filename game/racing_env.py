"""
racing_env.py — Gymnasium environment for PPO training.
Handles car physics, track, rewards, and observations.
"""
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .track import make_oval_track, point_on_track, progress_on_track
from .car import Car


class RacingEnv(gym.Env):
    """Single-car racing environment for PPO training."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Track
        self.track = make_oval_track()
        start = self.track["centers"][0]

        # Action space: [steer, accel, brake] all continuous
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Observation: 7 raycasts + speed + sin(angle) + cos(angle) = 10
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        self.car = None
        self.steps = 0
        self.max_steps = 2000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start = self.track["centers"][0]

        # Compute initial angle from first two waypoints
        next_pt = self.track["centers"][1]
        angle = math.degrees(math.atan2(next_pt[1] - start[1], next_pt[0] - start[0]))

        self.car = Car(start[0], start[1], angle)
        self.car.cast_rays(self.track)
        self.steps = 0
        self._prev_progress = 0
        self._prev_segment = 0

        return self.car.get_observation(), {}

    def step(self, action):
        self.steps += 1
        steer, accel, brake = float(action[0]), float(action[1]), float(action[2])

        # Update car
        self.car.update(steer, accel, brake)
        self.car.cast_rays(self.track)

        # Check if on track
        on_track, seg_idx, dist_from_center = point_on_track(self.track, self.car.position)

        # Progress reward
        progress = progress_on_track(self.track, seg_idx)

        # Detect forward movement (handle wrap-around at start/finish)
        seg_diff = seg_idx - self._prev_segment
        if seg_diff < -self.track["n_points"] // 2:
            seg_diff += self.track["n_points"]  # crossed finish line forward
        elif seg_diff > self.track["n_points"] // 2:
            seg_diff -= self.track["n_points"]  # went backward past start

        # Reward
        reward = 0.0

        # Forward progress reward (main signal)
        reward += seg_diff * 1.0

        # Speed bonus (small, encourages going fast)
        reward += self.car.speed / Car.MAX_SPEED * 0.1

        # Center lane bonus (small, encourages staying centered)
        center_ratio = 1.0 - (dist_from_center / (self.track["width"] / 2))
        reward += max(0, center_ratio) * 0.05

        # Lap completion bonus
        if seg_diff > 0 and self._prev_segment > self.track["n_points"] * 0.9 and seg_idx < self.track["n_points"] * 0.1:
            self.car.laps += 1
            reward += 50.0
            print(f"  LAP {self.car.laps} completed!", flush=True)

        # Penalties
        terminated = False
        truncated = False

        if not on_track:
            reward -= 10.0
            terminated = True
            self.car.alive = False

        if self.car.speed < 0.1 and self.steps > 30:
            reward -= 0.5  # penalize standing still

        if self.steps >= self.max_steps:
            truncated = True

        self._prev_progress = progress
        self._prev_segment = seg_idx

        obs = self.car.get_observation()
        info = {
            "laps": self.car.laps,
            "segment": seg_idx,
            "speed": self.car.speed,
            "distance": self.car.distance_traveled,
            "on_track": on_track,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        import pygame
        W, H = 1280, 720
        surf = pygame.Surface((W, H))
        surf.fill((20, 20, 30))

        # Draw track
        inner = self.track["inner"].astype(int)
        outer = self.track["outer"].astype(int)
        pygame.draw.lines(surf, (60, 60, 80), True, inner.tolist(), 2)
        pygame.draw.lines(surf, (60, 60, 80), True, outer.tolist(), 2)

        # Draw car
        if self.car:
            cx, cy = int(self.car.x), int(self.car.y)
            # Car body
            pygame.draw.circle(surf, (0, 255, 255), (cx, cy), 8)
            # Direction
            rad = math.radians(self.car.angle)
            ex = cx + int(math.cos(rad) * 15)
            ey = cy + int(math.sin(rad) * 15)
            pygame.draw.line(surf, (255, 0, 128), (cx, cy), (ex, ey), 2)

            # Raycasts
            for ep in self.car.ray_endpoints:
                pygame.draw.line(surf, (50, 50, 70), (cx, cy), (int(ep[0]), int(ep[1])), 1)

        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
