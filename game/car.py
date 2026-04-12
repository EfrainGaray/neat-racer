"""
car.py — Car physics with raycasting for obstacle detection.
Simple 2D car: position, velocity, angle. Raycasts detect distance to track walls.
"""
import math
import numpy as np


class Car:
    MAX_SPEED = 8.0
    ACCEL     = 0.3
    BRAKE     = 0.5
    FRICTION  = 0.02
    TURN_RATE = 4.0    # degrees per frame at full steering
    N_RAYS    = 7      # number of raycasts (spread across 180 degrees forward)

    def __init__(self, x: float, y: float, angle: float = 0):
        self.x     = x
        self.y     = y
        self.angle = angle   # degrees, 0 = right
        self.speed = 0.0
        self.alive = True
        self.distance_traveled = 0.0
        self.last_segment = 0
        self.laps  = 0
        self.ray_distances = np.ones(self.N_RAYS)  # normalized [0, 1]

    def update(self, steer: float, accel: float, brake: float):
        """Update car physics. steer in [-1, 1], accel/brake in [0, 1]."""
        if not self.alive:
            return

        # Steering (proportional to speed for realism)
        speed_factor = min(self.speed / 3.0, 1.0)
        self.angle += steer * self.TURN_RATE * (0.3 + 0.7 * speed_factor)

        # Acceleration / braking
        self.speed += accel * self.ACCEL
        self.speed -= brake * self.BRAKE
        self.speed -= self.speed * self.FRICTION
        self.speed = max(0, min(self.speed, self.MAX_SPEED))

        # Movement
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.speed
        dy = math.sin(rad) * self.speed
        self.x += dx
        self.y += dy
        self.distance_traveled += self.speed

    def cast_rays(self, track: dict, ray_length: float = 200):
        """Cast N_RAYS from car position and return normalized distances to walls."""
        angles = np.linspace(-90, 90, self.N_RAYS) + self.angle
        self.ray_distances = np.ones(self.N_RAYS)
        self.ray_endpoints = []

        for i, a in enumerate(angles):
            rad = math.radians(a)
            dx = math.cos(rad)
            dy = math.sin(rad)

            # Step along ray
            for step in range(1, int(ray_length)):
                px = self.x + dx * step
                py = self.y + dy * step

                # Check if point is outside track
                pos = np.array([px, py])
                diffs = track["centers"] - pos
                dists = np.linalg.norm(diffs, axis=1)
                min_dist = dists.min()

                if min_dist > track["width"] / 2:
                    self.ray_distances[i] = step / ray_length
                    self.ray_endpoints.append((px, py))
                    break
            else:
                self.ray_endpoints.append((self.x + dx * ray_length, self.y + dy * ray_length))

    def get_observation(self) -> np.ndarray:
        """Return observation vector for the neural network."""
        return np.concatenate([
            self.ray_distances,                        # 7 raycasts
            [self.speed / self.MAX_SPEED],             # normalized speed
            [math.sin(math.radians(self.angle))],      # angle sin
            [math.cos(math.radians(self.angle))],      # angle cos
        ]).astype(np.float32)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
