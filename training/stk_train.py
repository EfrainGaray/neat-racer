"""
stk_train.py — PPO training on SuperTuxKart with LIVE stream rendering.
Uses pystk2 directly for rendering + gymnasium wrapper for training.

Run on Fedora: python3 training/stk_train.py
"""
import sys, os, math, time, json, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pystk2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 30
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9998
N_ENVS = 1  # pystk2 runs single-process (GPU renderer)


# ── STK Gym Environment ──────────────────────────────────────────────────────
class STKEnv(gym.Env):
    """SuperTuxKart environment using pystk2 directly."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, track="lighthouse", num_karts=5, laps=1):
        super().__init__()
        self.track_name = track
        self.num_karts = num_karts
        self.laps = laps
        self._race = None
        self._steps = 0
        self._max_steps = 1500
        self._prev_distance = 0
        self._last_image = None

        # Action: [steer (-1 to 1), accel_brake (-1 to 1)]
        # negative accel_brake = brake, positive = accelerate
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation: key game state values (no image for training speed)
        # [distance_down_track, velocity_x, velocity_y, velocity_z,
        #  front_x, front_y, front_z, center_path_distance,
        #  center_path_x, center_path_y, center_path_z]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._race is not None:
            self._race.stop()
            del self._race

        race_config = pystk2.RaceConfig()
        race_config.track = self.track_name
        race_config.num_kart = self.num_karts
        race_config.laps = self.laps
        race_config.players.append(
            pystk2.PlayerConfig(
                controller=pystk2.PlayerConfig.Controller.PLAYER_CONTROL,
                team=0,
            )
        )

        self._race = pystk2.Race(race_config)
        self._race.start()
        self._race.step()

        self._world = pystk2.WorldState()
        self._world.update()

        self._steps = 0
        self._prev_distance = 0

        obs = self._get_obs()
        self._last_image = self._get_image()
        return obs, {}

    def step(self, action):
        self._steps += 1
        steer = float(action[0])
        accel_brake = float(action[1])

        # Convert to pystk2 action
        pystk_action = pystk2.Action()
        pystk_action.steer = np.clip(steer, -1, 1)
        if accel_brake >= 0:
            pystk_action.acceleration = accel_brake
            pystk_action.brake = False
        else:
            pystk_action.acceleration = 0
            pystk_action.brake = True
        pystk_action.drift = abs(steer) > 0.8  # auto-drift on sharp turns
        pystk_action.nitro = accel_brake > 0.9  # nitro on full throttle

        self._race.step(pystk_action)
        self._world.update()

        obs = self._get_obs()
        self._last_image = self._get_image()

        # Reward
        kart = self._world.karts[0]
        distance = kart.distance_down_track
        velocity = kart.speed

        # Forward progress (main reward)
        progress = distance - self._prev_distance
        if progress < -100:  # crossed finish line
            progress += 1000  # approximate track length
        reward = progress * 0.1

        # Speed bonus
        reward += velocity * 0.005

        # On-road bonus
        if kart.is_on_road:
            reward += 0.05
        else:
            reward -= 0.5

        self._prev_distance = distance

        # Termination
        terminated = False
        truncated = False

        if kart.has_finished_race:
            reward += 100
            terminated = True

        if self._steps >= self._max_steps:
            truncated = True

        # Stuck detection
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
        kart = self._world.karts[0]
        return np.array([
            kart.distance_down_track / 1000.0,
            kart.velocity[0], kart.velocity[1], kart.velocity[2],
            kart.front[0] / 100, kart.front[1] / 100, kart.front[2] / 100,
            kart.overall_distance / 1000.0,
            kart.location[0] / 100, kart.location[1] / 100, kart.location[2] / 100,
        ], dtype=np.float32)

    def _get_image(self):
        try:
            return np.array(self._race.render_data[0].image)
        except Exception:
            return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    def render(self):
        return self._last_image

    def close(self):
        if self._race:
            try:
                self._race.stop()
            except Exception:
                pass
        try:
            pystk2.clean()
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
    """Renders STK frames to stream during training."""

    def __init__(self):
        super().__init__()
        self.proxy_sock = None
        self.start_time = time.time()
        self.best_distance = 0
        self.frame_count = 0

    def _on_training_start(self):
        pass

    def _on_step(self) -> bool:
        self.frame_count += 1

        # Get image from env
        try:
            env = self.training_env.envs[0]
            image = env.render()
            if image is not None and image.size > 0:
                # Resize to stream resolution if needed
                if image.shape[:2] != (HEIGHT, WIDTH):
                    from PIL import Image
                    img = Image.fromarray(image).resize((WIDTH, HEIGHT))
                    image = np.array(img)

                # Send to proxy
                if self.proxy_sock:
                    try:
                        self.proxy_sock.sendall(image.tobytes())
                    except OSError:
                        self.proxy_sock = connect_proxy(retries=2, delay=1)

                # Save preview
                if self.frame_count % 30 == 0:
                    try:
                        from PIL import Image
                        Image.fromarray(image).save("/tmp/game_frame.jpg")
                    except Exception:
                        pass
        except Exception as e:
            if self.frame_count % 100 == 0:
                print(f"[STREAM] Render error: {e}", flush=True)

        # Update stats
        infos = self.locals.get("infos", [])
        for info in infos:
            d = info.get("distance", 0)
            if d > self.best_distance:
                self.best_distance = d

        # Write state
        if self.frame_count % 60 == 0:
            elapsed = time.time() - self.start_time
            try:
                with open("/tmp/racer_state.json", "w") as f:
                    json.dump({
                        "timesteps": self.num_timesteps,
                        "best_distance": round(self.best_distance, 1),
                        "elapsed": round(elapsed),
                        "game": "SuperTuxKart",
                    }, f)
            except Exception:
                pass

        return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[STK] SuperTuxKart PPO Training + Live Stream", flush=True)
    print(f"[STK] Resolution: {WIDTH}x{HEIGHT}", flush=True)

    proxy_sock = connect_proxy()

    # Init pystk2 graphics before creating env
    gfx = pystk2.GraphicsConfig.hd()
    gfx.screen_width = WIDTH
    gfx.screen_height = HEIGHT
    pystk2.init(gfx)

    tracks = pystk2.list_tracks()
    print(f"[STK] Available tracks: {len(tracks)}", flush=True)
    track = "lighthouse" if "lighthouse" in tracks else tracks[0]
    print(f"[STK] Using: {track}", flush=True)

    env = STKEnv(track=track, num_karts=5, laps=1)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1500)

    model = PPO(
        "MlpPolicy", env, verbose=1, device="cpu",
        learning_rate=3e-4, n_steps=1024, batch_size=128,
        n_epochs=4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
    )

    callback = StreamCallback()
    callback.proxy_sock = proxy_sock

    print("[STK] Training live...", flush=True)
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=callback,
        )
    except KeyboardInterrupt:
        print("\n[STK] Stopped", flush=True)

    model.save("/tmp/stk_racer_model")
    env.close()
    print("[STK] Done", flush=True)


if __name__ == "__main__":
    main()
