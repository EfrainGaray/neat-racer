# CLAUDE.md — neat-racer

## Fase: `xp`

Stream 24/7 de IA aprendiendo a correr carreras. PPO en GPU (RTX 4070 Ti Super).
Viewers controlan pistas, dificultad, y desafíos via votaciones.

## Stack

- **AI**: PPO (Proximal Policy Optimization) via Stable Baselines3 + PyTorch (GPU)
- **Rendering**: Godot 4 (Vulkan) — efectos, partículas, iluminación
- **Comunicación**: TCP socket Python ↔ Godot
- **Stream**: ffmpeg NVENC → Kick + Twitch (mismo pipeline flappy-neat)
- **Chat AI**: Kira (compartida con flappy-neat)
- **Dashboard**: Astro SSR (compartido, extendido)

## Arquitectura

```
PyTorch PPO (GPU, Python)
    ├── Gym environment (car physics, raycasts)
    ├── Training loop (paralelo, 50+ autos)
    └── TCP server → Godot
                        ├── Rendering (Vulkan)
                        ├── Shaders, bloom, particles
                        └── frames → ffmpeg → Kick+Twitch
```

## Entorno de IA

- **Observaciones**: 5 raycasts (distancia a paredes) + velocidad + ángulo + posición en pista
- **Acciones** (continuas): steering [-1, 1], acceleration [0, 1], brake [0, 1]
- **Reward**: distancia recorrida en pista + bonus por completar vueltas - penalización por chocar
- **Episode**: termina al chocar o timeout

## Reglas de desarrollo

- Mismas reglas globales de CLAUDE.md raíz
- GPU-first: todo entrenamiento en RTX 4070 Ti Super
- Commits atómicos
- NUNCA reiniciar el training innecesariamente — pierde progreso
