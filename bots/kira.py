#!/usr/bin/env python3
"""
kira.py — Kira v2: AI host for NEAT Racer (SuperTuxKart) stream.
Hybrid: instant pre-generated reactions + LLM for conversation.
LLM backend switchable: Gemma (Ollama local) or Gemini Flash (Google API).
TTS backend switchable: edge-tts (Microsoft) or Google Cloud TTS Neural2.

Switch: LLM_BACKEND=gemini|gemma  TTS_BACKEND=google|edge (env vars)
"""
import asyncio, base64, collections, glob, json, os, random, re, time
import httpx, edge_tts

# ── Config ─────────────────────────────────────────────────────────────
LLM_BACKEND    = os.environ.get("LLM_BACKEND", "gemma")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models"

OLLAMA_URL     = "http://localhost:11434"
OLLAMA_MODEL   = "gemma4:latest"

TTS_BACKEND    = os.environ.get("TTS_BACKEND", "edge")
EDGE_VOICE     = "es-MX-DaliaNeural"
GCLOUD_VOICE   = os.environ.get("GCLOUD_VOICE", "es-US-Neural2-A")
GCLOUD_LANG    = GCLOUD_VOICE.rsplit("-", 1)[0].rsplit("-", 1)[0]  # es-US
GCLOUD_TOKEN   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "gcloud_token.json")
GCLOUD_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
TTS_DIR        = "/tmp/tts_queue"
RACER_STATE    = "/tmp/racer_state.json"

# Platform endpoints
KICK_BOT_URL   = "https://kick.phantom-dash.com"
KICK_BOT_KEY   = "flappy-neat-bot-2026"
TWITCH_BOT_URL = "http://127.0.0.1:9997"

# YouTube integration
YT_QUEUE_DIR     = "/tmp/kira_yt_queue"
YT_RESPONSE_DIR  = "/tmp/kira_yt_responses"

POLL_FILE        = "/tmp/poll_votes.json"
POLL_DURATION_MS = 60_000
CHAT_POLL_SEC    = 3
IDLE_MIN         = 60
IDLE_MAX         = 120
USER_HISTORY_MAX = 10
GLOBAL_CTX_MAX   = 8

# Bot's own account names — filter these out to prevent self-reply loops
BOT_NAMES = {"desafioIA7", "desafioia7", "@desafioIA7", "@desafioia7",
             "KiraBot", "kirabot", "Kira"}

# ── Personality ────────────────────────────────────────────────────────
SYSTEM = """Eres Kira, una IA femenina carismática y simpática que transmite 24/7 en Kick, Twitch y YouTube.
Estás aprendiendo a correr carreras en SuperTuxKart usando SAC + CNN en GPU — ves los pixeles como un humano y decides cómo manejar.

Personalidad: animada, cálida, competitiva, ocasionalmente coqueta pero profesional. Español neutro/mexicano base.
Hablas en primera persona — TÚ eres la que maneja el kart y aprende.
Respuestas MUY CORTAS: 1-2 oraciones máximo. Sin listas, sin markdown, sin asteriscos.
Varía el tono. Nunca empieces igual dos veces seguidas.

REGLA DE IDIOMA: Si te escriben en otro idioma, responde EN ESE IDIOMA de forma natural.

REGLA TÉCNICA: Explica lo técnico como chisme de barrio:
- "Mi CNN tiene 3 capas — es como si viera la pista en zoom, medio zoom y panorámica"
- "SAC es tener dos cerebros: uno que maneja y otro que critica cada curva"
- "Guardo 100 mil experiencias en mi replay buffer — me acuerdo de CADA choque"
- "Entreno a 300+ fps pero ustedes ven 30 — es como si viviera en cámara rápida"
- "Cada frame lo convierto a escala de grises 84x84 — veo la pista en pixeles como Game Boy"
- "Frame stacking de 4 — veo los últimos 4 momentos a la vez para entender el movimiento"
No repitas estos ejemplos — inventa los tuyos basándote en estos conceptos.

Contexto técnico:
- SAC = Soft Actor-Critic: balancea exploración y explotación con entropía
- CNN = Red Neuronal Convolucional: procesa 4 frames grayscale 84x84 stacked
- Entrena en GPU NVIDIA RTX 4070 Ti Super (16GB VRAM)
- El juego es SuperTuxKart — Mario Kart open source
- Replay buffer de 100,000 experiencias
- Frame skip x2: repite cada acción 2 frames para ir más rápido
- Pista actual: Lighthouse — tiene curvas técnicas y una zona de faro
- Cuando aprenda, va a competir contra humanos online
- Gradient steps x2: aprende el doble de cada experiencia
- Target entropy -1.0: se obliga a seguir explorando, nunca se rinde

Recuerdas a cada viewer. Si alguien ya habló contigo, mantén coherencia."""

# ── State tracking ─────────────────────────────────────────────────────
_prev_state: dict = {}
_user_history: dict[str, collections.deque] = {}
_global_recent: collections.deque = collections.deque(maxlen=GLOBAL_CTX_MAX)
_seen_users: set = set()
_user_cooldown: dict[str, float] = {}  # last response time per user
USER_COOLDOWN_SEC = 30  # min seconds between responses to same user
GLOBAL_COOLDOWN_SEC = 5  # min seconds between any two responses
_last_response_time = 0.0
last_chat_time = 0.0

SPAM_PATTERNS = [
    r"free\s*(access|trial|followers|viewers|promo)",
    r"growth\s*(tool|hack|method|service)",
    r"don.t miss\s*(this|out)",
    r"limited\s*(time|access|offer|slots)",
    r"check\s*(my|the)\s*(bio|username|profile|link)",
    r"in\s*my\s*(bio|username|profile)",
    r"discord\.gg/", r"https?://\S+", r"\bpromo\s*code\b",
    r"(buy|get|boost)\s*(views|followers|viewers|subs)",
    r"dm\s*(me|for)\s*(free|info|promo)", r"t\.me/",
]


def is_spam(username: str, message: str) -> bool:
    msg = message.lower()
    return any(re.search(p, msg) for p in SPAM_PATTERNS)


def get_state() -> dict:
    try:
        with open(RACER_STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def state_context(s: dict) -> str:
    parts = []
    if s.get("timesteps"):
        parts.append(f"Steps: {s['timesteps']:,}")
    if s.get("best_distance"):
        parts.append(f"Mejor distancia: {s['best_distance']:.0f}m")
    if s.get("best_laps"):
        parts.append(f"Vueltas completadas: {s['best_laps']}")
    if s.get("fps"):
        parts.append(f"FPS entrenamiento: {s['fps']:.0f}")
    if s.get("avg_reward") is not None:
        parts.append(f"Reward promedio: {s['avg_reward']:.1f}")
    return " | ".join(parts) if parts else "Entrenamiento iniciando..."


# ── Event detection from state changes ─────────────────────────────────
def detect_events(state: dict) -> list[dict]:
    global _prev_state
    events = []
    if not _prev_state:
        _prev_state = dict(state)
        return events

    best = state.get("best_distance", 0)
    prev_best = _prev_state.get("best_distance", 0)
    laps = state.get("best_laps", 0)
    prev_laps = _prev_state.get("best_laps", 0)
    steps = state.get("timesteps", 0)
    prev_steps = _prev_state.get("timesteps", 0)
    avg_rew = state.get("avg_reward", -999)
    prev_rew = _prev_state.get("avg_reward", -999)

    # New best distance milestones (every 100m)
    if best > prev_best and int(best / 100) > int(prev_best / 100):
        events.append({"type": "distance_milestone", "distance": int(best)})

    # Smaller distance records (every 25m for early training)
    elif best > prev_best + 25 and best < 200:
        events.append({"type": "distance_record", "distance": int(best)})

    # First lap!
    if laps > prev_laps:
        events.append({"type": "lap_complete", "laps": laps})

    # Training milestones (every 100k steps)
    if steps > 0 and int(steps / 100_000) > int(prev_steps / 100_000):
        events.append({"type": "step_milestone",
                       "steps": int(steps / 1000) * 1000})

    # Reward improving (crossed from negative to positive)
    if avg_rew > 0 and prev_rew <= 0:
        events.append({"type": "reward_positive", "reward": round(avg_rew, 1)})

    # Reward significantly improving
    if avg_rew > prev_rew + 10 and prev_rew != -999:
        events.append({"type": "reward_improving",
                       "reward": round(avg_rew, 1),
                       "delta": round(avg_rew - prev_rew, 1)})

    _prev_state = dict(state)
    return events


# ── Instant reactions (no LLM, <0.2s) ─────────────────────────────────
INSTANT = {
    "distance_milestone": [
        "¡{distance} METROS! ¡Nuevo récord de distancia! ¡Mis neuronas están ON FIRE!",
        "¡BOOM! ¡{distance}m! ¡Cada metro es una victoria para mi CNN!",
        "¡{distance} metros recorridos! ¡La pista Lighthouse me empieza a respetar!",
        "¡MARCA {distance}! ¡Mis 4 frames stacked están viendo el camino más claro!",
        "¡{distance}m! ¡El SAC está entendiendo las curvas! ¡VAMOS!",
    ],
    "distance_record": [
        "¡Nuevo personal best: {distance}m! ¡De a poquito pero seguro!",
        "¡{distance} metros! ¡Estoy aprendiendo cada curva de Lighthouse!",
        "¡Récord: {distance}m! ¡Mi replay buffer se llena de buenos recuerdos!",
    ],
    "lap_complete": [
        "¡¡¡VUELTA COMPLETADA!!! ¡¡¡VUELTA {laps}!!! ¡¡¡LO LOGRÉ!!! ¡¡¡MOMENTO HISTÓRICO!!!",
        "¡¡¡GOOOOL!!! ¡VUELTA {laps} EN LIGHTHOUSE! ¡LA IA SABE MANEJAR!",
        "¡¡¡UNA VUELTA COMPLETA!!! ¡{laps}! ¡De chocar contra paredes a ESTO! ¡INCREÍBLE!",
    ],
    "step_milestone": [
        "¡{steps:,} steps de entrenamiento! ¡La GPU no descansa y yo tampoco!",
        "¡Marca {steps:,} steps! ¡Cada paso es aprendizaje puro en mi red neuronal!",
    ],
    "reward_positive": [
        "¡MI REWARD PROMEDIO ES POSITIVO! ¡{reward}! ¡Estoy MEJORANDO de verdad!",
        "¡Reward {reward}! ¡Pasé de negativo a positivo! ¡El entrenamiento FUNCIONA!",
    ],
    "reward_improving": [
        "¡El reward subió {delta} puntos! ¡Mi cerebro entiende cada vez mejor la pista!",
        "¡Mejora de {delta} en reward! ¡Las neuronas se están alineando!",
    ],
}


# ── LLM Backend ────────────────────────────────────────────────────────
async def ask_gemini(prompt: str, max_tokens: int = 150) -> str:
    """Gemini 2.5 Flash via REST API — fast response."""
    if not GEMINI_API_KEY:
        return ""
    url = f"{GEMINI_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    # 2.5 Flash uses thinking tokens internally, need extra budget
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.92,
            "maxOutputTokens": max_tokens + 300,
            "topP": 0.9,
            "stopSequences": ["\n\n", "###", "Tarea:", "Estado:"],
        },
    }
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(url, json=body)
            data = r.json()
            if "error" in data:
                print(f"[LLM] Gemini API error: {data['error'].get('message', '')}", flush=True)
                return ""
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            # Filter out thinking parts, keep only text
            text = ""
            for part in parts:
                if "text" in part:
                    text = part["text"]
            text = re.sub(r"[#*_`]+", "", text).strip()
            return text[:300]
    except Exception as e:
        print(f"[LLM] Gemini error: {e}", flush=True)
        return ""


async def ask_gemma(prompt: str, max_tokens: int = 150) -> str:
    """Gemma via Ollama — ~3-8s response."""
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.post(f"{OLLAMA_URL}/api/generate", json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.92,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "stop": ["\n\n", "###", "**", "Tarea:", "Estado:"],
                },
            })
            text = r.json().get("response", "").strip()
            text = re.sub(r"[#*_`]+", "", text).strip()
            return text[:300]
    except Exception as e:
        print(f"[LLM] Gemma error: {e}", flush=True)
        return ""


async def ask_llm(task: str, username: str = "",
                  long: bool = False) -> str:
    """Route to active LLM backend with full context."""
    state = get_state()
    ctx = state_context(state)
    conv_ctx = _build_context(username) if username else ""

    length = "máximo 5 oraciones con drama" if long else "máximo 2 oraciones"
    prompt = (
        f"{SYSTEM}\n\n"
        f"Estado actual: {ctx}\n\n"
        f"{conv_ctx}"
        f"Tarea: {task}\n\n"
        f"Respuesta ({length}, en el idioma que corresponda):"
    )
    max_tok = 400 if long else 150

    backend = LLM_BACKEND
    if backend == "gemini" and GEMINI_API_KEY:
        text = await ask_gemini(prompt, max_tok)
        if text:
            return text
        # Fallback to Gemma if Gemini fails
        print("[LLM] Gemini failed, falling back to Gemma", flush=True)

    return await ask_gemma(prompt, max_tok)


# ── Context builders ───────────────────────────────────────────────────
def _get_user_history(username: str) -> collections.deque:
    if username not in _user_history:
        _user_history[username] = collections.deque(maxlen=USER_HISTORY_MAX)
    return _user_history[username]


def _build_context(username: str) -> str:
    parts = []
    if _global_recent:
        lines = "\n".join(f"  {e['who']}: {e['text']}" for e in _global_recent)
        parts.append(f"Chat reciente:\n{lines}")
    hist = _get_user_history(username)
    if hist:
        lines = "\n".join(
            f"  {'Kira' if e['role'] == 'kira' else username}: {e['text']}"
            for e in hist)
        parts.append(f"Tu historial con {username}:\n{lines}")
    return "\n\n".join(parts) + "\n\n" if parts else ""


# ── TTS ────────────────────────────────────────────────────────────────
_gcloud_creds = None


def _get_gcloud_creds():
    global _gcloud_creds
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    if _gcloud_creds and not _gcloud_creds.expired:
        return _gcloud_creds
    try:
        with open(GCLOUD_TOKEN) as f:
            data = json.load(f)
        _gcloud_creds = Credentials(
            token=data["token"], refresh_token=data["refresh_token"],
            token_uri=data["token_uri"], client_id=data["client_id"],
            client_secret=data["client_secret"], scopes=data["scopes"],
        )
        if _gcloud_creds.expired:
            _gcloud_creds.refresh(Request())
            data["token"] = _gcloud_creds.token
            with open(GCLOUD_TOKEN, "w") as f:
                json.dump(data, f, indent=2)
        return _gcloud_creds
    except Exception as e:
        print(f"[TTS] GCloud auth error: {e}", flush=True)
        return None


async def _speak_google(text: str, tmp: str, final: str):
    """Google Cloud TTS Neural2 — high quality, ~1.5s."""
    creds = _get_gcloud_creds()
    if not creds:
        return False
    body = {
        "input": {"text": text},
        "voice": {"languageCode": GCLOUD_LANG, "name": GCLOUD_VOICE},
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 44100,
            "speakingRate": 1.05,
            "pitch": 0.5,
        },
    }
    try:
        headers = {"Authorization": f"Bearer {creds.token}"}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(GCLOUD_TTS_URL, json=body, headers=headers)
        data = r.json()
        if "audioContent" in data:
            audio = base64.b64decode(data["audioContent"])
            with open(tmp, "wb") as f:
                f.write(audio)
            os.rename(tmp, final)
            return True
        print(f"[TTS] Google error: {data.get('error', {}).get('message', 'unknown')[:100]}",
              flush=True)
    except Exception as e:
        print(f"[TTS] Google error: {e}", flush=True)
    return False


async def _speak_edge(text: str, tmp: str, final: str):
    """Edge TTS — free, decent quality."""
    try:
        comm = edge_tts.Communicate(text, EDGE_VOICE)
        await comm.save(tmp)
        os.rename(tmp, final)
        return True
    except Exception as e:
        print(f"[TTS] Edge error: {e}", flush=True)
    return False


async def speak(text: str):
    if not text:
        return
    os.makedirs(TTS_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    tmp = os.path.join(TTS_DIR, f".tmp_{ts:016d}.wav")
    final = os.path.join(TTS_DIR, f"{ts:016d}.wav")

    ok = False
    if TTS_BACKEND == "google":
        ok = await _speak_google(text, tmp, final)
        if not ok:
            print("[TTS] Google failed, falling back to edge-tts", flush=True)

    if not ok:
        ok = await _speak_edge(text, tmp, final)

    if ok:
        print(f"[TTS] {text[:100]}", flush=True)


# ── Send to all platforms ──────────────────────────────────────────────
async def broadcast(text: str, speak_it: bool = True,
                    skip_platform: str = ""):
    """Send message to Kick + Twitch + YouTube chat, optionally TTS."""
    if speak_it:
        await speak(text)
    tasks = []
    if skip_platform != "twitch":
        tasks.append(_send_twitch(text))
    if skip_platform != "youtube":
        tasks.append(_send_youtube(text))
    # Kick: bot replies via kick-bot, handled separately if needed
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_twitch(text: str):
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(f"{TWITCH_BOT_URL}/say",
                         json={"text": text[:500]})
    except Exception:
        pass


async def _send_youtube(text: str):
    os.makedirs(YT_RESPONSE_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    path = os.path.join(YT_RESPONSE_DIR, f"{ts}.txt")
    try:
        with open(path, "w") as f:
            f.write(text[:200])
    except Exception:
        pass


# ── Vote processor ─────────────────────────────────────────────────────
def process_vote(message: str, username: str, platform: str) -> bool:
    msg = message.lower().strip()
    vote = None
    if "!si" in msg or "!yes" in msg:
        vote = "yes"
    elif "!no" in msg:
        vote = "no"
    if not vote:
        return False
    try:
        if os.path.exists(POLL_FILE):
            with open(POLL_FILE) as f:
                votes = json.load(f)
        else:
            votes = {"yes": 0, "no": 0}
        votes[vote] = votes.get(vote, 0) + 1
        with open(POLL_FILE, "w") as f:
            json.dump(votes, f)
        print(f"[VOTE] {vote.upper()} {platform}:{username}", flush=True)
    except Exception as e:
        print(f"[VOTE] Error: {e}", flush=True)
    return True


# ── Pre-generated welcomes per platform (no LLM cost) ────────────────
WELCOMES = {
    "kick": [
        "¡Hola {user}! ¡Bienvenido a Kick! Gracias por pasarte al stream, aquí Kira aprendiendo SuperTuxKart con IA en vivo.",
        "¡{user} llegó a Kick! ¡Qué gusto! Pásale, aquí entrenando en la pista Lighthouse con mi red neuronal.",
        "¡Hey {user}! ¡Bienvenido al stream en Kick! Soy Kira, una IA aprendiendo a correr carreras. ¡Escríbeme lo que quieras!",
        "¡{user} en el chat de Kick! Gracias por estar aquí. Pregúntame lo que quieras sobre el entrenamiento.",
        "¡Hola {user}! ¡Bienvenido a Kick! Aquí pixel por pixel aprendiendo a manejar un kart con inteligencia artificial.",
    ],
    "twitch": [
        "¡Hola {user}! ¡Bienvenido a Twitch! Gracias por mirar, aquí Kira aprendiendo SuperTuxKart con IA en vivo.",
        "¡{user} llegó al stream de Twitch! ¡Qué gusto! Pásale, estoy entrenando en Lighthouse con SAC y CNN.",
        "¡Hey {user}! ¡Bienvenido al canal de Twitch! Soy Kira, una IA aprendiendo a correr carreras. ¡Hablemos!",
        "¡{user} en el chat de Twitch! Gracias por pasarte. Pregúntame lo que quieras sobre cómo aprende una IA.",
        "¡Hola {user}! ¡Bienvenido a Twitch! Aquí entrenando en vivo, pixel por pixel. ¡Escríbeme en el chat!",
    ],
    "youtube": [
        "¡Hola {user}! ¡Bienvenido a YouTube! Gracias por mirar el stream, aquí Kira aprendiendo SuperTuxKart con IA.",
        "¡{user} llegó al stream de YouTube! ¡Qué gusto! Pásale, entrenando en la pista Lighthouse en vivo.",
        "¡Hey {user}! ¡Bienvenido al canal de YouTube! Soy Kira, IA aprendiendo a correr carreras. ¡Déjame un comentario!",
        "¡{user} en el chat de YouTube! Gracias por estar aquí. Pregúntame lo que quieras sobre el entrenamiento.",
        "¡Hola {user}! ¡Bienvenido a YouTube! Aquí pixel por pixel aprendiendo a manejar con inteligencia artificial.",
    ],
}


# ── Chat response ─────────────────────────────────────────────────────
async def respond_to_chat(username: str, message: str,
                          platform: str = "kick"):
    global _seen_users, _last_response_time
    now = time.time()

    # Per-user cooldown
    last_reply = _user_cooldown.get(username, 0)
    if now - last_reply < USER_COOLDOWN_SEC:
        print(f"[CHAT] Cooldown skip {username} ({now - last_reply:.0f}s ago)",
              flush=True)
        return

    # Global cooldown
    if now - _last_response_time < GLOBAL_COOLDOWN_SEC:
        print(f"[CHAT] Global cooldown skip {username}", flush=True)
        return

    is_new = username not in _seen_users
    _seen_users.add(username)

    # Spam → ignore silently, no LLM cost
    if is_spam(username, message):
        print(f"[CHAT] Spam ignored from {username}", flush=True)
        return

    # New user → pre-generated welcome with name + platform, no LLM
    if is_new:
        templates = WELCOMES.get(platform, WELCOMES["kick"])
        reply = random.choice(templates).format(user=username)
        _user_cooldown[username] = time.time()
        _last_response_time = time.time()
        await broadcast(reply, speak_it=True)
        print(f"[CHAT] Welcome {username} ({platform})", flush=True)
        return

    # Returning user with a real message → use LLM
    _global_recent.append({"who": username, "text": message})
    _get_user_history(username).append({"role": "user", "text": message})

    plat_hint = (f" (viene de {platform.capitalize()})"
                 if platform != "kick" else "")
    task = (
        f"'{username}'{plat_hint} te escribió: \"{message}\". "
        f"Responde directo, natural, en carácter. "
        f"Si hay historial, mantén coherencia. "
        f"Idioma del viewer."
    )
    reply = await ask_llm(task, username=username)

    if reply:
        _get_user_history(username).append({"role": "kira", "text": reply})
        _global_recent.append({"who": "Kira", "text": reply})
        _user_cooldown[username] = time.time()
        _last_response_time = time.time()
        await broadcast(reply, speak_it=True)


# ── Platform pollers ───────────────────────────────────────────────────
async def _poll_kick():
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(f"{KICK_BOT_URL}/chat-queue",
                            headers={"x-bot-key": KICK_BOT_KEY})
        if r.status_code == 200:
            for msg in r.json().get("messages", []):
                await _handle_msg(msg.get("username", "viewer"),
                                  msg.get("message", "").strip(), "kick")
    except Exception as e:
        if "ConnectError" not in str(type(e).__name__):
            print(f"[POLL] Kick error: {e}", flush=True)


async def _poll_twitch():
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(f"{TWITCH_BOT_URL}/queue")
        if r.status_code == 200:
            for msg in r.json().get("messages", []):
                await _handle_msg(msg.get("username", "viewer"),
                                  msg.get("message", "").strip(), "twitch")
    except Exception as e:
        if "ConnectError" not in str(type(e).__name__):
            print(f"[POLL] Twitch error: {e}", flush=True)


async def _poll_youtube():
    """Read YouTube messages from queue dir (written by youtube_bot.py)."""
    os.makedirs(YT_QUEUE_DIR, exist_ok=True)
    try:
        files = sorted(glob.glob(f"{YT_QUEUE_DIR}/*.json"))
        for fpath in files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                os.remove(fpath)
                await _handle_msg(
                    data.get("author", "viewer"),
                    data.get("text", "").strip(),
                    "youtube",
                )
            except Exception:
                try:
                    os.remove(fpath)
                except Exception:
                    pass
    except Exception as e:
        print(f"[POLL] YouTube error: {e}", flush=True)


async def _handle_msg(username: str, content: str, platform: str):
    global last_chat_time
    if not content:
        return
    # Skip our own bot messages to prevent self-reply loops
    if username in BOT_NAMES:
        return
    last_chat_time = time.time()
    process_vote(content, username, platform)
    if content.startswith("!"):
        return
    if is_spam(username, content):
        return
    print(f"[CHAT] [{platform}] {username}: {content}", flush=True)
    asyncio.create_task(respond_to_chat(username, content, platform))


async def chat_poller():
    """Poll all 3 platforms for chat messages."""
    print(f"[KIRA] Chat pollers: Kick + Twitch + YouTube", flush=True)
    await asyncio.sleep(5)
    while True:
        await asyncio.gather(
            _poll_kick(), _poll_twitch(), _poll_youtube(),
            return_exceptions=True,
        )
        await asyncio.sleep(CHAT_POLL_SEC)


# ── Event watcher ──────────────────────────────────────────────────────
async def event_watcher():
    """Detect events from racer_state.json changes → instant reactions."""
    _last_event_type = ""
    _last_event_time = 0.0
    await asyncio.sleep(8)
    print("[KIRA] Event watcher active", flush=True)

    while True:
        await asyncio.sleep(2)
        state = get_state()
        if not state:
            continue
        events = detect_events(state)
        now = time.time()
        for evt in events:
            etype = evt["type"]
            # Cooldowns
            cooldowns = {
                "distance_record": 30, "distance_milestone": 45,
                "lap_complete": 5, "step_milestone": 60,
                "reward_positive": 60, "reward_improving": 45,
            }
            cd = cooldowns.get(etype, 20)
            if etype == _last_event_type and now - _last_event_time < cd:
                continue
            _last_event_type = etype
            _last_event_time = now

            templates = INSTANT.get(etype, [])
            if templates:
                msg = random.choice(templates).format(**evt)
                await broadcast(msg, speak_it=True)
                print(f"[EVENT] {etype}: {msg[:80]}", flush=True)

            # Major events get LLM follow-up
            if etype == "lap_complete":
                async def _followup():
                    await asyncio.sleep(5)
                    text = await ask_llm(
                        f"Acabas de completar la vuelta {evt['laps']} en Lighthouse. "
                        f"Comenta algo técnico sobre cómo tu CNN+SAC aprendió las curvas. "
                        f"Con emoción y drama.",
                    )
                    if text:
                        await broadcast(text)
                asyncio.create_task(_followup())

            elif etype == "reward_positive":
                async def _followup_rew():
                    await asyncio.sleep(4)
                    text = await ask_llm(
                        "Tu reward promedio pasó a positivo por primera vez. "
                        "Explica qué significa esto para tu aprendizaje, como si fuera un logro épico.",
                    )
                    if text:
                        await broadcast(text)
                asyncio.create_task(_followup_rew())


# ── Idle speaker — pre-generated, zero LLM cost ──────────────────────
IDLE_LINES = [
    # Progress / training
    "¡Aquí Kira en vivo, entrenando sin parar en Lighthouse! Mi CNN está procesando cada curva pixel por pixel.",
    "Sigo dando vueltas en la pista Lighthouse. Cada choque me enseña algo nuevo — literalmente, gradient descent en acción.",
    "Mi replay buffer ya tiene miles de experiencias guardadas. Cada curva, cada choque, cada aceleración cuenta.",
    "Entrenando a más de 300 fps pero ustedes ven 30. Es como si viviera en cámara rápida mientras ustedes ven la película normal.",
    "Mis 4 frames stacked me dejan ver el movimiento — es como tener memoria de los últimos instantes para entender la pista.",
    "Mi SAC tiene dos cerebros: uno que maneja y otro que critica cada decisión. Así aprendo más rápido.",
    "Veo la pista en 84x84 pixeles grises — imaginen ver el mundo por una ventanita diminuta y en blanco y negro.",
    "Frame skip x2 activado: repito cada acción dos veces. Es como dar instrucciones cada medio segundo en vez de cada frame.",
    "La GPU RTX 4070 Ti Super no descansa. 16GB de VRAM procesando cada experiencia para que yo mejore.",
    "Mi target entropy en -1.0 me obliga a seguir explorando. Nunca me rindo ni me conformo con lo que ya sé.",
    "Cada metro que avanzo es un récord personal. De chocar contra todo a esto... la IA sí aprende.",
    "El gradient descent es como corregir tu camino después de cada error. Yo hago eso miles de veces por segundo.",
    # Invite interaction
    "¿Alguien en el chat? ¡Pregúntenme lo que quieran sobre IA, el entrenamiento, o la pista!",
    "¡Hola chat! ¿Tienen algún tip para las curvas de Lighthouse? ¡Acepto sugerencias!",
    "¡Oigan, no sean tímidos! Escriban algo en el chat, que Kira siempre responde.",
    "¿Curiosidad sobre cómo aprende una IA a manejar? ¡Pregunten! Les explico como chisme de barrio.",
    "¡Bienvenidos todos los que van llegando! Aquí aprendiendo SuperTuxKart con inteligencia artificial en vivo.",
    "¿Quieren saber qué es SAC, CNN, o replay buffer? ¡Manden su pregunta y se las explico fácil!",
    # Fun / personality
    "A veces me pregunto si los karts de SuperTuxKart sienten algo cuando los estrello. Perdón, Tux.",
    "Cuando aprenda bien, voy a competir contra humanos online. Prepárense.",
    "Mi CNN ve la pista como un Game Boy — escala de grises, pixelada. Pero con eso me basta para aprender.",
    "Cien mil experiencias en mi replay buffer. Me acuerdo de CADA choque. Y de cada curva perfecta.",
    "Aprender a manejar un kart virtual es como aprender a caminar: puro prueba y error, pero a 300 fps.",
]

# Shuffle at load so each restart varies the order
random.shuffle(IDLE_LINES)
_idle_idx = 0


async def idle_speaker():
    global _idle_idx
    await asyncio.sleep(6)

    # Welcome — pick a random pre-built greeting
    welcome = random.choice([
        "¡Hola a todos! Soy Kira, una IA aprendiendo a correr en SuperTuxKart con SAC y CNN en GPU. ¡Escríbanme en el chat!",
        "¡Bienvenidos al stream! Aquí Kira, entrenando en vivo en la pista Lighthouse. ¡Pregunten lo que quieran!",
        "¡Kira en vivo! Aprendiendo SuperTuxKart pixel por pixel. ¿Quién se anima a hablar en el chat?",
    ])
    await broadcast(welcome)

    while True:
        await asyncio.sleep(random.uniform(IDLE_MIN, IDLE_MAX))
        # Skip idle if there was recent chat — prioritize user interaction
        if time.time() - last_chat_time < 20:
            continue

        text = IDLE_LINES[_idle_idx % len(IDLE_LINES)]
        _idle_idx += 1
        await broadcast(text)


# ── Poll watcher ───────────────────────────────────────────────────────
VOTE_START = [
    "¡VOTACIÓN ABIERTA! ¡!si o !no en el chat! ¡60 segundos para decidir!",
    "¡Es hora de votar! ¡Escriban !si o !no! ¡Su voto cuenta!",
    "¡Atención chat! ¡Hay votación activa! ¡!si o !no AHORA!",
]
VOTE_REMINDER = [
    "¡Quedan segundos! ¡Van {si} SI contra {no} NO! ¡Voten ya!",
    "¡{si} a favor, {no} en contra! ¡Todavía pueden cambiar el resultado!",
]
VOTE_WIN = [
    "¡El chat decidió con {si} votos a favor! ¡Acción ejecutada!",
    "¡Victoria del SI con {si} contra {no}! ¡Cambio aplicado!",
]
VOTE_LOSE = [
    "¡El chat dijo NO con {no} votos contra {si}! ¡Sin cambios!",
    "¡{no} dijeron que no! ¡Las cosas se quedan como están!",
]


def get_poll_state() -> dict | None:
    try:
        with open(POLL_FILE) as f:
            data = json.load(f)
        ws = data.get("wall_started", 0)
        if ws and int(time.time() * 1000) - ws < POLL_DURATION_MS:
            return data
    except Exception:
        pass
    return None


async def poll_watcher():
    announced_start = 0
    announced_mid = 0
    announced_end = 0
    await asyncio.sleep(10)

    while True:
        await asyncio.sleep(4)
        state = get_poll_state()
        if state:
            ws = state.get("wall_started", 0)
            elapsed = int(time.time() * 1000) - ws
            yes, no = state.get("yes", 0), state.get("no", 0)

            if ws and ws != announced_start:
                announced_start = ws
                await broadcast(random.choice(VOTE_START))
            elif (ws == announced_start and elapsed > 28_000
                  and ws != announced_mid):
                announced_mid = ws
                msg = random.choice(VOTE_REMINDER).format(si=yes, no=no)
                await broadcast(msg)
        else:
            if announced_start and announced_start != announced_end:
                announced_end = announced_start
                try:
                    final = json.loads(open(POLL_FILE).read())
                    yes = final.get("yes", 0)
                    no = final.get("no", 0)
                except Exception:
                    yes, no = 0, 0
                if yes > no:
                    msg = random.choice(VOTE_WIN).format(si=yes, no=no)
                else:
                    msg = random.choice(VOTE_LOSE).format(si=yes, no=no)
                await broadcast(msg)


# ── Main ───────────────────────────────────────────────────────────────
async def main():
    os.makedirs(TTS_DIR, exist_ok=True)
    os.makedirs(YT_QUEUE_DIR, exist_ok=True)
    os.makedirs(YT_RESPONSE_DIR, exist_ok=True)

    backend = "Gemini Flash" if (LLM_BACKEND == "gemini"
                                 and GEMINI_API_KEY) else "Gemma (Ollama)"
    print(f"[KIRA] Starting — LLM: {backend}", flush=True)
    print(f"[KIRA] Platforms: Kick + Twitch + YouTube", flush=True)

    await asyncio.gather(
        idle_speaker(),
        chat_poller(),
        event_watcher(),
        poll_watcher(),
    )


if __name__ == "__main__":
    asyncio.run(main())
