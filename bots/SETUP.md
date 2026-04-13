# Bots Setup — NEAT Racer

## Architecture

```
Viewers ──→ Kick / Twitch / YouTube chat
                    │
            ┌───────┴────────┐
            │  youtube_bot   │  ← YouTube Data API v3 (OAuth2)
            │  (polls chat)  │
            └───────┬────────┘
                    │ /tmp/kira_yt_queue/*.json
            ┌───────┴────────┐
            │     Kira       │  ← LLM: Gemini 2.5 Flash / Gemma 4
            │  (AI host)     │     TTS: Google Cloud Neural2 / edge-tts
            └───────┬────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   /tmp/tts_queue  Twitch     YouTube
   (audio files)   (HTTP)     (/tmp/kira_yt_responses)
```

## Services (systemd on Fedora 100.109.82.18)

| Service | Unit | Description |
|---|---|---|
| Kira AI Host | `kira-racer` | Chat AI, TTS, event reactions |
| YouTube Bot | `youtube-bot` | YouTube live chat polling + sending |

## API Keys & Credentials

All credentials stored in `bots/.env` (gitignored).
Token files (`yt_token.json`, `gcloud_token.json`) also gitignored.

| Key | Service | Free Tier | How to get |
|---|---|---|---|
| `GEMINI_API_KEY` | Gemini 2.5 Flash | 15 RPM / 1M tokens/day | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GCLOUD_*` tokens | Cloud TTS Neural2 | 1M chars/month | OAuth via `yt_auth.py` (scope: cloud-platform) |
| `YT_*` tokens | YouTube Data API v3 | 10,000 units/day | OAuth via `yt_auth.py` (scope: youtube.force-ssl) |
| `KICK_BOT_KEY` | Kick chat relay | Unlimited | Custom bot on phantom-dash.com |
| `TWITCH_BOT_URL` | Twitch chat relay | Unlimited | Local bot on port 9997 |

## Cost Controls

### Gemini 2.5 Flash (free tier)
- 15 requests/minute, 1,500 requests/day
- 1 million tokens/day input, 1 million output
- Kira only calls LLM for: user chat replies (with 30s/user cooldown)
- Idle commentary and welcomes are pre-generated (zero LLM cost)
- Set quota in [Google AI Studio Console](https://aistudio.google.com/) > API Keys > Edit

### Google Cloud TTS Neural2
- Free tier: 1 million characters/month (Neural2)
- Each Kira message ~50-100 chars → ~10,000-20,000 messages/month free
- Set budget alert: [GCP Billing](https://console.cloud.google.com/billing) > Budgets & Alerts
- Set TTS quota: [GCP Quotas](https://console.cloud.google.com/apis/api/texttospeech.googleapis.com/quotas)
- Fallback: set `TTS_BACKEND=edge` in service to use free edge-tts

### YouTube Data API v3
- 10,000 quota units/day
- liveChatMessages.list = 5 units per call
- liveChatMessages.insert = 50 units per call
- At 5s polling: ~17,280 list calls/day = 86,400 units → EXCEEDS free tier
- Current polling: API-provided interval (~5-8s) = safe within limits
- Set quota: [GCP API Dashboard](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas)

### How to set budget alerts (Google Cloud)
1. Go to https://console.cloud.google.com/billing
2. Select project → Budgets & Alerts → Create Budget
3. Set amount (e.g., $5/month)
4. Alert at 50%, 90%, 100%
5. Optionally enable "Cap spending" to auto-disable billing

## OAuth Token Refresh

Tokens auto-refresh using refresh_token. If a token expires permanently:

```bash
# YouTube token (run on Mac)
python3 bots/yt_auth.py
scp bots/yt_token.json clawadmin@100.109.82.18:/home/clawadmin/neat-racer/bots/

# Google Cloud token — same flow but with cloud-platform scope
# Edit yt_auth.py to use scope: https://www.googleapis.com/auth/cloud-platform
# Run, copy token, rename to gcloud_token.json
```

## Deploy to Fedora

```bash
# From Mac (dotfiles/projects/neat-racer/)
scp bots/kira.py clawadmin@100.109.82.18:/tmp/kira_update.py
scp bots/youtube_bot.py clawadmin@100.109.82.18:/tmp/yt_bot_update.py

# On Fedora
sudo cp /tmp/kira_update.py /home/clawadmin/neat-racer/bots/kira.py
sudo cp /tmp/yt_bot_update.py /home/clawadmin/neat-racer/bots/youtube_bot.py
sudo systemctl restart kira-racer youtube-bot
```

## Bot Behavior Summary

| Trigger | Action | LLM Cost | TTS Cost |
|---|---|---|---|
| New viewer arrives | Pre-generated welcome | None | 1 TTS call |
| Viewer writes in chat | LLM response (30s cooldown/user) | 1 Gemini call | 1 TTS call |
| No chat activity (60-120s) | Pre-generated idle line | None | 1 TTS call |
| Game event (record, lap) | Pre-generated template | None | 1 TTS call |
| Major event (lap complete) | Template + LLM follow-up | 1 Gemini call | 2 TTS calls |
| Spam detected | Silently ignored | None | None |
| Bot's own messages | Filtered (BOT_NAMES) | None | None |

## Self-Reply Prevention

Bot filters its own messages to prevent infinite loops:
- `BOT_NAMES` in both `kira.py` and `youtube_bot.py`
- Includes: `desafioIA7`, `@desafioIA7`, `KiraBot`, `Kira`
