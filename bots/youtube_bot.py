#!/usr/bin/env python3
"""
youtube_bot.py — YouTube Live Chat bot for NEAT Racer stream.

Reads chat messages from YouTube Live, feeds them to Kira pipeline,
and can send responses back to YouTube chat.

Requires: google-api-python-client, google-auth, google-auth-oauthlib
"""
import os, sys, time, json, threading, glob
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ── Config ─────────────────────────────────────────────────────────────
VIDEO_ID = os.environ.get("YT_VIDEO_ID", "S5rarKLBuzY")
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "yt_token.json")
CHAT_QUEUE_DIR = "/tmp/yt_chat_queue"
TTS_DIR = "/tmp/tts_queue"
KIRA_QUEUE = "/tmp/kira_yt_queue"
POLL_VOTES = "/tmp/poll_votes.json"

# ── Auth ───────────────────────────────────────────────────────────────
def get_credentials():
    with open(TOKEN_PATH) as f:
        data = json.load(f)
    creds = Credentials(
        token=data["token"],
        refresh_token=data["refresh_token"],
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data["scopes"],
    )
    if creds.expired:
        creds.refresh(Request())
        data["token"] = creds.token
        with open(TOKEN_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print("[YT] Token refreshed", flush=True)
    return creds


def get_youtube():
    creds = get_credentials()
    return build("youtube", "v3", credentials=creds)


# ── Get live chat ID ───────────────────────────────────────────────────
def get_live_chat_id(youtube, video_id):
    """Get liveChatId from a live broadcast video."""
    resp = youtube.videos().list(
        part="liveStreamingDetails",
        id=video_id,
    ).execute()
    items = resp.get("items", [])
    if not items:
        print(f"[YT] Video {video_id} not found", flush=True)
        return None
    details = items[0].get("liveStreamingDetails", {})
    chat_id = details.get("activeLiveChatId")
    if not chat_id:
        print(f"[YT] No active live chat for {video_id}", flush=True)
        return None
    print(f"[YT] Live chat ID: {chat_id}", flush=True)
    return chat_id


# ── Send message to YouTube chat ──────────────────────────────────────
def send_message(youtube, chat_id, text):
    """Send a message to YouTube live chat."""
    try:
        youtube.liveChatMessages().insert(
            part="snippet",
            body={
                "snippet": {
                    "liveChatId": chat_id,
                    "type": "textMessageEvent",
                    "textMessageDetails": {"messageText": text},
                }
            },
        ).execute()
        print(f"[YT] Sent: {text[:80]}", flush=True)
    except Exception as e:
        print(f"[YT] Send error: {e}", flush=True)


# ── Process incoming message ───────────────────────────────────────────
BOT_NAMES = {"desafioIA7", "desafioia7", "@desafioIA7", "@desafioia7",
             "KiraBot", "kirabot"}


def process_message(msg, youtube, chat_id):
    """Process a YouTube chat message — votes, commands, forward to Kira."""
    author = msg["authorDetails"]["displayName"]
    text = msg["snippet"]["textMessageDetails"]["messageText"].strip()
    ts = msg["snippet"]["publishedAt"]

    # Skip our own bot messages to prevent self-reply loops
    if author in BOT_NAMES:
        return

    print(f"[YT] {author}: {text}", flush=True)

    # Vote commands
    lower = text.lower()
    if lower in ("!si", "!yes", "!no"):
        try:
            if os.path.exists(POLL_VOTES):
                with open(POLL_VOTES) as f:
                    votes = json.load(f)
                if "voters" not in votes:
                    votes["voters"] = {}
                voter_key = f"yt_{author}"
                if voter_key not in votes["voters"]:
                    if lower in ("!si", "!yes"):
                        votes["yes"] = votes.get("yes", 0) + 1
                    else:
                        votes["no"] = votes.get("no", 0) + 1
                    votes["voters"][voter_key] = lower
                    with open(POLL_VOTES, "w") as f:
                        json.dump(votes, f)
        except Exception as e:
            print(f"[YT] Vote error: {e}", flush=True)
        return

    # Skip bot commands
    if text.startswith("!"):
        return

    # Forward to Kira pipeline
    os.makedirs(KIRA_QUEUE, exist_ok=True)
    msg_file = os.path.join(KIRA_QUEUE, f"{int(time.time()*1000)}.json")
    try:
        with open(msg_file, "w") as f:
            json.dump({
                "platform": "youtube",
                "author": author,
                "text": text,
                "ts": ts,
            }, f)
    except Exception as e:
        print(f"[YT] Queue error: {e}", flush=True)


# ── Response sender thread ─────────────────────────────────────────────
def response_sender(youtube, chat_id):
    """Watch for Kira responses to send to YouTube chat."""
    response_dir = "/tmp/kira_yt_responses"
    os.makedirs(response_dir, exist_ok=True)
    seen = set()
    while True:
        try:
            files = sorted(glob.glob(f"{response_dir}/*.txt"))
            for f in files:
                if f not in seen:
                    seen.add(f)
                    try:
                        with open(f) as fh:
                            text = fh.read().strip()
                        if text:
                            send_message(youtube, chat_id, text)
                        os.remove(f)
                        seen.discard(f)
                    except Exception as e:
                        print(f"[YT] Response send error: {e}", flush=True)
        except Exception:
            pass
        time.sleep(1)


# ── Main chat polling loop ─────────────────────────────────────────────
def main():
    print(f"[YT] YouTube Bot starting — video={VIDEO_ID}", flush=True)

    if not os.path.exists(TOKEN_PATH):
        print(f"[YT] ERROR: No token at {TOKEN_PATH}. Run yt_auth.py first.",
              flush=True)
        sys.exit(1)

    youtube = get_youtube()
    chat_id = get_live_chat_id(youtube, VIDEO_ID)
    if not chat_id:
        print("[YT] Waiting for live stream to become active...", flush=True)
        while not chat_id:
            time.sleep(30)
            try:
                youtube = get_youtube()
                chat_id = get_live_chat_id(youtube, VIDEO_ID)
            except Exception as e:
                print(f"[YT] Retry error: {e}", flush=True)
                time.sleep(30)

    print(f"[YT] Connected to live chat", flush=True)

    # Start response sender thread
    t = threading.Thread(target=response_sender, args=(youtube, chat_id),
                         daemon=True)
    t.start()

    # Poll chat messages
    page_token = None
    poll_interval = 5  # seconds, updated from API response
    first_poll = True  # skip historical messages on startup

    while True:
        try:
            params = {
                "liveChatId": chat_id,
                "part": "snippet,authorDetails",
                "maxResults": 200,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = youtube.liveChatMessages().list(**params).execute()

            page_token = resp.get("nextPageToken")
            poll_interval = resp.get("pollingIntervalMillis", 5000) / 1000.0

            if first_poll:
                first_poll = False
                count = len(resp.get("items", []))
                print(f"[YT] Skipped {count} historical messages", flush=True)
            else:
                for item in resp.get("items", []):
                    try:
                        process_message(item, youtube, chat_id)
                    except Exception as e:
                        print(f"[YT] Process error: {e}", flush=True)

        except Exception as e:
            err = str(e)
            if "liveChatEnded" in err or "403" in err:
                print(f"[YT] Chat ended or forbidden: {e}", flush=True)
                print("[YT] Waiting for new stream...", flush=True)
                chat_id = None
                while not chat_id:
                    time.sleep(60)
                    try:
                        youtube = get_youtube()
                        chat_id = get_live_chat_id(youtube, VIDEO_ID)
                    except Exception:
                        pass
                page_token = None
                continue
            print(f"[YT] Poll error: {e}", flush=True)
            time.sleep(10)
            continue

        time.sleep(max(poll_interval, 2))


if __name__ == "__main__":
    main()
