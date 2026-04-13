"""
yt_auth.py — One-time OAuth2 flow for YouTube Data API.
Run on Mac, copy token to Fedora.

Usage: python3 bots/yt_auth.py
"""
import json, os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
CLIENT_SECRETS = "/Users/efra/claudecode/gcloud-efrain.garay.a.json"
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "yt_token.json")


def main():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
    creds = flow.run_local_server(port=8085, prompt="consent",
                                  access_type="offline")
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes),
    }
    with open(TOKEN_PATH, "w") as f:
        json.dump(token_data, f, indent=2)
    print(f"[AUTH] Token saved to {TOKEN_PATH}")
    print(f"[AUTH] Copy to Fedora: scp {TOKEN_PATH} clawadmin@100.109.82.18:/home/clawadmin/neat-racer/bots/")


if __name__ == "__main__":
    main()
