import os
import requests

token = os.environ.get("TELEGRAM_BOT_TOKEN")
if not token:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN first")

url = f"https://api.telegram.org/bot{token}/getUpdates"
r = requests.get(url, timeout=30)
print(r.json())