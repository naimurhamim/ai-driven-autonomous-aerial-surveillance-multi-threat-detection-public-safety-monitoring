import requests

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}/sendPhoto"

    def send_photo(self, image_path: str, caption: str = ""):
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": self.chat_id, "caption": caption}
            r = requests.post(self.api_url, data=data, files=files, timeout=30)
            r.raise_for_status()