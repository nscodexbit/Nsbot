from __future__ import annotations

import requests


def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> None:
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
