from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import requests
from cryptography.fernet import Fernet

LOGGER = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    attempts: int
    backoff_seconds: int


class SecretStore:
    def __init__(self, key_env: str = "NSBOT_SECRET_KEY") -> None:
        key = os.getenv(key_env)
        if not key:
            key = Fernet.generate_key().decode()
            os.environ[key_env] = key
        self.cipher = Fernet(key.encode())

    def encrypt(self, secret: str) -> str:
        return self.cipher.encrypt(secret.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self.cipher.decrypt(token.encode()).decode()


class BinanceExecutor:
    def __init__(self, api_key: str, retry: RetryConfig) -> None:
        self.api_key = api_key
        self.retry = retry

    def place_order(self, base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"X-MBX-APIKEY": self.api_key}
        for attempt in range(1, self.retry.attempts + 1):
            try:
                r = requests.post(base_url, headers=headers, data=payload, timeout=15)
                r.raise_for_status()
                result = r.json()
                if "orderId" not in result:
                    raise ValueError(f"Order not confirmed: {result}")
                LOGGER.info("Order placed: %s", result.get("orderId"))
                return result
            except Exception as exc:
                LOGGER.warning("Order attempt %s failed: %s", attempt, exc)
                if attempt == self.retry.attempts:
                    raise
                time.sleep(self.retry.backoff_seconds * attempt)
        return {}
