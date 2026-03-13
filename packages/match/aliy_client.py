import os
import requests
import time
from pathlib import Path

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def _read_env_file(path):
    try:
        txt = Path(path).read_text(encoding="utf-8")
    except:
        return {}
    env = {}
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            env[k.strip()] = v.strip()
    return env

class AliyClient:
    def __init__(self, api_key=None, base_url=None):
        key = api_key or os.environ.get("ALI_LLM_API_KEY", "")
        if not key:
            here_env = _read_env_file(os.path.join(os.path.dirname(__file__), ".env"))
            key = here_env.get("ALI_LLM_API_KEY", key)
        if not key:
            up_env = _read_env_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
            key = up_env.get("ALI_LLM_API_KEY", key)
        self.api_key = key
        self.base_url = base_url or BASE_URL
        if not self.api_key:
            raise RuntimeError("ALI_LLM_API_KEY not set")
    def chat_completions(self, messages, model):
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages}
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=600)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
            except requests.exceptions.HTTPError as e:
                last_err = e
                if 500 <= r.status_code < 600:
                    time.sleep(2 * (attempt + 1))
                else:
                    break
        raise last_err
