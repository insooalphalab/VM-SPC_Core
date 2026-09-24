"""한투 Open API 접근토큰 발급 + 파일 캐싱. (V1 kis_auth.py 와 동일 — 독립 구성을 위해 복사)

접근토큰 발급은 5분당 1회로 제한된다(2023.10.27~). 호출마다 재발급하면 차단되므로
발급된 토큰을 kis_token_cache.json 에 저장해 유효기간 동안 재사용한다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import hashlib
import json
import logging
import time
from datetime import datetime

import requests

from v2_config import KST, state_dir
from secrets_loader import get_secret

log = logging.getLogger(__name__)

BASE_URL = "https://openapi.koreainvestment.com:9443"  # 실전
TOKEN_MIN_INTERVAL_SEC = 300      # 발급 최소 간격
EXPIRY_MARGIN_SEC = 600           # 만료 10분 전이면 새 토큰으로 간주
MAX_WAIT_FOR_REISSUE_SEC = 330    # 발급 제한 해제까지 기다려줄 최대 시간


def _cache_path():
    return state_dir() / "kis_token_cache.json"


def _key_fingerprint(app_key: str) -> str:
    return hashlib.sha256(app_key.encode()).hexdigest()[:16]  # 키 원문은 캐시에 남기지 않는다


def _read_cache() -> dict:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_cache(obj: dict) -> None:
    _cache_path().write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _cached_token_valid(cache: dict, fingerprint: str, now: float) -> bool:
    return (
        cache.get("fingerprint") == fingerprint
        and bool(cache.get("access_token"))
        and now < cache.get("expires_at", 0) - EXPIRY_MARGIN_SEC
    )


def _issue_token(app_key: str, app_secret: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/oauth2/tokenP",
        headers={"content-type": "application/json"},
        data=json.dumps({"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}),
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if "access_token" not in body:
        raise RuntimeError(f"토큰 발급 실패: {body}")
    now = time.time()
    if body.get("expires_in"):
        expires_at = now + float(body["expires_in"])
    else:  # "YYYY-MM-DD HH:MM:SS" (KST)
        expires_at = datetime.strptime(body["access_token_token_expired"], "%Y-%m-%d %H:%M:%S") \
            .replace(tzinfo=KST).timestamp()
    return {"access_token": body["access_token"], "expires_at": expires_at, "issued_at": now}


def get_token(force_refresh: bool = False) -> str:
    """유효한 캐시 토큰을 반환하고, 없을 때만 (발급 제한을 지켜서) 새로 발급한다."""
    app_key, app_secret = get_secret("KIS_APP_KEY"), get_secret("KIS_APP_SECRET")
    fp = _key_fingerprint(app_key)
    cache = _read_cache()
    now = time.time()

    if not force_refresh and _cached_token_valid(cache, fp, now):
        return cache["access_token"]

    # 발급 제한(5분당 1회): 직전 발급 후 5분이 안 지났으면 기다린다. 너무 길면 실패시킨다.
    if cache.get("fingerprint") == fp and cache.get("issued_at"):
        wait = cache["issued_at"] + TOKEN_MIN_INTERVAL_SEC - now
        if wait > 0:
            if wait > MAX_WAIT_FOR_REISSUE_SEC:
                raise RuntimeError(f"토큰 재발급 제한: {wait:.0f}초 후 재시도 필요")
            log.warning("토큰 발급 제한(5분/1회) — %.0f초 대기 후 재발급", wait)
            time.sleep(wait + 1)

    token = _issue_token(app_key, app_secret)
    _write_cache({**token, "fingerprint": fp})
    log.info("새 접근토큰 발급 (만료: %s)", datetime.fromtimestamp(token["expires_at"], KST))
    return token["access_token"]


def auth_headers(tr_id: str, force_refresh: bool = False) -> dict:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {get_token(force_refresh)}",
        "appkey": get_secret("KIS_APP_KEY"),
        "appsecret": get_secret("KIS_APP_SECRET"),
        "tr_id": tr_id,
        "custtype": "P",
    }
