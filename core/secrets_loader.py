"""비밀값 로더: 환경변수 우선, 없으면 .streamlit/secrets.toml."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import os
import tomllib
from pathlib import Path

SECRETS_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"  # core/ 의 부모 = 프로젝트 루트

_PLACEHOLDER_MARK = "여기에"


def _load_file() -> dict:
    if not SECRETS_PATH.exists():
        return {}
    with SECRETS_PATH.open("rb") as f:
        return tomllib.load(f)


def get_secret(key: str, required: bool = True) -> str | None:
    val = os.environ.get(key) or _load_file().get(key)
    if val and _PLACEHOLDER_MARK in str(val):  # 템플릿 값 그대로면 미설정 취급
        val = None
    if not val and required:
        raise KeyError(f"비밀값 '{key}' 없음 — 환경변수 또는 {SECRETS_PATH} 에 설정하세요.")
    return val
