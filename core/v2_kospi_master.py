"""KOSPI 종목마스터(kospi_code.mst) 다운로드·파싱 — ETF 전체 목록 확보용.

KIS 공개 정적 파일 다운로드라(API 호출이 아님) 토큰·호출 제한과 무관하다(kis_auth 불필요).
필드 레이아웃은 KIS 공식 저장소 `koreainvestment/open-trading-api`의
`stocks_info/kis_kospi_code_mst.py`·`종목마스터정보(코스피).h` 기준(2026-09 확인).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from v2_config import state_dir

log = logging.getLogger(__name__)

MST_URL = "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip"

# 앞부분(단축코드 9·표준코드 12·한글명 가변)을 뺀 나머지 228바이트 고정폭 필드.
_PART2_WIDTHS = [2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 1, 3, 12, 12, 8,
                 15, 21, 2, 7, 1, 1, 1, 1, 1, 9, 9, 9, 5, 9, 8, 9, 3, 1, 1, 1]
_PART2_COLUMNS = [
    "그룹코드", "시가총액규모", "지수업종대분류", "지수업종중분류", "지수업종소분류",
    "제조업", "저유동성", "지배구조지수종목", "KOSPI200섹터업종", "KOSPI100",
    "KOSPI50", "KRX", "ETP", "ELW발행", "KRX100",
    "KRX자동차", "KRX반도체", "KRX바이오", "KRX은행", "SPAC",
    "KRX에너지화학", "KRX철강", "단기과열", "KRX미디어통신", "KRX건설",
    "Non1", "KRX증권", "KRX선박", "KRX섹터_보험", "KRX섹터_운송",
    "SRI", "기준가", "매매수량단위", "시간외수량단위", "거래정지",
    "정리매매", "관리종목", "시장경고", "경고예고", "불성실공시",
    "우회상장", "락구분", "액면변경", "증자구분", "증거금비율",
    "신용가능", "신용기간", "전일거래량", "액면가", "상장일자",
    "상장주수", "자본금", "결산월", "공모가", "우선주",
    "공매도과열", "이상급등", "KRX300", "KOSPI", "매출액",
    "영업이익", "경상이익", "당기순이익", "ROE", "기준년월",
    "시가총액", "그룹사코드", "회사신용한도초과", "담보대출가능", "대주가능",
]


def _cache_path() -> Path:
    return state_dir() / "kospi_code.mst"


def download_master(force: bool = False) -> Path:
    """정적 zip을 받아 .mst만 캐시에 풀어둔다. 이미 있으면(force=False) 재다운로드 생략."""
    path = _cache_path()
    if path.exists() and not force:
        return path
    resp = requests.get(MST_URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".mst"))
        path.write_bytes(zf.read(name))
    log.info("KOSPI 종목마스터 다운로드: %s (%d bytes)", path, path.stat().st_size)
    return path


def load_master(force: bool = False) -> pd.DataFrame:
    """전체 종목마스터 — {단축코드, 표준코드, 한글명, 그룹코드, ...} (cp949 고정폭 파일)."""
    path = download_master(force=force)
    rows1, rows2 = [], []
    with path.open(mode="r", encoding="cp949") as f:
        for row in f:
            head = row[: len(row) - 228]
            rows1.append([head[0:9].rstrip(), head[9:21].rstrip(), head[21:].strip()])
            rows2.append(row[-228:])
    df1 = pd.DataFrame(rows1, columns=["단축코드", "표준코드", "한글명"])
    df2 = pd.read_fwf(io.StringIO("".join(rows2)), widths=_PART2_WIDTHS, names=_PART2_COLUMNS)
    return pd.concat([df1, df2], axis=1)


def load_etf_universe(force: bool = False) -> pd.DataFrame:
    """그룹코드=='EF' 인 ETF만 {code, name} 두 컬럼으로."""
    df = load_master(force=force)
    etf = df.loc[df["그룹코드"] == "EF", ["단축코드", "한글명"]].rename(
        columns={"단축코드": "code", "한글명": "name"})
    return etf.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    u = load_etf_universe(force="--force" in sys.argv)
    print(f"ETF {len(u)}개")
    print(u.head(10).to_string(index=False))
