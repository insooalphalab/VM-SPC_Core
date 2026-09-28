# VM-SPC Core (KIS_TA V2)

V1(5대시그널 스코어카드)의 칼만필터·신호 계산을 재사용해 반도체 FDC/VM 엔지니어링 패턴
(**가상계측(Virtual Metrology)** + **SPC 드리프트 탐지(CUSUM + Hotelling's T²)**)으로 섹터 ETF와
구성종목의 **익일 방향과 그 확률**을 매일 장 마감 후 산출하는 도구입니다. 39개 섹터 ETF를 평일
16:30에 자동으로 갱신하고, 결과를 텔레그램과 구글 드라이브로 받아 봅니다. 자동 주문은 없으며
진입 여부는 사람이 판단합니다. 설계 배경은 `kis_ta_v2_design.md` 참고.

```
센서(top N 종목) VM Score(4개 신호, 수급 제외) 균등가중 평균
  → Index Breadth Score → 익일 타겟 ETF 등락 예측 → 컨퓨전 매트릭스(TP/FP/FN/TN)

CUSUM(양방향)  : 타겟 ETF 자체 칼만 잔차 + breadth 시계열, 둘 다에 독립 적용
Hotelling's T² : 바스켓 평균 4개 신호 간 상관관계 붕괴(복합 이상) 탐지
```

```
[Stage 1: v2_data_collector.py] --(data/*.csv)--> [Stage 2: v2_compute_engine.py]
  --(results/*.json)--> [Stage 3: v2_render_dashboard.py] --> dashboard_v2.html
```

## 폴더 구조

파일이 늘어나면서(2026-09-24) 역할별로 하위 폴더로 정리했습니다. 모든 모듈이 자기 파일 위치
기준으로 프로젝트 루트를 `sys.path`에 추가하는 부트스트랩을 갖고 있어서, 아래 폴더 안 스크립트를
프로젝트 루트에서 그대로 `python <폴더>/<파일>.py`로 실행하면 됩니다(별도 `PYTHONPATH` 설정 불필요).

```
core/         — 공용 인프라: KIS 인증·API 클라이언트, 경로/Params 설정, 칼만·CUSUM·T²·신호 계산,
                ETF 스캐너. 두 파이프라인(vm_predict, pair_spc) 모두 이 폴더에 의존합니다.
vm_predict/   — 예측모델(VM-SPC) 파이프라인: Stage1 수집 → Stage2 연산 → Stage3 대시보드 렌더링
vm_spc/       — 챔피언-챌린저(VM-SPC Core) 의사결정 보조 모델: Layer2 피처 → Walk-Forward → 3-분기 게이트.
                dashboard_v2.html 에 "챔피언-챌린저" 탭으로 결과를 얹습니다.
pair_spc/     — 대표종목 상관관계(PAIR-SPC) 파이프라인(신규): Stage0 대표종목 실증 → Stage1/2
                공적분·SPC. vm_predict의 dashboard_v2.html에 탭으로 결과를 얹습니다.
scripts/      — 매일 자동 실행용 배치(작업 스케줄러가 호출)
data/ results/ state/ logs/  — 자동 생성물(.gitignore 대상)
```

## 빠른 시작

```powershell
pip install -r requirements.txt                                    # Python 3.11+
copy .streamlit\secrets.example.toml .streamlit\secrets.toml       # KIS 앱키/시크릿 입력
copy basket_watchlist.example.json basket_watchlist.json           # 실제 종목·타겟 ETF로 수정
python core/kis_client.py verify                                   # 종목코드↔종목명 검증
python vm_predict/v2_run.py                                        # 예측모델: 수집→연산→렌더링 한 번에
python pair_spc/run_pair_spc.py                                    # PAIR-SPC: 대표종목 실증→공적분·SPC (선택)
python vm_spc/pipeline.py                                          # 챔피언-챌린저: 3-Way 벤치마크→게이트 판정 (선택)
```

`results/{바스켓이름}/dashboard_v2.html`을 더블클릭해 브라우저로 엽니다(서버 불필요) — 상단
"예측모델 (VM-SPC)" / "대표종목 상관관계 (PAIR-SPC)" 탭으로 전환합니다(PAIR-SPC를 아직 안 돌렸으면
탭 자체가 안 보이고 예측모델 화면만 나옵니다 — 하위 호환).
`basket_watchlist.json`에 타겟 ETF가 지정된 섹터가 여러 개면 `v2_run.py`가 전부 순서대로 처리합니다.
특정 하나만: `python vm_predict/v2_run.py --basket semiconductor_to_etf`. 이미 수집한 데이터로
연산·화면만 다시: `python vm_predict/v2_run.py --no-fetch`(API 호출 안 함, `data/*.csv` 재사용).

일봉 수집은 **증분 방식**입니다 — 처음 보는 종목만 약 3년치(1095일)를 백필하고, 이후에는 저장된
마지막 날짜 기준 최근 15일만 다시 받아 병합합니다(종목당 API 1회). 겹치는 날짜의 종가가 기존 값과
0.5% 넘게 다르면 액면분할·무상증자 등으로 수정주가가 소급 조정된 것으로 보고 그 종목만 전체를
다시 받습니다(조정 전/후 가격이 섞여 시계열이 끊기는 것을 방지).

### 매일 자동 실행 (작업 스케줄러 · 텔레그램 · 구글 드라이브)

Windows 작업 스케줄러 "VM-SPC Core 일일 업데이트"가 **평일 16:30**에
`scripts/run_daily_pipeline_hidden.vbs` → `scripts/run_daily_pipeline.ps1`을 실행합니다(PC가 켜져
있을 때만, 꺼져 있었으면 켜질 때 따라잡음). 순서:

1. `vm_predict/v2_run.py` — 일봉 증분 수집 → 연산 → 대시보드
2. `pair_spc/run_pair_spc.py` — PAIR-SPC (대표종목 후보도 여기서 최신화)
3. `vm_spc/pipeline.py` — 챔피언-챌린저 → `results/index.html` 재생성
4. `results/` → `G:\내 드라이브\VM-SPC_Core\results` 미러링(robocopy `/MIR`, Drive for Desktop이 업로드)
5. `vm_spc/notify_telegram.py` — 요약 + `index.html` 첨부 전송. 마지막으로 알린 거래일과 최신
   거래일이 같으면(평일 휴장일) 전송을 생략하고, 앞 단계 실패 시에는 경고를 붙여 항상 전송합니다.

로그는 `logs/pipeline_YYYYMMDD_HHMMSS.log`(30일 보관). 주의:

- 창을 확실히 숨기려고 스케줄러는 PowerShell을 직접 부르지 않고 VBScript(`WScript.Shell.Run(cmd, 0)`)로
  띄웁니다 — `powershell -WindowStyle Hidden`은 스케줄러 경유 시 인자가 누락돼 콘솔 창이 보이고,
  그 창을 클릭하면 빠른 편집 모드로 프로세스가 멈추는 문제가 실제로 있었습니다.
- `run_daily_pipeline.ps1`은 **UTF-8 BOM 포함**으로 저장해야 합니다. BOM이 없으면 Windows
  PowerShell 5.1이 CP949로 읽어 한글 경로(`G:\내 드라이브`)가 깨집니다.
- 텔레그램은 `.streamlit/secrets.toml`의 `TELEGRAM_BOT_TOKEN`·`TELEGRAM_CHAT_ID`를 씁니다.
  내용만 확인: `python vm_spc/notify_telegram.py --dry-run`, 같은 거래일이어도 강제 전송: `--force`.

### basket_watchlist.json 형식

V1 `watchlist.json`처럼 종목마다 `theme`(섹터) 태그만 붙이고, `themes` 블록에 그 섹터가 예측할
타겟 ETF를 지정하면 테마별로 바스켓이 자동 생성됩니다 — 바스켓을 손으로 조립할 필요가 없습니다.

```json
{
  "stocks": [
    {"code": "005930", "name": "삼성전자", "theme": "반도체"},
    {"code": "000660", "name": "SK하이닉스", "theme": "반도체"}
  ],
  "themes": {
    "반도체": {"basket_name": "semiconductor_to_etf", "target": {"code": "091160", "name": "KODEX 반도체"}}
  }
}
```

`target`이 없는 섹터는 바스켓이 만들어지지 않고 조용히 건너뜁니다 — 타겟 ETF를 아직 못 정한
섹터를 그냥 목록에 남겨둬도 됩니다. `basket_name`은 `data/`·`results/` 폴더명(영문 권장, 생략하면
섹터명 그대로).

### 3단계 개별 실행(화면만 다시 만들고 싶을 때 등)

```powershell
python vm_predict/v2_data_collector.py --basket semiconductor_to_etf   # Stage 1: 일봉 수집
python vm_predict/v2_compute_engine.py --basket semiconductor_to_etf   # Stage 2: VM/Breadth/CUSUM/T² 연산
python vm_predict/v2_render_dashboard.py --basket semiconductor_to_etf # Stage 3: 대시보드 HTML만 재생성
python vm_predict/v2_explain.py --basket semiconductor_to_etf --report # 오늘자 신호 사후분해 리포트(CLI)
```

### 바스켓 후보 스캔 (Stage 0, 선택)

종목을 직접 안 골라도, 전체 ETF에서 섹터 대표를 자동으로 찾아 바스켓 후보를 뽑을 수 있습니다.
설계 근거는 `KIS_ETF_섹터스캐너_설계노트.md` 참고.

```powershell
python core/v2_etf_scanner.py                    # 전체 ETF(~1,100여개) 스캔 — 수십 분 소요
python core/v2_etf_scanner.py --limit 60         # 동작만 빠르게 확인하고 싶을 때
```

### 챔피언-챌린저: VM-SPC Core 의사결정 보조 모델 (선택)

Layer 2 피처 5종(칼만 잔차·가격 Z·CUSUM·Hotelling T²·공적분 잔차 Z)으로 종목별 익일 상대순위
(그날 바스켓 내 상위 40%=아웃퍼폼/하위 40%=언더퍼폼)를 예측하는 세 모델(Legacy 룰 / Ridge
Baseline / LightGBM Challenger, 조기종료 적용)을 Walk-Forward(학습 250일 → Embargo 10일 → 검증
60일, 60일씩 슬라이딩)로 비교하고, Gate 1(High-Conf Precision ≥60% & 95% CI 하한 >50% & 표본
≥30건)·Gate 2(오컴의 면도날: Challenger가 +3%p 이상 & CI 하한으로 우위 확인될 때만 채택)로
챔피언을 정합니다. 통과해도 "고신뢰"가 아니라 "통계적으로 확인된 약한 방향성 참고 신호"입니다.
자동 주문은 없고, 장 마감 후 한 번 돌려 익일 시나리오 라인을 사람이 보고 판단합니다.

라벨링 방식 비교(절대방향 vs 상대순위)·Gate 재보정·LightGBM 과적합 진단 등 설계 검증 과정은
`VM_SPC_Core_검증이력.md` 에 따로 정리했습니다.

```powershell
python vm_spc/pipeline.py                                  # 전체 바스켓(상대순위 기본) → 대시보드 반영
python vm_spc/pipeline.py --basket semiconductor_krx_scan  # 특정 바스켓만 (대표 예시)
python vm_spc/pipeline.py --label-mode absolute            # 비교용 구버전 라벨링(별도 파일에 저장)
```

- Stage1 이 저장한 `data/{바스켓}/*.csv` 만 읽습니다(API 호출 없음). 유효 거래일이 280일 미만이라
  fold 를 못 만드는 바스켓은 건너뜁니다.
- 한계: Point-in-time 구성종목 이력이 없어 현재 구성종목으로 근사합니다(생존편향, `vm_spc/universe.py`
  의 `POINT_IN_TIME_HISTORY` 를 채우면 해소). `coint_z` 는 ETF-ex-A 대신 타겟 ETF 대비 롤링 OLS
  스프레드입니다. 둘 다 대시보드 탭의 "검증 설정 · 한계"에 표시됩니다.
- **현재 결과(2026-09-28, 5년 데이터, 대칭 순위 라벨)**: 검증 가능한 31개 중 BASELINE 0 / REJECTED 29 / CHALLENGER 2.
  이전의 "BASELINE 15"는 순위 라벨 비대칭(평소 비율 57%를 50%와 비교)으로 후하게 나온 것이었다(검증이력 9.9·9.10).
  전 바스켓 합산으로는 T+1에서 52.7% vs 평소 50.0%의 약한 신호가 확인된다.
  나머지 8개는 타겟 ETF가 최근 상장돼 유효 거래일이 280일 미만이라 대기 중입니다(같은 섹터의 다른
  ETF로 커버되므로 데이터가 쌓일 때까지 기다림). 대표 예시는 `semiconductor_krx_scan`(타겟 KODEX 반도체).
- 챔피언이 없는(REJECTED) 바스켓도 빈칸으로 두지 않고 Baseline 원값을 흐리게 "게이트 미달·참고"로
  표시합니다 — 계산은 됐지만 기준을 못 넘은 것과, 데이터가 없어 계산 자체가 안 된 것을 구분합니다.

`results/index.html`에 모든 ETF를 한 화면(카드형, 모바일 대응)에서 봅니다. 카드마다 성격이 다른 두
예측을 분리해서 보여줍니다:

- **ETF 자체 예측** — 예측모델(Breadth)이 그 ETF 자신의 익일 방향을 추정한 값과 적중률. 적중률은 그 ETF를
  같은 방향으로 예측했던 날의 적중률에 오늘 괴리율 효과(동의 +3.1%p / 반대 -2.9%p 등, 전 ETF 합산 추정)를
  더한 값이고, 옆의 **평소**는 예측과 상관없이 그 방향으로 움직인 날의 비율입니다 — 실력은 평소 대비 차이.
  적중률이 높은 순으로 정렬합니다(파이프라인의 1차 목표).
- **구성종목 중 최고 신호** — 챔피언-챌린저가 바스켓 안 종목들끼리 비교해 가장 확신이 강한 종목.
  상대적 우열일 뿐, "이 종목 때문에 ETF가 오른다"는 뜻이 아닙니다.

변동성이 응축된 ETF(20일 변동성·고저폭이 모두 자기 과거 250일 하위 20%)에는 **"⚡ 응축 중 N일째 · 큰 움직임
가능성↑ (방향 모름)"** 배지가 붙습니다. 응축 뒤에는 20일 안에 변동성이 1.5배 이상 커지는 경우가 평소의 약 2배라는 게
검증됐지만(ETF 27~28% vs 14~15%), 방향은 알려주지 않습니다(검증이력 9.14, `vm_predict/v2_compression.py`).

외부 CDN 없이 CSS를 파일에 직접 넣어서, 텔레그램으로 받아 폰에서 열어도 그대로 보입니다. 각 카드는
그 바스켓의 `dashboard_v2.html`로 연결됩니다(폴더 구조가 함께 있어야 하므로 폰에서는 구글 드라이브의
`VM-SPC_Core/results`에서 여는 것을 권장). `python vm_spc/pipeline.py`(전체 바스켓, `--no-render`
아닐 때)를 돌리면 자동으로 다시 생성됩니다. 따로 만들고 싶으면 `python vm_spc/build_index.py`.

### ETF 순위(섹터 순환, 프로토타입)

ETF 단위 SPC 상태(breadth·칼만·CUSUM·T²·괴리율·20일 상대수익)로 향후 1·5·10·20거래일 동안 KODEX 200보다 더 오를
확률을 점수로 매기고, 날짜별 ETF 간 순위 상관으로 검증합니다(테마 대표 ETF 16개 기준). 현재 T+1만 통과했고 그 신호는
대부분 괴리율 되돌림에서 옵니다. 인덱스 상단 랭킹 카드와 ETF 상세 예측 카드 아래 한 줄로 표시 — 검증이력 9.12절.

```powershell
python etf_rank/run.py          # 계산 → 검증 → 대시보드·인덱스 재생성 (v2_run.py 이후)
python etf_rank/run.py --dry    # 검증 결과만
```

### 종목 추적(역방향, 프로토타입)

ETF·지수 흐름 + 종목 통계로 각 종목이 1·5·10·20거래일 뒤 **소속 시장 지수보다 더 오를지**를 봅니다(기존 예측모델의
반대 방향). 바스켓마다 그 ETF 기준으로 보고, 종목별 ETF 내 비중을 함께 표시합니다. 기간별로 전 바스켓을 합쳐
검증하고, 통과한 기간만 Active(나머지는 흐리게 HOLD). 현재 4개 기간 모두 HOLD — 결과와 설계 이력은
`VM_SPC_Core_검증이력.md` 9.6·9.7절. 아직 일일 자동 실행에는 넣지 않았습니다.

```powershell
python vm_predict/backfill_history.py      # (1회) 일봉·NAV 이력을 과거 방향으로 5년까지 확장
python stock_track/run.py                  # 기준 지수·ETF 비중 갱신 → 계산 → 검증 → 대시보드 "종목 추적" 탭
python stock_track/run.py --no-fetch       # API 호출 없이
```

### PAIR-SPC: 섹터 대표종목 실증 + 공적분 SPC (신규, 선택)

"섹터 ETF는 대표 종목 하나로 대변 가능한가"를 먼저 실증(ex-self 지수 대조)하고, 실증된 종목만
공적분·SPC 관리도로 넘기는 별도 파이프라인. 설계 근거는 `PAIR_SPC_v2_대표종목검증_스펙.md` 참고.

```powershell
python pair_spc/run_pair_spc.py                                   # 전체 바스켓 Stage0→1/2→대시보드 반영
python pair_spc/run_pair_spc.py --basket semiconductor_to_etf     # 특정 바스켓만
python pair_spc/pair_representativeness.py --basket <이름>         # Stage 0만 (대표종목 실증)
python pair_spc/pair_cointegration.py --basket <이름>              # Stage 1/2만 (Stage0 결과 필요)
```

대표성 기준(ex-self 동시상관 ≥ 0.7)을 통과한 종목이 하나도 없으면 그 바스켓은 Stage1/2를 건너뛰고
"단일 종목으로 대변 불가"라는 결론 자체를 `stage0_representativeness.json`에 남깁니다(억지로 다음
단계로 넘기지 않음). 결과는 `results/{바스켓}/pair/`에 저장되고, 기존 `dashboard_v2.html`의
"대표종목 상관관계 (PAIR-SPC)" 탭에서 확인합니다.

`basket_watchlist.scanned.json`(`basket_watchlist.json`과 같은 스키마)을 생성합니다 — **이 파일을
그대로 쓰지 말고, 검토 후 필요한 테마만 골라 `basket_watchlist.json`에 직접 옮기세요.** 자동
필터는 레버리지/인버스·커버드콜류를 걸러내지만, "KOSPI200"처럼 시장 전체를 추종하는 ETF나
분류가 애매한 신상품은 `v2_etf_scanner.py`의 `SECTOR_BLACKLIST_DEFAULT`/`NAME_EXCLUDE`에 안 걸리고
나올 수 있습니다 — 설계노트도 "최종 리스트는 하드코딩하고 스캐너는 신규 후보 점검용으로만 쓸 것"을
권장합니다.

## 구성

### core/ — 공용 인프라

| 파일 | 역할 |
|---|---|
| `v2_config.py` | 경로(`ROOT`는 이 파일의 부모의 부모, 즉 프로젝트 루트)·`Params`(튜닝 임계값)·바스켓 로더 |
| `kis_auth.py` / `kis_client.py` | 한투 Open API 인증·일봉 조회 (V1에서 독립 구성) |
| `secrets_loader.py` | 비밀값 로더(환경변수 우선, 없으면 `.streamlit/secrets.toml`) |
| `v2_datastore.py` | `data/{basket}/{code}.csv` 일봉 저장소 (`save_bars(merge=True)`로 증분 병합) |
| `v2_signals.py` | 4개 신호(가격밴드·거래량·매물대·상대강도) 연속값 + 시그모이드 VM Score |
| `v2_kalman.py` | 인과적 칼만필터(로컬선형추세/로컬레벨) — V1과 동일 |
| `v2_cusum.py` | 양방향 CUSUM 누적합 관리도 |
| `v2_hotelling.py` | Hotelling's T² 다변량 SPC (신호별 기여도 분해 포함) |
| `v2_etf_scanner.py` | Stage 0(선택) — 전체 ETF에서 섹터 대표를 찾아 바스켓 후보 생성 |
| `v2_kospi_master.py` | KOSPI 종목마스터 다운로드·파싱 (ETF 전체 목록 확보용, 인증 불필요) |

### vm_predict/ — 예측모델(VM-SPC) 파이프라인

| 파일 | 역할 |
|---|---|
| `v2_run.py` | 3단계 한 번에 실행하는 통합 진입점 (`--basket`·`--no-fetch` 지원) |
| `v2_data_collector.py` | Stage 1 — 센서·타겟 일봉(`FHKST03010100`) 증분 수집 + 수정주가 소급 조정 감지(`update_bars`), 타겟 ETF NAV·괴리율(`FHPST02440200`)·상장좌수 스냅샷(`update_nav`) |
| `v2_compute_engine.py` | Stage 2 — VM Score·Breadth·컨퓨전매트릭스·CUSUM·T²·일일 리포트 연산 |
| `v2_render_dashboard.py` | Stage 3 — 단일 HTML 대시보드 생성 (PAIR-SPC 탭도 여기서 얹음) |
| `v2_explain.py` | breadth/T² 신호를 종목별·신호별로 사후 분해(CLI, `--report`로 요약문 출력) |
| `v2_premium.py` | 타겟 ETF 괴리율(시장가 vs NAV) 표준화·예측과의 동의/반대 판정 (보정값은 `v2_etf_extras`가 전 ETF 합산으로 추정) |
| `v2_reliability.py` | 예측 신뢰도 지수(T² 유사도·적중 CUSUM·변동성 국면) — 검증 미통과로 현재 화면 미표시 |
| `v2_etf_extras.py` | 위 둘을 전 바스켓 계산 + 풀링 검증(`results/etf_extras_validation.json`), 통과 항목만 화면 반영. `v2_run.py`가 자동 호출 |
| `v2_investor_flow.py` | (실험적) 투자자별 순매수 수집 + 기타기관(기관합계-금융투자) CUSUM — 검증 결과 유의미한 예측력은 못 찾음, 인프라만 남겨둠 |

### pair_spc/ — 대표종목 상관관계(PAIR-SPC) 파이프라인 (신규)

| 파일 | 역할 |
|---|---|
| `run_pair_spc.py` | Stage0→1/2 한 번에 실행 + 대시보드 재렌더링 |
| `pair_representativeness.py` | Stage 0 — ex-self 지수 대조로 섹터 대표종목 실증 (센서 밖 후보 종목도 타겟 최신일까지 갱신) |
| `pair_cointegration.py` | Stage 1/2 — OLS 공적분 회귀·ADF·Half-life·히스테리시스 SPC |
| `pair_config.py` | PAIR-SPC 전용 경로·`PairParams` |

### vm_spc/ — 챔피언-챌린저(VM-SPC Core)

| 파일 | 역할 |
|---|---|
| `pipeline.py` | EOD 오케스트레이션 엔트리포인트 (피처→레이블→Walk-Forward→판정→시나리오 JSON) |
| `features.py` | Layer 2 피처 5종 — core/ 칼만·CUSUM·T² 함수를 재사용하는 wrapper |
| `labeling.py` | 라벨링 — 상대순위(기본) / 절대방향(±0.2% 데드존, 비교용) |
| `universe.py` | Point-in-time 유니버스 (이력 미확보 시 현재 구성종목 근사 + 생존편향 경고) |
| `splitter.py` | Walk-Forward + Purge/Embargo 분할 |
| `models/` | `legacy_rule.py`(VM Score≥0.75) / `ridge_baseline.py` / `lgbm_challenger.py`(조기종료 적용) |
| `evaluation.py` | Tier1/2 지표·Brier·F1 + 날짜 클러스터 부트스트랩 CI(B=1000) |
| `gate_decision.py` | 3-분기 Gated Two-Stage 챔피언 판정 |
| `build_index.py` | `results/index.html` — 전체 ETF를 ETF 자체 예측 적중률 순으로 한 화면에 정리 |
| `notify_telegram.py` | 일일 요약 + `index.html` 텔레그램 전송 (같은 거래일 중복 전송 방지) |

### scripts/ — 자동 실행

| 파일 | 역할 |
|---|---|
| `run_daily_pipeline_hidden.vbs` | 작업 스케줄러가 호출하는 진입점 — 창 없이 아래 .ps1 실행 |
| `run_daily_pipeline.ps1` | 수집→PAIR-SPC→챔피언-챌린저→드라이브 동기화→텔레그램 순차 실행 + 로그 (UTF-8 BOM 필수) |

### 루트

| 파일 | 역할 |
|---|---|
| `basket_watchlist.json` | 종목별 테마 태그 + 테마별 타겟 ETF 실제 구성(현재 39개 테마) — `.gitignore` 대상 |
| `VM_SPC_Core_검증이력.md` | 챔피언-챌린저 검증 과정과 운영 중 수정 이력 |

## 설계상 확정된 판단 (구현 시 참고)

- **VM Score(센서 4개 신호)는 수급(5번째 신호) 제외.** Stage1이 일봉만 수집하고 수급 API를
  호출하지 않기 때문 — 데이터 소스가 하나로 유지됩니다.
- **Hotelling's T² 입력은 3개 신호(가격밴드·거래량·매물대)로, 상대강도(rs_excess)도 추가 제외합니다.**
  바스켓을 균등가중 평균하면 leave-one-out 초과수익의 합은 Σ(rᵢ − mean_{j≠i} rⱼ) = 0 항등식이
  성립해(커버리지가 균일할 때) 그 열의 분산이 기계입실론 수준으로 사라지고, 공분산행렬이 특이해져
  SVD가 수렴하지 못합니다 — 센서 수가 적고 결측이 거의 없는 바스켓(`power_equipment_to_etf`)에서
  실제로 재현된 버그였습니다. 개별 센서의 VM Score(`rs_score`)는 이 항등식과 무관해 영향받지
  않습니다. `v2_hotelling.py`에는 추가로 작은 능형(ridge) 보정도 넣어 유사 사례를 방어합니다.
- **T²는 바스켓 평균(센서 N종목의 일별 신호 평균) 시계열에 적용합니다.** breadth CUSUM(Panel 3)과
  마찬가지로 "바스켓 전체" 레벨의 이상 탐지이며, Panel 2(타겟 ETF 자체 CUSUM)와는 대상이 다릅니다.
- **연속 VM Score는 V1의 boolean 판정과 같은 경계값에서 시그모이드로 0.5를 지나가도록 설계**했습니다
  (`v2_signals.soft_gate`) — 겉보기엔 연속점수이지만 V1과 동일한 통과/미달 규칙을 유지합니다.
- **결측 처리 철학이 V1과 다릅니다.** V1은 일별 리포트용이라 데이터 부족 시 "보수적으로 0점"
  처리하지만, V2는 breadth 컨퓨전 매트릭스 백테스트 통계가 왜곡되지 않도록 명시적 결측(NaN)으로
  남기고 집계에서 제외합니다.
- **컨퓨전 매트릭스에 데드존(`Params.dead_zone`, 기본 0.2%)을 둡니다.** 익일 등락폭이 이 값
  미만인 날은 방향이 사실상 동전던지기라 판정 자체를 안 하고 집계에서 뺍니다. KOSPI top10
  기준 0.1~2.0% 전 구간에서 z가 대체로 1.96을 넘어(0.2%에서 1.68→2.55) 단일 지점 우연이 아닌
  것으로 보이나, 반도체 바스켓 교차검증에서는 같은 방향이지만 z가 꾸준히 1.96을 넘진 않아
  (0.2%에서 1.73) 바스켓마다 효과 크기가 다릅니다. `breadth_threshold` 재탐색과 달리 "결과가
  좋은 값을 사후에 고른" 다중비교가 아니라 "확신 없는 날은 판정을 안 한다"는 사전에 동기부여된
  규칙이라는 점에서 성격이 다릅니다. 근거는 `v2_config.Params.dead_zone` 주석에 상세히 남겼습니다.
- **`breadth_threshold`는 0.345로 고정합니다.** 데이터가 바뀔 때마다 다시 지정하지 않기로 한
  운영상 결정입니다. 다만 이 값은 과거 전체 기간 분포로 정한 것이라, index/대시보드의 "ETF 자체
  예측 과거 적중률"은 약간 낙관적으로 편향돼 있을 수 있다는 점을 감안해서 봅니다.

## 참고

- 모든 계산은 인과적입니다(미래 데이터를 쓰지 않음, look-ahead 방지). 임계값은 `v2_config.Params`
  에 모여 있습니다.
- 장 마감 후 계산한 참고용 방향 신호이며 투자 권유가 아닙니다. 자동 주문은 하지 않습니다.
