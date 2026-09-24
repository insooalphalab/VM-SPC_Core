# VM-SPC Core (KIS_TA V2)

V1(5대시그널 스코어카드, 실전용)의 칼만필터·신호 계산을 재사용해 **가상계측(Virtual Metrology)**
과 **SPC 드리프트 탐지(CUSUM + Hotelling's T²)** 를 증명하는 포트폴리오/해커톤용 확장입니다.
실사용(매매 판단) 목적이 아니라, 반도체 FDC/VM 엔지니어링 패턴이 주식 데이터로 그대로
전이되는지를 코드와 시각화로 보여주는 데 목적이 있습니다. 설계 배경은 `kis_ta_v2_design.md` 참고.

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
pair_spc/     — 대표종목 상관관계(PAIR-SPC) 파이프라인(신규): Stage0 대표종목 실증 → Stage1/2
                공적분·SPC. vm_predict의 dashboard_v2.html에 탭으로 결과를 얹습니다.
data/ results/ state/  — 이전과 동일하게 프로젝트 루트에 그대로 있습니다(폴더 이동과 무관)
```

## 빠른 시작

```powershell
pip install -r requirements.txt                                    # Python 3.11+
copy .streamlit\secrets.example.toml .streamlit\secrets.toml       # KIS 앱키/시크릿 입력
copy basket_watchlist.example.json basket_watchlist.json           # 실제 종목·타겟 ETF로 수정
python core/kis_client.py verify                                   # 종목코드↔종목명 검증
python vm_predict/v2_run.py                                        # 예측모델: 수집→연산→렌더링 한 번에
python pair_spc/run_pair_spc.py                                    # PAIR-SPC: 대표종목 실증→공적분·SPC (선택)
```

`results/{바스켓이름}/dashboard_v2.html`을 더블클릭해 브라우저로 엽니다(서버 불필요) — 상단
"예측모델 (VM-SPC)" / "대표종목 상관관계 (PAIR-SPC)" 탭으로 전환합니다(PAIR-SPC를 아직 안 돌렸으면
탭 자체가 안 보이고 예측모델 화면만 나옵니다 — 하위 호환).
`basket_watchlist.json`에 타겟 ETF가 지정된 섹터가 여러 개면 `v2_run.py`가 전부 순서대로 처리합니다.
특정 하나만: `python vm_predict/v2_run.py --basket semiconductor_to_etf`. 이미 수집한 데이터로
연산·화면만 다시: `python vm_predict/v2_run.py --no-fetch`(API 호출 안 함, `data/*.csv` 재사용).

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
| `v2_datastore.py` | `data/{basket}/{code}.csv` 일봉 저장소 |
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
| `v2_data_collector.py` | Stage 1 — 센서·타겟 일봉(`FHKST03010100`) 수집 |
| `v2_compute_engine.py` | Stage 2 — VM Score·Breadth·컨퓨전매트릭스·CUSUM·T²·일일 리포트 연산 |
| `v2_render_dashboard.py` | Stage 3 — 단일 HTML 대시보드 생성 (PAIR-SPC 탭도 여기서 얹음) |
| `v2_explain.py` | breadth/T² 신호를 종목별·신호별로 사후 분해(CLI, `--report`로 요약문 출력) |
| `v2_investor_flow.py` | (실험적) 투자자별 순매수 수집 + 기타기관(기관합계-금융투자) CUSUM — 검증 결과 유의미한 예측력은 못 찾음, 인프라만 남겨둠 |

### pair_spc/ — 대표종목 상관관계(PAIR-SPC) 파이프라인 (신규)

| 파일 | 역할 |
|---|---|
| `run_pair_spc.py` | Stage0→1/2 한 번에 실행 + 대시보드 재렌더링 |
| `pair_representativeness.py` | Stage 0 — ex-self 지수 대조로 섹터 대표종목 실증 |
| `pair_cointegration.py` | Stage 1/2 — OLS 공적분 회귀·ADF·Half-life·히스테리시스 SPC |
| `pair_config.py` | PAIR-SPC 전용 경로·`PairParams` |

### 루트

| 파일 | 역할 |
|---|---|
| `basket_watchlist.json` | 종목별 테마 태그 + 테마별 타겟 ETF 실제 구성 — `.gitignore` 대상 |

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

## 참고

- 모든 계산은 인과적입니다(미래 데이터를 쓰지 않음, look-ahead 방지). 임계값은 `v2_config.Params`
  에 모여 있고 백테스트로 튜닝하는 것을 전제로 한 초기값입니다.
- 이 도구는 방법론 검증용이며 투자 권유가 아닙니다.
