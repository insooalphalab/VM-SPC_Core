# VM-SPC Core

국내 섹터 ETF 39개와 구성종목을 매일 장 마감 후 분석하는 **의사결정 보조 도구**입니다. 반도체 공정의 가상계측(VM)과
SPC(칼만 · CUSUM · Hotelling T²) 기법을 주가에 적용합니다. 평일 16:30에 자동으로 돌고, 결과를 텔레그램과 구글 드라이브로 받습니다.
자동 주문은 없고, 진입 여부는 사람이 판단합니다.

## 이 도구가 알려주는 것

2016년부터의 데이터로, 기준을 먼저 적고(사전 등록) 두 기간(2016–2021-05 / 2021-06 이후)과 코스피 장세 3분할(60일선 기준 상승장 · 횡보·전환 · 하락장)로
판정한 결과입니다. 숫자는 비용 0.3%를 뺀 값이고, R = 손절 거리 1개 단위 손익, 배수 = 수익 배수(이긴 거래 R 합 ÷ 진 거래 R 합).
자세한 수치와 과정은 `VM_SPC_Core_검증이력.md`.

| 등급 | 내용 | 숫자 | 화면 |
|---|---|---|---|
| **검증됨** | 상승장 · RS 70 이상 30일 박스 첫 돌파 → 다음날 시가, 손절 돌파일 종가 − 1ATR, 50일선 이탈까지 보유 (9.62 · 9.72) | 승률 17.4% · +0.45R · ×1.51 | 오늘의 후보 돌파 목록 |
| | 하락장에서 추세 강도 상위 20% 보유 종목은 이후 시장보다 약함 → 이익 실현 고려 (9.81) | 이후 20일 시장 대비 −1.3%p에서 −3.1%p (두 기간 모두) | 보유 탭 |
| | 응축(변동성 · 고저폭 하위 20%) 뒤 20일 안 변동성 1.5배 이상 확대 (9.14) | ETF 27% vs 평소 14% | 인덱스 ⚡ 배지 |
| | ETF 괴리율 되돌림 (9.5 · 9.9) | +6.7%p | 인덱스 예측 카드 |
| **최근경향** (2021-06 이후 두 반쪽 모두 같은 방향, 매년 재판정) | 횡보장 가짜 이탈(박스 하단 이탈 후 3일 안 복귀) (9.85) | +0.58R | 오늘의 후보 가짜 이탈 목록 |
| | 횡보장 T² 동반 가짜 이탈 (9.37 · 9.84: 앞 −0.24R · 최근 +0.95R) | 전체 기간 승률 51% · +0.49R · ×1.99 | 후보 승률 아래 R · 배수 |
| | 상승장 추세 강도 상위 20% = 보유 유리, 횡보장 60일선 위치 상위 20% = 되돌림 주의 (9.81 · 9.86) | +2.4%p / −1.3%p (20일) | 보유 탭 |
| **승률 모형** | 가짜 이탈: T² · 장세 · 이탈 깊이 · 복귀 캔들 종가 위치 · 이탈일 프로그램 매도 z (9.45 · 9.48 · 9.79) | 2021-06 이후 AUC 0.61 | 후보 승률, ★ 깊이 6% 이상 · ☆ 3.5–6% |
| **기대치 안내** | 가짜 이탈 매수 후 1–3일째 종가 위치별 손절로 끝날 확률 (9.92) | 3일째 −0.5R 아래 82% · +0.5R 이상 29% (평소 59%) | 셋업 보유 탭 |
| **약함** (방향 일관, 기준 미달) | 돌파의 매출 +25% · 이익 +20% (9.78), 과열 돌파(상단 +4% 초과) R 낮음 (9.67), 보유 판단 참고 태그 (9.87) | | 칩 · 참고 태그 |
| **상태 정보** | 바스켓 · 종목 PER/PBR 5년 위치, 칼만 관리선 평소 범위 | | 인덱스, 리스크 페이지 |

**방향 정보 없음** (기준 미달 또는 기간마다 엇갈림 — 화면에 넣지 않음)

| 영역 | 확인한 것 (절) |
|---|---|
| 수급 | 투자자별 · 기관 세부 · 쌍끌이 (9.15 · 9.21–9.23), 20 · 60 · 120일 누적 (9.82), 매집 · 분산 · 가격 충격 비대칭 (9.99), 돌파일 프로그램 순매도 (9.100) |
| 돌파 변형 | 돌파의 힘 · 거래량 · 베이스 길이 · 이평선 동시 돌파 · RS 가속 · 다른 진입 방아쇠 · 익절 · 본전 손절 · 나눠 사기 · 확인 후 매수 (9.52–9.77), 거래량 가뭄 돌파 (9.101), 6개월 매물대 돌파 (9.108 · 9.109 — 현행과 같은 수준) |
| 횡보장 | 횡보장 돌파 · 지수 20일선 · 상승장 전환일 매수 (9.89–9.91), 박스 하단 지지 · 단기 과매도 · 공포 상태 지수 매수 (9.103), 업종 대비 되돌림 (9.104) |
| 가짜 이탈 변형 | 3일 기다려 매수 (9.94), 강한 복귀 다음날 매수 (9.95), 복귀일 종가 매수 (9.107), 업종 과열 시 조기 정리 (9.105), 10일 짧은 호흡 (9.102) |
| 기타 | 1–20일 ETF · 종목 방향, DART 공시 (9.17), 교과서 패턴 4종 (9.29 · 9.30), 유동성 지표 (9.98), 다중 MACD 단일 종목 타이밍 (9.96 · 9.97), 종목 리포트 3개사 상향 (9.106), 장세 GMM 군집 (9.84) |

추세 추종형(돌파)은 코스피 상승장에서만, 되돌림형(가짜 이탈)은 횡보 · 하락장에서 됩니다. 그래서 두 페이지 맨 위에
**지금 장세와 그 장세에 맞는 전략**을 한 줄로 보여 줍니다. 검증 틀은 알려진 5일 단기 되돌림(순위 상관 −0.024)을 잡아내는 것으로
확인했습니다(9.28) — 위의 "없음"들은 틀이 둔해서가 아니라 실제로 없는 것입니다.

## 화면

- **`results/index.html`**: 전체 ETF 카드(모바일 대응, 외부 CDN 없음). ETF 자체 예측과 적중률(평소 비율 병기), 구성종목 중 최고 신호,
  응축 배지, 바스켓 PER/PBR, 리스크 페이지 링크, 맨 아래 섹터 리포트 흐름 한 줄. 카드를 누르면 바스켓별 `dashboard_v2.html`.
- **`results/scenario/risk_scenarios.html`**: **보유 · 관심 종목**. 맨 위 장세 한 줄 · 시황 한 줄 · 보유 요약(종목당 오늘 할 일).
  - **보유**(평단 무관): 장세별 순위 판단(위 표), "(강함)"은 상승장 상위 5%만(9.93). 매수일을 적으면 그 셋업 규칙(돌파: 손절 · 50일선 · 120일 /
    가짜 이탈: 손절 · 박스 상단 · 20일)을 추적하고, 가짜 이탈은 1–3일째 손절 확률을 표시. 탭 위쪽은 현재 상태 숫자(50 · 60일선, 추세 순위, 52주 고가, 5일선)와 50일선 차트.
  - **관심**: 오늘의 후보와 같은 매수 규칙(상승장 RS 70 돌파 / 그 외 가짜 이탈), 근거 점수(검증됨 3 · 유망 2 · 최근경향 1.5 · 약함 1 · 참고 0.5), 탭 노란 점 = 오늘 매수 신호.
  - **참고 자료(트래커)**: 탭 맨 아래 접힌 칸 — 리포트 60일 목록, 최근 5분기 실적(YoY · QoQ · 이익률), 90일 자본 정책 공시, 수급 20 · 60일 누적, PER/PBR 5년 범위. 판단에 쓰지 않음.
- **`results/scenario/screen.html`**: **오늘의 후보** — 코스피 시총 상위 200에서 지금 장세 전략표(`scenario/screen.py` PLAYBOOK)에 맞는 종목.
  시점별 목록(오늘 종가로 완료 → 내일 시가 / 최근 5일 안 완료 · 아직 유효 / 조건 전) 안에서 승률 높은 순. 오른쪽 숫자 = 승률 · 기대값 R · ×수익 배수 · 평소 승률.
  얕은 이탈(3.5% 미만) · 거래소 조치 종목은 뺌. 탭은 자바스크립트 없이 동작.

## 빠른 시작

```powershell
pip install -r requirements.txt                                    # Python 3.11+
copy .streamlit\secrets.example.toml .streamlit\secrets.toml       # KIS 앱키 · DART 키 · 텔레그램 입력
copy basket_watchlist.example.json basket_watchlist.json           # 종목 · 테마 · 타겟 ETF
copy scenario_targets.example.json scenario_targets.json           # 보유 · 관심 종목
python vm_predict/v2_run.py                                        # 수집 → 연산 → 대시보드
python vm_spc/pipeline.py                                          # 챔피언-챌린저 + index.html
```

`basket_watchlist.json`은 종목마다 `theme` 태그를 달고 `themes`에 테마별 타겟 ETF를 지정하면 바스켓이 자동으로 만들어집니다.
`scenario_targets.json`은 `{"stocks": [{"code": "005930", "hold": true}, {"code": "039490", "hold": true, "buy_date": "2026-10-02"}, {"code": "222800"}]}` 형식입니다.
`hold` = 보유, `buy_date` = 셋업으로 산 날(그 셋업 규칙을 추적), 둘 다 없으면 관심 종목. 평단은 쓰지 않습니다.
두 파일과 `.streamlit/secrets.toml`은 깃허브에 올리지 않고(`.gitignore`), 앞의 두 파일은 매일 구글 드라이브에 백업됩니다.

## 매일 자동 실행

작업 스케줄러 "VM-SPC Core 일일 업데이트"가 평일 16:30에 `scripts/run_daily_pipeline_hidden.vbs` → `run_daily_pipeline.ps1`을
실행합니다(PC가 꺼져 있었으면 켜질 때 따라잡음). 약 15분, 로그는 `logs/pipeline_*.log`에 30일 보관.

1. `vm_predict/v2_run.py`: 일봉 증분 수집 → 연산 → 대시보드
2. `pair_spc/run_pair_spc.py`: 대표종목 공적분 SPC
3. `dart_events/collect.py --recent`: DART 최근 2년 재무 · 공시 + PER/PBR
4. `reports/run.py`: 텔레그램 6개 채널 새 글 → 리포트 표 · 종목 뉴스 · 시황 한 줄 · 거래소 조치 · 섹터 리포트 흐름
5. `vm_spc/pipeline.py`: 챔피언-챌린저 → `index.html`
6. `scenario/render_risk.py`: 보유 · 관심 페이지(목록 종목 수급 · 신용 증분 포함) → `stock_track/collect_program.py --screen` → `scenario/screen.py`: 오늘의 후보
7. `results/` → `G:\내 드라이브\VM-SPC_Core` 미러링, `basket_watchlist.json` · `scenario_targets.json` 백업
8. `vm_spc/notify_telegram.py`: ① ETF 요약 + `index.html` ② 보유 · 관심 요약 + `risk_scenarios.html` ③ 오늘의 후보 + `screen.html`
   (실패 시 5 · 15초 뒤 재시도, 같은 거래일은 생략, 앞 단계 실패 시 경고)
9. 금요일만(급하지 않은 수집): 검증용 791종목 프로그램매매 · 신용잔고 · DART 분기 재무, 센서 191종목 투자자별 수급, 코스피 시장 투자자별 수급

주의:
- 스케줄러는 PowerShell을 VBScript로 창 없이 띄웁니다. `powershell -WindowStyle Hidden`은 창이 보이고, 클릭하면 프로세스가 멈춥니다.
- `run_daily_pipeline.ps1`은 **UTF-8 BOM 포함**으로 저장해야 합니다. 없으면 PowerShell 5.1이 한글 경로를 깨뜨립니다.
- 텔레그램 확인만: `python vm_spc/notify_telegram.py --dry-run`, 강제 재전송: `--force`, 빠진 묶음만: `--only etf|stock|screen`.

## 도구별 실행

| 도구 | 명령 | 비고 |
|---|---|---|
| 예측모델 (VM-SPC) | `python vm_predict/v2_run.py [--basket 이름] [--no-fetch]` | 매일 |
| 챔피언-챌린저 | `python vm_spc/pipeline.py [--basket 이름]` | 매일, 9.10 |
| PAIR-SPC | `python pair_spc/run_pair_spc.py` | 매일 |
| DART | `python dart_events/collect.py --recent` · `--validation` · `valuation.py` · `growth.py` | 9.17 · 9.78 |
| 보유 · 관심 페이지 | `python scenario/render_risk.py [종목코드 ...] [--no-fetch]` | 매일 |
| 오늘의 후보 | `python scenario/screen.py [--no-fetch]` | 매일, 전략표 = `PLAYBOOK` |
| 텔레그램 수집 · 파싱 | `python reports/run.py` · `tg_login.py`(첫 로그인, 사용자가 직접) | 매일. 채널: butler_works · ked_epic_ai(리포트), aicorporateanalysisdeepdive(뉴스), shmstory(시황), darthacking(거래소 조치), easobi(심리, 표시 안 함) |
| 후보 승률표 | `python research/build_winrate_table.py` | 몇 달에 한 번(약 15분), 승률 · R · 수익 배수 |
| 수급 · 프로그램 · 신용 | `python stock_track/collect_investor_detail.py` · `collect_program.py` · `collect_credit.py` [`--validation` · `--screen`] · `collect_market_investor.py` | 검증용 791종목만(전 종목은 하지 않음) |
| ETF 순위 / 종목 추적 | `python etf_rank/run.py` · `python stock_track/run.py` | 수동, 화면 제외(9.12 · 9.7) |
| 바스켓 후보 스캔 | `python core/v2_etf_scanner.py [--limit 60]` | 결과는 검토 후 직접 옮김 |

검증 스크립트는 `research/`에 모았습니다(재현용). 결과는 `results/*.json`.

| 검증 | 명령 (`python research/…`) | 절 |
|---|---|---|
| 정방향 도구 · 섹터 장기 이력 · DART | `forward_horizons.py` · `fetch_long_history.py` · `validate_dart_events.py` | 9.9 · 9.16 · 9.17 |
| 박스 패턴 · 거래량 · 신고가 · 수급 CUSUM | `validate_box.py` · `validate_volume.py` · `validate_high52.py` · `validate_flow_cusum.py` · `flow_cusum_grid.py` · `validate_joint_flow.py` | 9.18–9.23 |
| 리포트 → 섹터 · 레벨 비교 · 메커니즘 · 패턴 | `validate_report_sector.py` · `validate_levels.py` · `validate_mechanisms.py` · `validate_flag.py` · `validate_patterns.py` | 9.25–9.30 |
| 셋업 손절 · 익절 · 탐색 | `validate_stops.py` · `validate_targets.py` · `validate_weak_touch.py` · `explore_reversal.py` · `explore_nolift.py` | 9.31–9.35 |
| 장세 · 가짜 이탈 보조 신호 | `validate_market_trend.py` · `validate_regime.py` · `validate_compression_regime.py` · `validate_breakout_trend.py` · `validate_sentiment.py` · `validate_credit_failure.py` · `validate_program_shock.py` · `validate_signal_combo.py` · `validate_recover_window.py` · `validate_market_arb.py` · `validate_program_flip.py` | 9.37–9.51 · 9.79 |
| 돌파 강도 · 추세추종 원칙 · RS 변형 | `validate_breakout_edge.py` 외 9.52–9.61 스크립트 · `validate_trend_principles.py` · `validate_rs_divergence.py` · `validate_rs_accel.py` · `validate_rs_rising.py` · `validate_base_duration.py` · `validate_breakout_margin.py` | 9.52–9.67 |
| 진입 방아쇠 · 청산 · 확인 매수 · 거래량 · 실적 | `validate_entry_trigger.py` · `validate_breakout_exits.py` · `validate_confirm_entry.py` · `validate_breakout_volume.py` · `validate_earnings_growth.py` 외 | 9.68–9.78 |
| 보유 판단 · 수급 장기 · 메타 라벨 · GMM · 최근경향 | `validate_holding_factors.py` · `validate_holding_regime.py` · `validate_holding_flows.py` · `validate_meta_label.py` · `validate_regime_gmm.py` · `reclassify_recent.py` · `validate_sideways_holding.py` · `holding_weak_tier.py` · `holding_overlap_count.py` | 9.80–9.88 |
| 횡보장 돌파 · 가짜 이탈 변형 | `validate_sideways_breakout.py` · `validate_sideways_ma20.py` · `validate_sideways_flip.py` · `validate_fail_early.py` · `validate_fail_wait3.py` · `validate_fail_strong_day.py` · `validate_fail_entry_close.py` | 9.89–9.95 · 9.107 |
| 60일선 문턱 · MACD · 유동성 · 수급 3가설 · 거래량 가뭄 | `validate_ma60_threshold.py` · `validate_macd_sector_rule.py` · `validate_liquidity.py` · `validate_flow_hypotheses.py` · `validate_breakout_prog_sell.py` · `validate_volume_drought.py` | 9.93 · 9.96–9.101 |
| 짧은 호흡 · 횡보 되돌림 · 업종 · 리포트 · 매물대 | `validate_short_rules.py` · `validate_sideways_reversion.py` · `validate_industry_reversion.py` · `validate_industry_overheat_exit.py` · `validate_stock_reports.py` · `validate_volume_profile_breakout.py` · `validate_volume_profile_first3.py` | 9.102–9.109 |

## 폴더

```
core/         공용 인프라: KIS 인증 · 클라이언트, 설정(Params), 칼만 · CUSUM · T² · 신호 계산, ETF 스캐너
vm_predict/   예측모델: 수집 → 연산 → 대시보드, 괴리율 · 응축 · 신뢰도
vm_spc/       챔피언-챌린저(Walk-Forward + 게이트), index.html, 텔레그램
pair_spc/     대표종목 실증 + 공적분 SPC
dart_events/  DART 재무 · 공시 수집, PER/PBR 상태, 실적 성장
scenario/     보유 · 관심 페이지, 오늘의 후보(screen.py), 박스 규칙 · T² 판정(box_rules.py — 검증과 화면이 같이 씀)
stock_track/  수급 · 프로그램 · 신용 · 시장 투자자별 수집, 검증용 종목 목록(universe_all.py)
etf_rank/     섹터 ETF 순위(수동, 화면 제외)
reports/      텔레그램 6개 채널 수집 · 리포트/뉴스 파싱 · 시황 한 줄 · 거래소 조치 · 섹터 리포트 흐름
research/     검증 스크립트(재현용)
scripts/      매일 자동 실행(VBS + PowerShell)
data/ results/ state/ logs/   자동 생성물(.gitignore)
```

모든 스크립트는 프로젝트 루트에서 `python <폴더>/<파일>.py`로 실행합니다.

## 설계상 확정된 판단

- **모든 계산은 인과적**입니다(그날까지의 데이터만 사용). 임계값은 `core/v2_config.Params`에 모여 있습니다.
- **검증은 사전 등록 → 두 기간 · 장세 3분할 → 블록 부트스트랩**으로 합니다. 결과를 보고 기준을 바꾸지 않습니다.
  확률은 항상 평소 비율과, 규칙 성과는 승률 · R · 수익 배수를 함께 보여 줍니다.
- **"최근경향" 등급**: 2016–2021엔 없거나 반대였지만 2021-06 이후 두 반쪽 모두 같은 방향인 결과. 시장이 바뀌면 사라질 수 있어 매년 다시 판정합니다.
- **`breadth_threshold`는 0.345로 고정**합니다(데이터가 바뀔 때마다 재지정하지 않음).
- **T² 입력은 3개 신호**(가격밴드 · 거래량 · 매물대)입니다.
- **한계**: 구성종목 과거 이력(point-in-time)이 없어 현재 구성으로 근사합니다(생존 편향).

장 마감 후 계산한 참고 정보이며 투자 권유가 아닙니다.
