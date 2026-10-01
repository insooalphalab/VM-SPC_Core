# VM-SPC Core

국내 섹터 ETF 39개와 구성종목을 매일 장 마감 후 분석하는 **의사결정 보조 도구**입니다. 반도체 공정의 가상계측(VM)과
SPC(칼만·CUSUM·Hotelling T²) 기법을 주가에 적용합니다. 평일 16:30에 자동으로 돌고, 결과를 텔레그램과 구글 드라이브로 받습니다.
자동 주문은 없고, 진입 여부는 사람이 판단합니다.

## 이 도구가 알려주는 것

5년(일부 2006~) 데이터로 사전 등록 기준에 따라 검증한 결과입니다. 자세한 수치와 과정은 `VM_SPC_Core_검증이력.md`에 있습니다.

| 구분 | 내용 | 화면 |
|---|---|---|
| **확실히 작동** | 응축(변동성·고저폭 하위 20%) 뒤 20일 안에 변동성 1.5배 이상 확대: ETF 27% vs 평소 14% (9.14) | 인덱스 ⚡ 배지 |
| | ETF 괴리율 되돌림 +6.7%p (9.5·9.9) | 인덱스 예측 카드 |
| **상태 정보** (검증 불필요) | 바스켓·종목 PER/PBR의 5년 중 위치 (9.17) | 인덱스, 손절·수량 페이지 |
| | 칼만 관리선(±2σ) 기준 "평소 범위", 손절 거리(σ) | 손절·수량 페이지 |
| **검증된 셋업** (표본 밖 재현) | 개별 종목 가짜 이탈 후 복귀 + 이탈일 T² 관리한계 초과: 무작위 대비 거래당 +1.4~2.0%p, 목표 도달 16~18%(9.27) | 손절·수량 페이지 ★ 표시 |
| **장세별 차이** (9.37~9.40, 코스피 60일선 기준) | T² 가짜 이탈은 코스피 상승장에서 −0.11R vs 그 외 +0.32R(새 표본 재현) · 하락장 응축은 20일 뒤 −1.9~−2.9%p(4표본) · 가짜 이탈 전체는 조정장, 돌파 리테스트는 상승장에서만 방향 일치 | 두 페이지 맨 위 장세 한 줄, 찬성·반대 근거 |
| **상승장 돌파 규칙** (9.62·9.68·9.72) | RS 70 이상 종목의 30일 박스 첫 돌파 → 다음 날 시가, 손절 돌파일 종가 − 1ATR, 50일선 이탈까지 보유: 승률 17% · +0.45R (RS 미만 +0.06R, 20일선 청산 +0.27R) | 오늘의 후보 돌파 목록 |
| **가짜 이탈 승률 모형** (9.45·9.48·9.79) | T² · 장세 · 이탈 깊이 · 복귀 캔들 종가 위치 · 이탈일 프로그램 매도 z, 2021-06 이후 AUC 0.61 | 오늘의 후보 승률, ★ 깊이 6%↑ · ☆ 3.5~6% |
| **약한 경향** (판정 미달) | T+1 동료 대비 순위 +2.7%p (9.10), 12개월 섹터 모멘텀 (9.16), 외국인 꾸준한 순매수·순매도 CUSUM (9.21) | 흐리게 참고 표시 |
| | 돌파의 매출 +25% · 4분기 이익 +20%(9.78), 과열 돌파(상단 +4% 초과 마감, R 낮음, 9.67) — 두 기간 같은 방향 | 돌파 목록 칩 |
| **주의 신호** (두 표본에서 확인) | 투매(−5% · 거래량 3배) 뒤 1~3일 더 약함, 개인만 받은 투매는 더 약함 — 되돌림 없음 (9.28) | 가이드 "투매 당일 — 복귀 확인 전 진입 보류" |
| **유망, 판단 불가** (간발 미달, 데이터 13개월) | 애널리스트 목표가 순상향 → 섹터 ETF 20일 초과수익, 순위 상관 +0.08 · 상위−하위 1/3 +2.6%p (9.25) | 인덱스 맨 아래 참고 한 줄(업데이트 날짜 표시, 20일 누적이라 천천히 바뀜) |
| **방향 정보 없음** | 1~20일 ETF·종목 방향, 수급(주체를 나눠도, 9.15·9.22), 프로그램매매, DART 공시 (9.17), T² 없는 가짜 이탈·돌파 리테스트(표본 밖 재현 실패)·거래량·매물대·칼만 관리선 레벨 (9.18~9.19·9.27), 신고가 돌파 (9.20), SPC 이탈 되돌림 (9.14), 섹터 안 대장주 → 후행주 다음 날(같은 날 다 반영, 9.28), 교과서 패턴 4종(깃발·컵앤핸들·역헤드앤숄더·하이 타이트 플래그, 9.29·9.30), 셋업의 손절·익절·조기 청산 변형(모두 현행보다 나쁨, 9.31~9.34), 돌파의 힘·거래량·베이스 길이·이평선 동시 돌파·RS 가속·다른 진입 방아쇠(20일선·칼만·스윙 고점·변동성 돌파)·익절·본전 손절·나눠 사기·확인 후 매수(9.52~9.77), 신용 급감·프로그램 충격(2016~ 재판정, 9.79) | 표시 안 함 |

추세 추종형(돌파 리테스트·돌파)은 코스피 상승장에서만, 되돌림형(가짜 이탈)은 조정·횡보장에서 됩니다(9.37~9.40). 그래서 두 페이지
맨 위에 **지금 장세와 그 장세에 맞는 전략**을 한 줄로 보여 줍니다. 이 도구는 **얼마나 움직일지, 어디서 끊을지, 얼마나 살지**를 돕고, 방향은 사람이 판단합니다. 검증 틀 자체는 알려진 5일 단기 되돌림
(순위 상관 −0.024)을 잡아내는 것으로 확인했습니다(9.28) — 위의 "없음"들은 틀이 둔해서가 아니라 실제로 없는 것입니다.

## 화면

- **`results/index.html`**: 전체 ETF 카드(모바일 대응, 외부 CDN 없음). ETF 자체 예측과 적중률(평소 비율 병기), 구성종목 중 최고 신호,
  응축 배지, 바스켓 PER/PBR, 손절·수량 가이드 링크, 맨 아래 참고 한 줄(섹터 리포트 흐름, 업데이트 날짜 표시). 카드를 누르면 바스켓별 `dashboard_v2.html`(예측모델 · PAIR-SPC · 챔피언-챌린저 탭).
- **`results/scenario/risk_scenarios.html`**: 박스권 손절·수량 페이지. 탭 맨 위 **결론 가이드**: 규칙 충족 여부와 지금 가격 기준
  손절·목표·손익비, 허용 손실 → 수량, **찬성·반대 근거 한 줄씩과 무게 막대·신뢰 문구**(검증 등급 가중: 검증됨 3 · 유망 2 · 약함 1 · 참고 0.5),
  "수량 절반"(응축·관리선 밖·거래량 급증), 투매 당일 진입 보류. 아래에 30일 박스·칼만 관리선·거래량 차트, 두 시나리오(가짜 이탈 후 복귀 /
  돌파 후 리테스트) 카드 — T² 동반 가짜 이탈은 **★ 검증된 셋업**. 기업 정보(PER/PBR, 종목 리포트, 공시)는 하단 카드. 계산 기준가:
  보유 = 평균가, 이미 매수 신호 = 지금 가격, 신호 전 = 조건 충족 시 예상 매수가. 맨 위에 **코스피 장세 한 줄**(상승장 / 횡보·전환 / 하락장,
  할 것 + 근거 숫자)과 **시황 한 줄**(shmstory 최신 시황 제목), 종목 카드에 리포트·뉴스 한 줄.
- **`results/scenario/screen.html`**: **오늘의 후보** — 코스피 시총 상위 200에서 지금 장세의 전략표(`scenario/screen.py` PLAYBOOK)에 맞는
  종목(상승장 = RS 70↑ 돌파, 그 외 = 가짜 이탈). 시점별 목록(오늘 종가로 완료 → 내일 시가 매수 / 최근 5일 안 완료·아직 유효 → 지금 가격 /
  아직 조건 전) 안에서 **승률 높은 순**(위 승률 모형, 평소 승률 병기). 얕은 이탈(3.5% 미만)·거래소 조치 종목은 뺌. 탭은 자바스크립트 없이 동작.

## 빠른 시작

```powershell
pip install -r requirements.txt                                    # Python 3.11+
copy .streamlit\secrets.example.toml .streamlit\secrets.toml       # KIS 앱키·DART 키·텔레그램 입력
copy basket_watchlist.example.json basket_watchlist.json           # 종목·테마·타겟 ETF
copy scenario_targets.example.json scenario_targets.json           # 손절·수량 페이지 대상 종목
python vm_predict/v2_run.py                                        # 수집 → 연산 → 대시보드
python vm_spc/pipeline.py                                          # 챔피언-챌린저 + index.html
```

`basket_watchlist.json`은 종목마다 `theme` 태그를 달고 `themes`에 테마별 타겟 ETF를 지정하면 바스켓이 자동으로 만들어집니다
(`target`이 없는 테마는 건너뜀). `scenario_targets.json`은 `{"stocks": [{"code": "005930"}, ...]}` 형식이고, 바스켓에 없는 종목도
됩니다. 보유 종목은 `{"code": "058470", "avg_price": 72000, "qty": 80}`처럼 평균 매수가(와 수량)를 적으면 규칙 충족 시나리오를
평균가 기준으로 계산하고 정리 기준을 보여 줍니다(미보유는 현재가 기준, 대기 시나리오는 예상 진입가 기준). 두 파일과 `.streamlit/secrets.toml`은 깃허브에 올리지 않습니다(`.gitignore`). 앞의 두 파일은 매일 구글 드라이브에 백업됩니다.

## 매일 자동 실행

작업 스케줄러 "VM-SPC Core 일일 업데이트"가 평일 16:30에 `scripts/run_daily_pipeline_hidden.vbs` → `run_daily_pipeline.ps1`을
실행합니다(PC가 꺼져 있었으면 켜질 때 따라잡음). 약 15분 걸리고, 로그는 `logs/pipeline_*.log`에 30일 보관합니다.

1. `vm_predict/v2_run.py`: 일봉 증분 수집(최근 15일, 수정주가 소급 조정이 감지되면 그 종목만 재수집) → 연산 → 대시보드
2. `pair_spc/run_pair_spc.py`: 대표종목 공적분 SPC
3. `dart_events/collect.py --recent`: DART 최근 2년 재무·공시 + PER/PBR
4. `reports/run.py`: 텔레그램 6개 채널 새 글(로그인 세션이 있으면 API, 없으면 웹 미리보기) → 리포트 표·종목 뉴스 표·시황 한 줄·섹터 리포트 흐름
5. `vm_spc/pipeline.py`: 챔피언-챌린저 → `index.html`
6. `scenario/render_risk.py`: 손절·수량 가이드 페이지(목록 종목 수급·신용 증분 갱신 포함, 인덱스 상단에 링크)
   → `stock_track/collect_program.py --screen`(후보 200종목 프로그램매매 증분) → `scenario/screen.py`: 오늘의 후보(코스피 시총 상위 200)
7. `results/` → `G:\내 드라이브\VM-SPC_Core` 미러링, `basket_watchlist.json`·`scenario_targets.json` 백업
8. `vm_spc/notify_telegram.py`: ① ETF 요약 + `index.html`, ② 관심 종목 전략 요약(장세·시황 한 줄 + 종목당 한 줄) + `risk_scenarios.html`, ③ 오늘의 후보(목록별 승률 순) + `screen.html` 전송(같은 거래일은 생략, 앞 단계 실패 시 경고)
9. 금요일만: 검증용 790종목 프로그램매매 · 신용잔고 · DART 분기 재무 증분(`--validation`, 약 30분)

주의:
- 스케줄러는 PowerShell을 VBScript로 창 없이 띄웁니다. `powershell -WindowStyle Hidden`은 창이 보이고, 클릭하면 프로세스가 멈춥니다.
- `run_daily_pipeline.ps1`은 **UTF-8 BOM 포함**으로 저장해야 합니다. 없으면 PowerShell 5.1이 한글 경로를 깨뜨립니다.
- 텔레그램 확인만: `python vm_spc/notify_telegram.py --dry-run`, 강제 재전송: `--force`.

## 도구별 실행

| 도구 | 명령 | 비고 |
|---|---|---|
| 예측모델 (VM-SPC) | `python vm_predict/v2_run.py [--basket 이름] [--no-fetch]` | 매일 실행 |
| 챔피언-챌린저 | `python vm_spc/pipeline.py [--basket 이름]` | 매일 실행, 9.10 |
| PAIR-SPC | `python pair_spc/run_pair_spc.py` | 매일 실행 |
| DART | `python dart_events/collect.py --recent` · `--validation`(검증용 790종목 분기 재무, 약 3분) · `valuation.py` | 첫 수집은 `--prices`(약 35분), 9.17 · 실적 성장 계산 `growth.py`(9.78) |
| 손절·수량 페이지 | `python scenario/render_risk.py [종목코드 ...] [--no-fetch]` | 매일 실행, 수급 증분 갱신 포함 |
| 오늘의 후보 | `python scenario/screen.py [--no-fetch]` | 매일 실행. 전략표 = 파일 안 `PLAYBOOK`(장세별 규칙 목록) |
| 텔레그램 수집·파싱 | `python reports/run.py` · `tg_login.py`(첫 로그인, 사용자) · `tg_collect.py` · `parse.py` · `news.py` · `market_brief.py` · `market_actions.py` · `sentiment.py` · `render.py` | 매일 실행. 채널: butler_works · ked_epic_ai(리포트), aicorporateanalysisdeepdive(뉴스), shmstory(시황), darthacking(거래소 조치 → 후보 제외), easobi(심리, 표시 안 함) |
| 후보 승률표 | `python research/build_winrate_table.py` | 몇 달에 한 번 갱신(약 15분) |
| 수급·프로그램매매·신용 수집 | `python stock_track/collect_investor_detail.py` · `collect_program.py` · `collect_credit.py` [`--validation [--since=20160101]` · `--screen`] · `collect_market_program.py` | 검증용 790종목만(전 종목은 하지 않음). 2016~ 이력 확장은 약 4.5시간 |
| ETF 순위 / 종목 추적 | `python etf_rank/run.py [--dry]` · `python stock_track/run.py [--no-fetch]` | 수동, 화면에서는 뺌(9.12 · 9.7) |
| 바스켓 후보 스캔 | `python core/v2_etf_scanner.py [--limit 60]` | 결과 `basket_watchlist.scanned.json`은 검토 후 직접 옮김 |
| 5년 이력 확장 | `python vm_predict/backfill_history.py` | 1회 |

검증 스크립트는 `research/`에 모았습니다(결론이 난 일회성 검증, 재현용). 결과는 `results/*_validation.json`.

| 검증 | 명령 (`python research/…`) | 절 |
|---|---|---|
| 정방향 도구 4개 기간 | `forward_horizons.py` | 9.9 |
| 섹터 ETF 장기 이력(2005~) | `fetch_long_history.py` | 9.16 |
| DART 실적·자본 정책 공시 | `validate_dart_events.py` | 9.17 |
| 박스 패턴 (종목 / ETF 재현) | `validate_box.py [--etf]` | 9.18 |
| 거래량 규칙 · 신고가 리테스트 | `validate_volume.py` · `validate_high52.py` | 9.19 · 9.20 |
| 수급·프로그램 CUSUM / 기관 세부 / 격자 / 쌍끌이 | `validate_flow_cusum.py [--detail]` · `flow_cusum_grid.py` · `validate_joint_flow.py` | 9.21~9.23 |
| 애널리스트 리포트 → 섹터 | `validate_report_sector.py` | 9.25 |
| 레벨 정의 비교(박스·매물대·칼만 + T²) / 표본 밖 200종목 | `validate_levels.py [--oos]` | 9.27 |
| 양성 대조군 · 투매 압력 · 섹터 선행·후행 | `validate_mechanisms.py control\|pressure\|leadlag` | 9.28 |
| 강세 깃발 / 교과서 패턴 4종 (`--kosdaq` · `--kospi2`) | `validate_flag.py` · `validate_patterns.py` | 9.29 · 9.30 |
| 셋업 손절 · 익절 · 약한 20일선 청산 | `validate_stops.py` · `validate_targets.py` · `validate_weak_touch.py` | 9.31 · 9.32 · 9.34 |
| 꺾이는 거래 · 안 뜨는 거래 탐색 | `explore_reversal.py` · `explore_nolift.py [--oos]` | 9.33 · 9.35 |
| 코스피 장세 가설 · 반려 전략 장세별 · 응축 장세별 · 추세 추종 돌파 | `validate_market_trend.py` · `validate_regime.py` · `validate_compression_regime.py` · `validate_breakout_trend.py` | 9.37~9.40 |
| 가짜 이탈 보조 신호(심리·신용·프로그램·조합·복귀 기한·차익) | `validate_sentiment.py` · `validate_credit_failure.py` · `validate_program_shock.py` · `validate_signal_combo.py` · `validate_recover_window.py` · `validate_market_arb.py` · `validate_program_flip.py` (`--hist` = 2016~ 재판정) | 9.41~9.51 · 9.79 |
| 돌파 강도·재돌파·유지·손절·포켓 피봇·섹터·종목 지속성 | `validate_breakout_edge.py` · `validate_breakout_r.py` · `validate_rebreakout.py` · `validate_breakout_hold.py` · `validate_breakout_stop_h.py` · `validate_pocket_pivot[_stops].py` · `validate_industry_action.py` | 9.52~9.61 |
| 추세추종 원칙(RS·2단계·신저가·윗꼬리) · RS 변형 · 베이스 · 마진 | `validate_trend_principles.py` · `validate_rs_divergence.py` · `validate_rs_accel.py` · `validate_rs_rising.py` · `validate_base_duration.py` · `validate_breakout_margin.py` | 9.62~9.67 |
| 진입 방아쇠(20일선·칼만·스윙·변동성) · 청산 · 다음날 대응 · 나눠 사기 · 확인 매수 · 이평 동시 돌파 · 거래량 · 실적 | `validate_entry_trigger.py` · `validate_kalman_breakout.py` · `validate_swing_breakout.py` · `validate_vbo.py` · `validate_breakout_exits.py` · `validate_day1_response.py` · `validate_pyramid.py` · `validate_confirm_entry.py` · `validate_confluence.py` · `validate_breakout_volume.py` · `validate_earnings_growth.py` | 9.68~9.78 |

## 폴더

```
core/         공용 인프라: KIS 인증·클라이언트, 설정(Params), 칼만·CUSUM·T²·신호 계산, ETF 스캐너
vm_predict/   예측모델: 수집 → 연산 → 대시보드, 괴리율·응축·신뢰도(v2_premium / v2_compression / v2_reliability)
vm_spc/       챔피언-챌린저(Walk-Forward + 게이트), index.html, 텔레그램
pair_spc/     대표종목 실증 + 공적분 SPC
dart_events/  DART 재무·공시 수집, PER/PBR 상태, 실적 성장(growth.py)
scenario/     손절·수량 가이드 페이지, 오늘의 후보(screen.py), 박스 규칙·T² 판정(box_rules.py — 검증과 화면이 같이 씀)
stock_track/  수급·프로그램매매·신용 수집, 검증용 종목 목록(universe_all.py), 수급 CUSUM 정의(flow_alarm.py), 종목 추적(역방향, 수동)
etf_rank/     섹터 ETF 순위(수동, 화면 제외)
reports/      텔레그램 6개 채널 수집(API, 없으면 웹)·리포트/뉴스 파싱·시황 한 줄·거래소 조치·섹터 리포트 흐름
research/     결론이 난 검증 스크립트(재현용)
scripts/      매일 자동 실행(VBS + PowerShell)
data/ results/ state/ logs/   자동 생성물(.gitignore)
```

모든 스크립트는 프로젝트 루트에서 `python <폴더>/<파일>.py`로 실행합니다(`PYTHONPATH` 설정 불필요).

## 설계상 확정된 판단

- **모든 계산은 인과적**입니다(그날까지의 데이터만 사용). 임계값은 `core/v2_config.Params`에 모여 있습니다.
- **검증은 사전 등록 → 풀링 → 블록 부트스트랩**으로 합니다. 기준을 먼저 적고, 전 바스켓·종목을 합쳐(중복은 1/바스켓 수 가중)
  판정하며, 결과를 보고 기준을 바꾸지 않습니다. 확률은 항상 **평소 비율**과 함께 보여 주고, 약한 효과를 종목별 작은 표본으로 쪼개지 않습니다.
- **`breadth_threshold`는 0.345로 고정**합니다(데이터가 바뀔 때마다 재지정하지 않음). 전체 기간으로 정한 값이라 과거 적중률은 약간
  낙관적일 수 있습니다.
- **데드존 0.2%**: 익일 등락이 이보다 작은 날은 판정하지 않습니다(사전에 정한 규칙).
- **T² 입력은 3개 신호**(가격밴드·거래량·매물대)입니다. 상대강도는 바스켓 평균에서 항등식으로 0이 되어 공분산이 특이해지므로 뺐습니다.
- **챔피언이 없는 바스켓도 빈칸으로 두지 않고** Baseline 값을 흐리게 "게이트 미달·참고"로 표시합니다.
- **한계**: 구성종목 과거 이력(point-in-time)이 없어 현재 구성으로 근사합니다(생존 편향). 박스·공시 검증의 일부 결과는 이 편향의 영향을 받습니다.

장 마감 후 계산한 참고 정보이며 투자 권유가 아닙니다.
