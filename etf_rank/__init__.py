"""ETF 순위 점수판(섹터 순환) — ETF 단위 SPC 상태로 "향후 k일 동안 어느 ETF가 지수를 이길 확률이 높은가"를
점수로 매기고, ETF 간 순위 상관으로 검증한다. 설계는 VM_SPC_Core_검증이력.md 9.12절.

  model.py   피처·라벨·Walk-Forward·순위 상관 검증
  run.py     진입점 (results/etf_rank/etf_rank.json)
  render.py  인덱스 랭킹 카드·ETF 상세 한 줄 HTML
"""
