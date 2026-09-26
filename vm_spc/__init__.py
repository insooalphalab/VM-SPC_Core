"""VM-SPC Core: 챔피언-챌린저 의사결정 보조 모델 (VM-SPC_Architecture_Implementation.md PART 2).

Layer 2 피처(칼만잔차·가격Z·CUSUM·T²·공적분Z) → 데드존 레이블 → Walk-Forward(Purge/Embargo)
→ Legacy/Ridge/LightGBM 3-Way 벤치마크 → 3-분기 게이트 판정 → EOD 시나리오 라인.
엔트리포인트: python vm_spc/pipeline.py
"""
