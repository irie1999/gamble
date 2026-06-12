"""Kelly基準によるベットサイズ計算。

参考:
  - Kelly, J. L. (1956). "A New Interpretation of Information Rate."
  - Thorp, E. O. (1969). "Optimal Gambling Systems for Favorable Games."
  - MacLean, Thorp & Ziemba (2010). "The Kelly Capital Growth Investment Criterion."

要点:
  - 単純な単勝ベットなら f* = (b*p - q) / b
        b = オッズ - 1（純利益率）, p = 真の勝率, q = 1 - p
  - 推定誤差を踏まえて fractional Kelly（0.1〜0.25倍）を採用するのが実務標準。
"""
from __future__ import annotations

import numpy as np


def kelly_fraction(p: float, odds: float) -> float:
    """単勝1点の最適ベット比率（資金に対する割合）。

    odds: 単勝オッズ（払戻倍率, 例: 3.0 なら 100円→300円）。
    """
    if p <= 0 or odds <= 1:
        return 0.0
    b = odds - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    return float(max(f, 0.0))


def fractional_kelly(p: float, odds: float, fraction: float = 0.25) -> float:
    """0.25 Kelly のように調整された比率を返す。"""
    return kelly_fraction(p, odds) * float(fraction)


def kelly_stake(
    bankroll: float,
    p: float,
    odds: float,
    *,
    fraction: float = 0.25,
    min_stake: float = 100.0,
    max_fraction_of_bankroll: float = 0.05,
) -> float:
    """実ベット額（円）。100円単位を想定して下限を100円に。
    バンクロールに対する上限（5%デフォルト）でも切り詰める。
    """
    f = fractional_kelly(p, odds, fraction=fraction)
    f = min(f, max_fraction_of_bankroll)
    raw = bankroll * f
    if raw < min_stake:
        return 0.0
    # 100円単位に丸め
    return float(np.floor(raw / 100.0) * 100.0)
