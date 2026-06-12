"""戦略モジュールの単体テスト（ネット不要）"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.backtest import BacktestConfig, run_backtest
from src.strategy.blending import benter_blend, odds_to_implied
from src.strategy.ev import select_value_bets
from src.strategy.kelly import kelly_fraction, fractional_kelly, kelly_stake


def test_kelly_fraction_basic():
    # オッズ3倍, 真の勝率0.5 -> b=2, q=0.5 -> f = (2*0.5 - 0.5)/2 = 0.25
    assert abs(kelly_fraction(0.5, 3.0) - 0.25) < 1e-9
    # 不利なベットは0
    assert kelly_fraction(0.1, 2.0) == 0.0


def test_fractional_kelly_and_stake():
    assert abs(fractional_kelly(0.5, 3.0, fraction=0.25) - 0.0625) < 1e-9
    # 100,000 円 × 6.25% = 6250 → 100円単位で6200
    assert kelly_stake(100_000.0, 0.5, 3.0, fraction=0.25, max_fraction_of_bankroll=1.0) == 6200.0


def test_odds_to_implied_normalized():
    p = odds_to_implied(np.array([2.0, 4.0, 4.0]))
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > p[1] == p[2]


def test_benter_blend_extremes():
    p_model = np.array([0.6, 0.2, 0.2])
    p_market = np.array([0.2, 0.4, 0.4])
    full_model = benter_blend(p_model, p_market, alpha=1.0)
    full_market = benter_blend(p_model, p_market, alpha=0.0)
    np.testing.assert_allclose(full_model, p_model, atol=1e-6)
    np.testing.assert_allclose(full_market, p_market, atol=1e-6)


def test_select_value_bets():
    df = pd.DataFrame({
        "race_id": ["R1"] * 6,
        "lane": [1, 2, 3, 4, 5, 6],
        "blended_win_prob": [0.50, 0.20, 0.10, 0.10, 0.05, 0.05],
        "odds_win": [3.0, 4.0, 8.0, 12.0, 30.0, 60.0],
    })
    bets = select_value_bets(df, ev_threshold=1.05, min_prob=0.05, top_k_per_race=1)
    # min_prob=0.05 は strict > なので p=0.05 の lane5/6 は除外される
    # 残った {1,2,3,4} のEV = {1.5, 0.8, 0.8, 1.2}, 閾値 1.05 を超える {1,4} のうちEV最大は lane1
    assert len(bets) == 1
    assert bets.iloc[0]["lane"] == 1


def test_run_backtest_kelly_minimal():
    pred_df = pd.DataFrame({
        "race_id": ["R1"] * 6 + ["R2"] * 6,
        "race_date": pd.to_datetime(["2024-08-01"] * 6 + ["2024-08-02"] * 6),
        "lane": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        "pred_win_prob": [0.6, 0.1, 0.1, 0.1, 0.05, 0.05,
                          0.4, 0.2, 0.15, 0.1, 0.1, 0.05],
    })
    odds_df = pd.DataFrame({
        "race_id": ["R1"] * 6 + ["R2"] * 6,
        "lane": [1, 2, 3, 4, 5, 6] * 2,
        "odds_win": [2.0, 8.0, 8.0, 8.0, 16.0, 16.0,
                     3.0, 5.0, 6.0, 10.0, 10.0, 20.0],
    })
    payouts = pd.DataFrame([
        {"race_id": "R1", "bet_type": "win", "combo": "1", "payout_yen": 200},
        {"race_id": "R2", "bet_type": "win", "combo": "1", "payout_yen": 300},
    ])
    cfg = BacktestConfig(strategy="kelly", initial_bankroll=100_000.0)
    res = run_backtest(pred_df, payouts, odds_df=odds_df, config=cfg)
    # n_bets と ROI が定義されていることだけ確認（中身はパラメータ依存）
    assert "roi" in res.summary
    assert res.summary["n_bets"] >= 0


def test_run_backtest_always_top1_hits():
    pred_df = pd.DataFrame({
        "race_id": ["R1"] * 6,
        "race_date": pd.to_datetime(["2024-08-01"] * 6),
        "lane": [1, 2, 3, 4, 5, 6],
        "pred_win_prob": [1 / 6] * 6,
    })
    payouts = pd.DataFrame([
        {"race_id": "R1", "bet_type": "win", "combo": "1", "payout_yen": 250},
    ])
    cfg = BacktestConfig(strategy="always_top1", flat_stake=1_000.0)
    res = run_backtest(pred_df, payouts, odds_df=None, config=cfg)
    assert res.summary["n_bets"] == 1
    # 1艇1000円ベット, 払戻250円/100円 → 2500円 → PnL=+1500
    assert abs(res.summary["pnl"] - 1500.0) < 1e-6
    assert res.summary["hit_rate"] == 1.0
