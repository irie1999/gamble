"""バックテストエンジン。

戦略:
  1. (option) 学習済みモデルで各艇の1着確率 p_model を推定
  2. (option) 公衆オッズから p_market を計算し、Benter式でブレンド -> p_final
  3. EV = p_final * odds が閾値以上のベットを抽出
  4. fractional Kelly でベットサイズを決定
  5. 結果（払戻DataFrame）と突合して PnL を計算

戦略選択:
  - "flat":          EV閾値超過の艇に固定額ベット
  - "kelly":         fractional Kelly でサイズ決定（推奨）
  - "always_top1":   常に1コース単勝（ベンチマーク）
  - "model_top1":    モデル予測1位に固定額ベット（オッズ無視ベンチマーク）

参考文献:
  - Benter (1994), Hausch-Ziemba-Rubinstein (1981), Kelly (1956)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.strategy.blending import add_blended_probability
from src.strategy.ev import select_value_bets
from src.strategy.kelly import kelly_stake
from src.utils.logger import get_logger

logger = get_logger(__name__)


Strategy = Literal["flat", "kelly", "always_top1", "model_top1", "lane1_value", "lane1_kelly"]


@dataclass
class BacktestConfig:
    strategy: Strategy = "kelly"
    initial_bankroll: float = 100_000.0
    flat_stake: float = 1_000.0
    kelly_fraction: float = 0.25
    ev_threshold: float = 1.05
    min_prob: float = 0.05
    blend_alpha: float = 0.7        # Benter ブレンド比（モデル側）
    takeout: float = 0.25           # 単勝の控除率
    bet_type: str = "win"           # 当面は単勝のみ
    # 1コース勝率が低い場（例: 大村24）を除外する。lane1_value/lane1_kelly でのみ効く。
    excluded_venues: tuple[str, ...] = ()
    # 1号艇のオッズがこの値超だと「構造的に1号艇が弱いレース」とみなして除外。None で無効。
    max_odds: Optional[float] = None


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    summary: dict
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _winner_lane_from_payouts(payouts_df: pd.DataFrame) -> pd.DataFrame:
    """単勝の払戻から、レース毎の勝者lane と payout(100円ベース) を取得。"""
    win_rows = payouts_df[payouts_df["bet_type"] == "win"].copy()
    win_rows["winner_lane"] = pd.to_numeric(win_rows["combo"], errors="coerce")
    win_rows["payout_yen"] = pd.to_numeric(win_rows["payout_yen"], errors="coerce")
    return (
        win_rows.dropna(subset=["winner_lane"])
        .assign(winner_lane=lambda d: d["winner_lane"].astype(int))
        .groupby("race_id")
        .agg(winner_lane=("winner_lane", "first"), win_payout_yen=("payout_yen", "first"))
        .reset_index()
    )


def _build_pred_with_odds(
    pred_df: pd.DataFrame,
    odds_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """予測DFに実オッズを結合。オッズが渡されない/不足する行は NaN のまま。

    EV系戦略は実オッズが揃っているレースだけで動作する（補完はしない）。
    """
    if odds_df is None or odds_df.empty:
        out = pred_df.copy()
        out["odds_win"] = np.nan
        return out
    out = pred_df.merge(
        odds_df[["race_id", "lane", "odds_win"]],
        on=["race_id", "lane"], how="left",
    )
    return out


def run_backtest(
    pred_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    *,
    odds_df: Optional[pd.DataFrame] = None,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """予測DataFrame×払戻DataFrameでバックテスト。

    pred_df: race_id, lane, pred_win_prob (+ optional race_date)
    payouts_df: race_id, bet_type, combo, payout_yen
    odds_df:  race_id, lane, odds_win  (実オッズ。EV系戦略では必須)

    EV系（kelly/flat）は実オッズが揃ったレースだけを対象にする。
    オッズが無い場合はそのレースをスキップ（人工オッズで埋めない）。

    戻り値: bets（個別ベット内訳）、summary（指標）、equity_curve。
    """
    # --- 確率の準備 ---
    pred = _build_pred_with_odds(pred_df, odds_df)

    # 市場ブレンド：odds が全艇分そろっているレースのみで適用
    full_odds = pred.groupby("race_id")["odds_win"].transform(lambda s: s.notna().all())
    pred["blended_win_prob"] = pred["pred_win_prob"]
    if full_odds.any():
        sub = pred[full_odds].copy()
        sub = add_blended_probability(sub, alpha=config.blend_alpha, takeout=config.takeout)
        pred.loc[full_odds, "blended_win_prob"] = sub["blended_win_prob"].values

    # --- 戦略ごとのベット選択 ---
    if config.strategy == "always_top1":
        bets = pred[pred["lane"] == 1].copy()
        bets["stake"] = config.flat_stake
    elif config.strategy == "model_top1":
        bets = (
            pred.sort_values(["race_id", "pred_win_prob"], ascending=[True, False])
                .groupby("race_id", group_keys=False)
                .head(1)
                .copy()
        )
        bets["stake"] = config.flat_stake
    elif config.strategy in ("lane1_value", "lane1_kelly"):
        # 1号艇限定: 実オッズと推定確率から EV>閾値 のレースのみベット
        # 過小評価された1号艇本命を狙う（穴狙いではなく本命の妙味）
        eligible = pred[full_odds & (pred["lane"] == 1)].copy()
        # 場フィルタ: 1コース勝率が構造的に低い場を除外
        if config.excluded_venues:
            venue_codes = eligible["race_id"].astype(str).str.split("-").str[1]
            eligible = eligible[~venue_codes.isin(config.excluded_venues)]
            logger.info("excluded_venues=%s 適用後 %d候補", config.excluded_venues, len(eligible))
        if config.max_odds is not None:
            before = len(eligible)
            eligible = eligible[eligible["odds_win"] <= config.max_odds]
            logger.info("max_odds=%s 適用後 %d候補（%d→）", config.max_odds, len(eligible), before)
        eligible["ev"] = eligible["blended_win_prob"] * eligible["odds_win"]
        candidates = eligible[eligible["ev"] > config.ev_threshold].copy()
        if config.strategy == "lane1_kelly":
            stakes = []
            for _, r in candidates.iterrows():
                stakes.append(kelly_stake(
                    bankroll=config.initial_bankroll,
                    p=float(r["blended_win_prob"]),
                    odds=float(r["odds_win"]),
                    fraction=config.kelly_fraction,
                ))
            candidates["stake"] = stakes
            candidates = candidates[candidates["stake"] > 0]
        else:
            candidates["stake"] = config.flat_stake
        bets = candidates
    else:
        # EV系戦略は実オッズが揃ったレースのみ対象
        eligible = pred[full_odds].copy()
        if eligible.empty:
            logger.warning(
                "EV系戦略 '%s' を実行するための実オッズが1件もありません。"
                " --odds で odds_win.csv を渡してください。",
                config.strategy,
            )
        candidates = select_value_bets(
            eligible,
            prob_col="blended_win_prob",
            odds_col="odds_win",
            ev_threshold=config.ev_threshold,
            min_prob=config.min_prob,
            top_k_per_race=1,
        )
        if config.strategy == "flat":
            candidates["stake"] = config.flat_stake
        else:  # kelly
            stakes = []
            bankroll = config.initial_bankroll
            for _, r in candidates.iterrows():
                stakes.append(kelly_stake(
                    bankroll=bankroll,
                    p=float(r["blended_win_prob"]),
                    odds=float(r["odds_win"]),
                    fraction=config.kelly_fraction,
                ))
            candidates["stake"] = stakes
            candidates = candidates[candidates["stake"] > 0]
        bets = candidates

    if bets.empty:
        return BacktestResult(
            bets=bets,
            summary={
                "strategy": config.strategy,
                "n_bets": 0,
                "stake_total": 0.0,
                "return_total": 0.0,
                "pnl": 0.0,
                "roi": 0.0,
                "hit_rate": 0.0,
            },
        )

    # --- 結果突合 ---
    winners = _winner_lane_from_payouts(payouts_df)
    bets = bets.merge(winners, on="race_id", how="left")
    # 未開催 or 結果未公開: winner_lane NaN → race_finished=False（レポートで「未確定」表示）
    bets["race_finished"] = bets["winner_lane"].notna()
    bets["hit"] = (bets["lane"] == bets["winner_lane"]).fillna(False)
    bets["payout_per_100"] = np.where(bets["hit"], bets["win_payout_yen"], 0.0)
    bets["return_yen"] = bets["stake"] * bets["payout_per_100"] / 100.0
    bets["pnl"] = bets["return_yen"] - bets["stake"]

    # 確定済みベットだけで equity curve と集計（未確定ベットは未着地として除外）
    finished = bets[bets["race_finished"]]
    n_pending = int((~bets["race_finished"]).sum())

    sort_cols = [c for c in ["race_date", "race_id"] if c in finished.columns]
    finished_sorted = finished.sort_values(sort_cols).reset_index(drop=True)
    equity = config.initial_bankroll + finished_sorted["pnl"].cumsum()
    equity_curve = pd.Series(equity.values, index=finished_sorted["race_id"])

    stake_total = float(finished["stake"].sum())
    return_total = float(finished["return_yen"].sum())
    pnl = return_total - stake_total
    roi = pnl / stake_total if stake_total > 0 else 0.0

    # 最大ドローダウン
    if len(equity):
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())
    else:
        max_dd = 0.0

    summary = {
        "strategy": config.strategy,
        "n_bets": int(len(bets)),
        "n_pending": n_pending,
        "stake_total": stake_total,
        "return_total": return_total,
        "pnl": float(pnl),
        "roi": float(roi),
        "hit_rate": float(finished["hit"].mean()) if len(finished) else 0.0,
        "max_drawdown": max_dd,
        "ending_bankroll": float(equity.iloc[-1]) if len(equity) else config.initial_bankroll,
    }

    return BacktestResult(bets=bets, summary=summary, equity_curve=equity_curve)
