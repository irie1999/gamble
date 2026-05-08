"""
競輪ベッティング戦略モジュール
3連単 / 3連複 / ワイド に対応（keirin.jpで実際に販売される賭け式）
競輪固有のライン先頭フィルタも選択可能
"""

import sys
from pathlib import Path
from itertools import permutations, combinations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from prob import top_combinations, BET_TYPE_NAMES, ALL_BET_TYPES, KEIRIN_BET_TYPES

DEDUCTION_RATE = 0.25

# 賭け式ごとのデフォルト設定: (min_prob, -, max_combos, top_n)
# min_prob: その組み合わせの予測確率の最低閾値
# keirin.jpで実際に販売される賭け式: trifecta / trio / wide
BET_CONFIG = {
    "trifecta": (0.04, 0, 3, 20),
    "trio":     (0.08, 0, 3, 15),
    "wide":     (0.15, 0, 3, 10),
}

# ---- 戦略プリセット ----
# bet_config の第1要素 = min_prob（組み合わせ予測確率の最低閾値）
# 第3要素 = max_combos（1レースあたりの最大ベット数）
# 第4要素 = top_n（候補として評価する上位N組み合わせ）
STRATEGIES: dict[str, dict] = {
    "wide_only": {
        "description": "ワイドのみ：的中率重視・低リスク",
        "bet_types": ["wide"],
        "line_leader_only": False,
        "bet_config": {"wide": (0.15, 0, 3, 10)},
        "fixed_amount": 100,
    },
    "trio_wide": {
        "description": "3連複＋ワイド：バランス重視",
        "bet_types": ["trio", "wide"],
        "line_leader_only": False,
        "bet_config": BET_CONFIG,
        "fixed_amount": 100,
    },
    "balanced": {
        "description": "全賭け式：3連単・3連複・ワイド（標準）",
        "bet_types": KEIRIN_BET_TYPES,
        "line_leader_only": False,
        "bet_config": BET_CONFIG,
        "fixed_amount": 100,
    },
    "value_hunt": {
        "description": "高確率厳選：予測確率上位のみ",
        "bet_types": KEIRIN_BET_TYPES,
        "line_leader_only": False,
        "bet_config": {
            "trifecta": (0.06, 0, 2, 20),
            "trio":     (0.12, 0, 2, 15),
            "wide":     (0.20, 0, 2, 10),
        },
        "fixed_amount": 100,
    },
    "line_leader": {
        "description": "ライン先頭：競輪固有の先頭選手に絞る",
        "bet_types": KEIRIN_BET_TYPES,
        "line_leader_only": True,
        "bet_config": BET_CONFIG,
        "fixed_amount": 100,
    },
    "trio_only": {
        "description": "3連複のみ：中リスク・中配当",
        "bet_types": ["trio"],
        "line_leader_only": False,
        "bet_config": {"trio": (0.08, 0, 3, 15)},
        "fixed_amount": 100,
    },
    "trifecta_mid": {
        "description": "3連単のみ：予測確率上位3点",
        "bet_types": ["trifecta"],
        "line_leader_only": False,
        "bet_config": {"trifecta": (0.04, 0, 3, 20)},
        "fixed_amount": 100,
    },
    "trifecta_sharp": {
        "description": "3連単1点勝負：モデル確信レースのみ・最高確率1点",
        "bet_types": ["trifecta"],
        "line_leader_only": False,
        "bet_config": {"trifecta": (0.10, 0, 1, 20)},
        "min_top_prob": 0.35,   # 1位予測確率35%以上のレースのみ
        "fixed_amount": 100,
    },
    "trifecta_sharp2": {
        "description": "3連単2点勝負：モデル確信レースのみ",
        "bet_types": ["trifecta"],
        "line_leader_only": False,
        "bet_config": {"trifecta": (0.07, 0, 2, 20)},
        "min_top_prob": 0.30,   # 1位予測確率30%以上のレースのみ
        "fixed_amount": 100,
    },
    "trio_sharp": {
        "description": "3連複厳選：確信レースのみ・的中率重視",
        "bet_types": ["trio"],
        "line_leader_only": False,
        "bet_config": {"trio": (0.10, 0, 2, 15)},
        "min_top_prob": 0.30,   # 1位予測確率30%以上のレースのみ
        "fixed_amount": 100,
    },
    "wide_sharp": {
        "description": "ワイド厳選：確信レースのみ・最高的中率",
        "bet_types": ["wide"],
        "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 2, 10)},
        "min_top_prob": 0.30,
        "fixed_amount": 100,
    },
    # ---- wide_sharp パラメータチューニングバリアント ----
    # 命名規則: ws_{min_top_prob%}_{max_combos}c_{min_combo_prob%}
    "ws_30_1c": {
        "description": "[tune] wide: top30% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.30, "fixed_amount": 100,
    },
    "ws_35_1c": {
        "description": "[tune] wide: top35% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.35, "fixed_amount": 100,
    },
    "ws_40_1c": {
        "description": "[tune] wide: top40% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.40, "fixed_amount": 100,
    },
    "ws_45_1c": {
        "description": "[tune] wide: top45% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.45, "fixed_amount": 100,
    },
    "ws_50_1c": {
        "description": "[tune] wide: top50% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.50, "fixed_amount": 100,
    },
    "ws_35_2c": {
        "description": "[tune] wide: top35% 2点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 2, 10)},
        "min_top_prob": 0.35, "fixed_amount": 100,
    },
    "ws_40_2c": {
        "description": "[tune] wide: top40% 2点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 2, 10)},
        "min_top_prob": 0.40, "fixed_amount": 100,
    },
    "ws_45_2c": {
        "description": "[tune] wide: top45% 2点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 2, 10)},
        "min_top_prob": 0.45, "fixed_amount": 100,
    },
    "ws_35_1c_25": {
        "description": "[tune] wide: top35% 1点 combo25%以上",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.25, 0, 1, 10)},
        "min_top_prob": 0.35, "fixed_amount": 100,
    },
    "ws_40_1c_25": {
        "description": "[tune] wide: top40% 1点 combo25%以上",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.25, 0, 1, 10)},
        "min_top_prob": 0.40, "fixed_amount": 100,
    },
    "ws_45_1c_25": {
        "description": "[tune] wide: top45% 1点 combo25%以上",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.25, 0, 1, 10)},
        "min_top_prob": 0.45, "fixed_amount": 100,
    },
    "ws_50_1c_25": {
        "description": "[tune] wide: top50% 1点 combo25%以上",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.25, 0, 1, 10)},
        "min_top_prob": 0.50, "fixed_amount": 100,
    },
    "ws_55_1c": {
        "description": "[tune] wide: top55% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.55, "fixed_amount": 100,
    },
    "ws_60_1c": {
        "description": "[tune] wide: top60% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.60, "fixed_amount": 100,
    },
    "ws_57_1c": {
        "description": "[tune] wide: top57% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.57, "fixed_amount": 100,
    },
    "ws_58_1c": {
        "description": "[tune] wide: top58% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.58, "fixed_amount": 100,
    },
    "ws_59_1c": {
        "description": "[tune] wide: top59% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.59, "fixed_amount": 100,
    },
    "ws_61_1c": {
        "description": "[tune] wide: top61% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.61, "fixed_amount": 100,
    },
    "ws_62_1c": {
        "description": "[tune] wide: top62% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.62, "fixed_amount": 100,
    },
    "ws_63_1c": {
        "description": "[tune] wide: top63% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.63, "fixed_amount": 100,
    },
    "ws_65_1c": {
        "description": "[tune] wide: top65% 1点",
        "bet_types": ["wide"], "line_leader_only": False,
        "bet_config": {"wide": (0.20, 0, 1, 10)},
        "min_top_prob": 0.65, "fixed_amount": 100,
    },
    "combo_sharp": {
        "description": "3連複＋ワイド厳選：確信レースで的中率と配当を両立",
        "bet_types": ["trio", "wide"],
        "line_leader_only": False,
        "bet_config": {
            "trio": (0.10, 0, 2, 15),
            "wide": (0.20, 0, 2, 10),
        },
        "min_top_prob": 0.30,
        "fixed_amount": 100,
    },
    # ---- ボックス戦略 ----
    # box_n_cars: 予測上位N車を選び、全順列(3連単)or全組合(3連複)を買う
    "trifecta_box3": {
        "description": "3連単ボックス上位3車：6点買い・着順不問",
        "bet_types": ["trifecta"],
        "line_leader_only": False,
        "bet_config": BET_CONFIG,
        "box_n_cars": 3,          # 上位3車 → 3!=6点
        "fixed_amount": 100,
    },
    "trifecta_box3_sharp": {
        "description": "3連単ボックス上位3車（確信レース限定）",
        "bet_types": ["trifecta"],
        "line_leader_only": False,
        "bet_config": BET_CONFIG,
        "box_n_cars": 3,
        "min_top_prob": 0.35,
        "fixed_amount": 100,
    },
    "trifecta_box4": {
        "description": "3連単ボックス上位4車：24点買い・高的中率",
        "bet_types": ["trifecta"],
        "line_leader_only": False,
        "bet_config": BET_CONFIG,
        "box_n_cars": 4,          # 上位4車 → 4*3*2=24点
        "fixed_amount": 100,
    },
    "trio_box4": {
        "description": "3連複ボックス上位4車：4点買い・的中率重視",
        "bet_types": ["trio"],
        "line_leader_only": False,
        "bet_config": BET_CONFIG,
        "box_n_cars": 4,          # 上位4車 → C(4,3)=4点
        "fixed_amount": 100,
    },
    "trio_box4_sharp": {
        "description": "3連複ボックス上位4車（確信レース限定）",
        "bet_types": ["trio"],
        "line_leader_only": False,
        "bet_config": BET_CONFIG,
        "box_n_cars": 4,
        "min_top_prob": 0.35,
        "fixed_amount": 100,
    },
}


@dataclass
class BettingResult:
    race_id: str
    bet_type: str
    selections: tuple
    predicted_prob: float
    implied_prob: float
    edge: float
    odds: float          # 推定事前オッズ（ベット選択用）
    bet_amount: int
    expected_value: float
    win_flag: bool = False
    return_odds: float = 0.0  # 実際の払戻倍率（当選時に設定）
    actual_payout: float = 0.0  # 実際の受取額（loss=0, win=bet*odds, refund=bet）


@dataclass
class SessionResult:
    initial_bankroll: float
    final_bankroll: float
    bets: list[BettingResult] = field(default_factory=list)
    wins: int = 0
    losses: int = 0

    @property
    def roi(self) -> float:
        tb = self.total_bet
        return (self.final_bankroll - self.initial_bankroll) / tb if tb > 0 else 0.0

    @property
    def total_bet(self) -> int:
        return sum(b.bet_amount for b in self.bets)

    @property
    def profit(self) -> float:
        return self.final_bankroll - self.initial_bankroll


def kelly_fraction(prob: float, odds: float, frac: float) -> float:
    b = odds - 1.0
    q = 1.0 - prob
    k = (b * prob - q) / b if b > 0 else 0.0
    return max(0.0, k) * frac


def calc_bet_amount(
    bankroll: float,
    prob: float,
    odds: float,
    kelly_frac: float,
    max_ratio: float = 0.02,
    max_amount: int = 300,
    min_bet: int = 100,
) -> int:
    frac = kelly_fraction(prob, odds, kelly_frac)
    amount = bankroll * frac
    upper = min(bankroll * max_ratio, max_amount)
    amount = min(amount, upper)
    return (int(max(0.0, amount)) // min_bet) * min_bet


def pick_bets_box(
    pred_df: pd.DataFrame,
    bet_type: str = "trifecta",
    n_cars: int = 3,
    no_col: str = "car_no",
    line_leader_only: bool = False,
    fixed_amount: int = 100,
) -> list[BettingResult]:
    """予測上位N車を選び、全順列(3連単)または全組合(3連複)をボックス買い"""
    df = pred_df.copy()
    if line_leader_only and "is_line_leader" in df.columns:
        leaders = df[(df["is_line_leader"] == 1) | (df["line_no"] == 0)]
        if not leaders.empty:
            df = leaders

    df = df.drop_duplicates(subset=no_col, keep="first")
    df = df.sort_values("win_prob", ascending=False)
    top_nos = df[no_col].astype(int).head(n_cars).tolist()

    bets = []
    if bet_type == "trifecta":
        combos = list(permutations(top_nos, 3))
    elif bet_type == "trio":
        combos = list(combinations(top_nos, 3))
    else:
        return []

    all_nos = df[no_col].astype(int).tolist()
    win_probs = df["win_prob"].values.astype(float)
    win_probs = win_probs / win_probs.sum()
    from prob import trifecta_prob, trio_prob as _trio_prob
    import numpy as np
    prob_arr = np.array(win_probs)
    idx_map = {no: i for i, no in enumerate(all_nos)}

    for sel in combos:
        if bet_type == "trifecta":
            try:
                pred_p = trifecta_prob(prob_arr, idx_map[sel[0]], idx_map[sel[1]], idx_map[sel[2]])
            except Exception:
                pred_p = 0.0
        else:
            try:
                pred_p = _trio_prob(prob_arr, idx_map[sel[0]], idx_map[sel[1]], idx_map[sel[2]])
            except Exception:
                pred_p = 0.0
        bets.append(BettingResult(
            race_id="",
            bet_type=bet_type,
            selections=tuple(sel),
            predicted_prob=float(pred_p),
            implied_prob=0.0,
            edge=0.0,
            odds=0.0,
            bet_amount=fixed_amount,
            expected_value=0.0,
        ))
    return bets


def pick_bets(
    pred_df: pd.DataFrame,
    bet_type: str = "trifecta",
    no_col: str = "car_no",
    line_leader_only: bool = False,
    bet_config: dict | None = None,
    fixed_amount: int = 100,
) -> list[BettingResult]:
    """モデル予測確率の上位組み合わせを選択（市場オッズ不要）"""
    cfg_source = bet_config if bet_config else BET_CONFIG
    cfg = cfg_source.get(bet_type, BET_CONFIG.get(bet_type, (0.05, 0.03, 3, 10)))
    min_prob, _, max_combos, top_n = cfg

    df = pred_df.copy()
    if line_leader_only:
        if "is_line_leader" in df.columns:
            leaders = df[(df["is_line_leader"] == 1) | (df["line_no"] == 0)]
            if not leaders.empty:
                df = leaders

    df = df.drop_duplicates(subset=no_col, keep="first")
    nos = df[no_col].astype(int).tolist()
    probs = df["win_prob"].values.astype(float)
    probs = probs / probs.sum()

    candidates = top_combinations(probs, bet_type, nos, top_n=top_n)

    bets = []
    for sel, pred_prob in candidates:
        if len(bets) >= max_combos:
            break
        if pred_prob < min_prob:
            continue

        bets.append(BettingResult(
            race_id="",
            bet_type=bet_type,
            selections=sel,
            predicted_prob=pred_prob,
            implied_prob=0.0,
            edge=0.0,
            odds=0.0,
            bet_amount=fixed_amount,
            expected_value=0.0,
        ))

    return bets


def check_win(bet: BettingResult, finish_order: list[int]) -> bool:
    sel = bet.selections
    bt = bet.bet_type
    if len(finish_order) < 3:
        return False
    top1, top2, top3 = finish_order[0], finish_order[1], finish_order[2]

    if bt == "win":
        return sel[0] == top1
    elif bt == "place":
        return sel[0] in (top1, top2, top3)
    elif bt == "exacta":
        return sel[0] == top1 and sel[1] == top2
    elif bt == "quinella":
        return set(sel) == {top1, top2}
    elif bt == "trifecta":
        return sel[0] == top1 and sel[1] == top2 and sel[2] == top3
    elif bt == "trio":
        return set(sel) == {top1, top2, top3}
    elif bt == "wide":
        top3_set = {top1, top2, top3}
        return sel[0] in top3_set and sel[1] in top3_set
    return False


def simulate_session(
    races: list[dict],
    initial_bankroll: float = 50000.0,
    bet_types: list[str] | None = None,
    line_leader_only: bool = False,
    no_col: str = "car_no",
    strategy: dict | None = None,
) -> SessionResult:
    """バックテスト（市場オッズ不要・実際の払戻額のみ使用）

    races 各要素:
      pred_df:       win_prob 付きDF
      payouts:       {bet_type: {sel_tuple: 払戻倍率}}  ← 実データ
      finish_order:  [1着車番, 2着車番, 3着車番, ...]
      race_id:       文字列
    """
    s_bet_types = strategy.get("bet_types", bet_types or ALL_BET_TYPES) if strategy else (bet_types or ALL_BET_TYPES)
    s_line_leader = strategy.get("line_leader_only", line_leader_only) if strategy else line_leader_only
    s_bet_config = strategy.get("bet_config") if strategy else None
    s_fixed_amount = strategy.get("fixed_amount", 100) if strategy else 100
    s_min_top_prob = strategy.get("min_top_prob", 0.0) if strategy else 0.0
    s_box_n_cars = strategy.get("box_n_cars", 0) if strategy else 0  # 0=通常選択

    bankroll = initial_bankroll
    session = SessionResult(initial_bankroll=initial_bankroll, final_bankroll=bankroll)

    for race in races:
        pred_df = race["pred_df"]
        payouts_all = race.get("payouts", {})
        finish_order = race["finish_order"]
        race_id = race.get("race_id", "")

        # レース単位の信頼度フィルタ
        if s_min_top_prob > 0.0:
            top_prob = pred_df["win_prob"].max() if not pred_df.empty else 0.0
            if top_prob < s_min_top_prob:
                continue

        for bt in s_bet_types:
            if s_box_n_cars > 0:
                new_bets = pick_bets_box(
                    pred_df, bt, n_cars=s_box_n_cars,
                    no_col=no_col, line_leader_only=s_line_leader,
                    fixed_amount=s_fixed_amount,
                )
            else:
                new_bets = pick_bets(
                    pred_df, bt, no_col, s_line_leader,
                    bet_config=s_bet_config,
                    fixed_amount=s_fixed_amount,
                )

            for bet in new_bets:
                bet.race_id = race_id
                bankroll -= bet.bet_amount
                bet.win_flag = check_win(bet, finish_order)

                if bet.win_flag:
                    # 実際の払戻額を使用（当選組み合わせのみ存在）
                    actual = payouts_all.get(bt, {}).get(bet.selections)
                    if actual and actual > 1.0:
                        bet.return_odds = actual
                        bet.actual_payout = bet.bet_amount * actual
                        bankroll += bet.actual_payout
                        session.wins += 1
                    else:
                        # 払戻データなし → 的中としてカウントしない（返金扱い）
                        bet.actual_payout = bet.bet_amount
                        bankroll += bet.bet_amount
                        bet.win_flag = False
                        session.losses += 1
                else:
                    bet.actual_payout = 0.0
                    session.losses += 1

                session.bets.append(bet)

    session.final_bankroll = bankroll
    return session


def print_session_report(session: SessionResult) -> None:
    total = session.wins + session.losses
    win_rate = session.wins / total if total > 0 else 0
    print(f"\n{'='*55}")
    print(f"  ベッティング結果")
    print(f"{'='*55}")
    print(f"  初期資金:   {session.initial_bankroll:>10,.0f}円")
    print(f"  最終資金:   {session.final_bankroll:>10,.0f}円")
    print(f"  損益:       {session.profit:>+10,.0f}円")
    print(f"  ROI:        {session.roi:>+9.1%}")
    print(f"  総賭け金:   {session.total_bet:>10,.0f}円")
    print(f"  賭け回数:   {total}回  (的中{session.wins} / 外れ{session.losses})")
    print(f"  的中率:     {win_rate:.1%}")

    by_type: dict[str, dict] = {}
    for b in session.bets:
        bt = b.bet_type
        if bt not in by_type:
            by_type[bt] = {"bets": 0, "wins": 0, "spent": 0, "payout": 0.0}
        by_type[bt]["bets"] += 1
        by_type[bt]["spent"] += b.bet_amount
        if b.win_flag:
            by_type[bt]["wins"] += 1
            payout_odds = b.return_odds if b.return_odds > 0 else b.odds
            by_type[bt]["payout"] += b.bet_amount * payout_odds

    if by_type:
        print(f"\n  {'賭け式':<8} {'回数':>5} {'的中':>5} {'的中率':>7} {'賭け金':>10} {'回収':>10} {'ROI':>8}")
        print(f"  {'-'*56}")
        for bt, d in by_type.items():
            wr = d['wins'] / d['bets'] if d['bets'] else 0
            roi = (d['payout'] - d['spent']) / d['spent'] if d['spent'] else 0
            name = BET_TYPE_NAMES.get(bt, bt)
            print(f"  {name:<8} {d['bets']:>5} {d['wins']:>5} {wr:>7.1%} "
                  f"{d['spent']:>9,}円 {d['payout']:>9,.0f}円 {roi:>+7.1%}")


def make_mock_odds(
    nos: list[int],
    probs: np.ndarray,
    bet_types: list[str],
    noise: float = 0.03,
) -> dict[str, dict]:
    from prob import (
        win_prob, place_prob, exacta_prob, quinella_prob,
        trifecta_prob, trio_prob,
    )
    from itertools import permutations, combinations

    idx = {no: i for i, no in enumerate(nos)}
    n = len(nos)
    mp = probs + np.random.normal(0, noise, n)
    mp = np.clip(mp, 0.005, 1)
    mp = mp / mp.sum()

    def to_odds(p: float) -> float:
        return round((1 - DEDUCTION_RATE) / max(p, 0.005) * 10) / 10

    result: dict[str, dict] = {}
    for bt in bet_types:
        d: dict = {}
        if bt == "win":
            for no in nos:
                d[(no,)] = to_odds(win_prob(mp, idx[no]))
        elif bt == "place":
            for no in nos:
                d[(no,)] = to_odds(place_prob(mp, idx[no]))
        elif bt == "exacta":
            for a, b in permutations(nos, 2):
                d[(a, b)] = to_odds(exacta_prob(mp, idx[a], idx[b]))
        elif bt == "quinella":
            for a, b in combinations(nos, 2):
                d[(a, b)] = to_odds(quinella_prob(mp, idx[a], idx[b]))
        elif bt == "trifecta":
            for a, b, c in permutations(nos, 3):
                d[(a, b, c)] = to_odds(trifecta_prob(mp, idx[a], idx[b], idx[c]))
        elif bt == "trio":
            for combo in combinations(nos, 3):
                a, b, c = combo
                d[combo] = to_odds(trio_prob(mp, idx[a], idx[b], idx[c]))
        elif bt == "wide":
            # P(a,b 両方が3着以内) = Σ_{k≠a,k≠b} trio_prob({a,b,k})
            for a, b in combinations(nos, 2):
                p = sum(
                    trio_prob(mp, idx[a], idx[b], idx[k])
                    for k in nos if k != a and k != b
                )
                d[(a, b)] = to_odds(p)
        result[bt] = d
    return result


if __name__ == "__main__":
    np.random.seed(42)

    races = []
    for i in range(200):
        n = np.random.choice([7, 8, 9])
        nos = list(range(1, n + 1))
        true_p = np.random.dirichlet(np.ones(n) * 2)
        finish = list(np.random.choice(nos, size=min(n, 3), replace=False, p=true_p))
        while len(finish) < 3:
            finish.append(nos[len(finish)])

        pred_p = true_p + np.random.normal(0, 0.015, n)
        pred_p = np.clip(pred_p, 0.005, 1)
        pred_p /= pred_p.sum()

        pred_df = pd.DataFrame({
            "car_no": nos, "win_prob": pred_p,
            "player_name": [f"選手{j}" for j in nos],
            "is_line_leader": [1 if j % 3 == 1 else 0 for j in nos],
            "line_no": [(j - 1) // 3 + 1 for j in nos],
        })
        odds = make_mock_odds(nos, true_p, ALL_BET_TYPES, noise=0.03)
        races.append({"race_id": f"mock_{i+1:03d}", "pred_df": pred_df,
                       "odds": odds, "finish_order": finish})

    for bet_types in [["win"], ["trifecta"], ALL_BET_TYPES]:
        label = " + ".join(BET_TYPE_NAMES[bt] for bt in bet_types)
        print(f"\n{'━'*55}")
        print(f"  賭け式: {label}")
        print(f"{'━'*55}")
        session = simulate_session(races, initial_bankroll=50000, bet_types=bet_types)
        print_session_report(session)
