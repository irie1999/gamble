"""
競輪ベッティング戦略モジュール
単勝 / 複勝 / 2車単 / 2車複 / 3連単 / 3連複 に対応
競輪固有のライン先頭フィルタも選択可能
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from prob import top_combinations, BET_TYPE_NAMES, ALL_BET_TYPES

DEDUCTION_RATE = 0.25

# 賭け式ごとのデフォルト設定: (min_edge, kelly_frac, max_combos, top_n)
BET_CONFIG = {
    "win":      (0.05, 0.25, 1,  9),
    "place":    (0.05, 0.20, 1,  9),
    "exacta":   (0.07, 0.15, 3, 20),
    "quinella": (0.07, 0.15, 3, 10),
    "trifecta": (0.08, 0.10, 6, 30),
    "trio":     (0.07, 0.12, 4, 15),
}

# ---- 戦略プリセット ----
# 各戦略: bet_config / bet_types / line_leader_only / min_odds / max_odds
STRATEGIES: dict[str, dict] = {
    "conservative": {
        "description": "堅実：高エッジ・小Kelly・単勝複勝のみ",
        "bet_types": ["win", "place"],
        "line_leader_only": False,
        "min_odds": 1.0, "max_odds": 9999,
        "bet_config": {
            "win":   (0.10, 0.15, 1, 5),
            "place": (0.10, 0.10, 1, 5),
        },
    },
    "balanced": {
        "description": "バランス：デフォルト設定・全賭け式",
        "bet_types": ALL_BET_TYPES,
        "line_leader_only": False,
        "min_odds": 1.0, "max_odds": 9999,
        "bet_config": BET_CONFIG,
    },
    "aggressive": {
        "description": "積極：低エッジ閾値・大Kelly・全賭け式",
        "bet_types": ALL_BET_TYPES,
        "line_leader_only": False,
        "min_odds": 1.0, "max_odds": 9999,
        "bet_config": {
            "win":      (0.03, 0.35, 2,  9),
            "place":    (0.03, 0.30, 2,  9),
            "exacta":   (0.04, 0.25, 5, 30),
            "quinella": (0.04, 0.25, 5, 20),
            "trifecta": (0.04, 0.20, 8, 50),
            "trio":     (0.04, 0.20, 5, 25),
        },
    },
    "single_only": {
        "description": "単系：単勝・複勝のみ（標準設定）",
        "bet_types": ["win", "place"],
        "line_leader_only": False,
        "min_odds": 1.0, "max_odds": 9999,
        "bet_config": BET_CONFIG,
    },
    "combo_only": {
        "description": "連系：2連複・3連複・2連単・3連単のみ",
        "bet_types": ["exacta", "quinella", "trifecta", "trio"],
        "line_leader_only": False,
        "min_odds": 1.0, "max_odds": 9999,
        "bet_config": BET_CONFIG,
    },
    "favorite": {
        "description": "本命狙い：オッズ5倍以下のみ",
        "bet_types": ALL_BET_TYPES,
        "line_leader_only": False,
        "min_odds": 1.0, "max_odds": 5.0,
        "bet_config": BET_CONFIG,
    },
    "longshot": {
        "description": "穴狙い：オッズ10倍以上のみ",
        "bet_types": ALL_BET_TYPES,
        "line_leader_only": False,
        "min_odds": 10.0, "max_odds": 9999,
        "bet_config": BET_CONFIG,
    },
    "line_leader": {
        "description": "ライン先頭：競輪固有の先頭選手に絞る",
        "bet_types": ["win", "place"],
        "line_leader_only": True,
        "min_odds": 1.0, "max_odds": 9999,
        "bet_config": BET_CONFIG,
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
    odds: float
    bet_amount: int
    expected_value: float
    win_flag: bool = False


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
    max_ratio: float = 0.05,
    max_amount: int = 1000,
    min_bet: int = 100,
) -> int:
    frac = kelly_fraction(prob, odds, kelly_frac)
    amount = bankroll * frac
    upper = min(bankroll * max_ratio, max_amount)
    amount = min(amount, upper)
    return (int(max(0.0, amount)) // min_bet) * min_bet


def pick_bets(
    pred_df: pd.DataFrame,
    odds_dict: dict,
    bankroll: float,
    bet_type: str = "win",
    no_col: str = "car_no",
    line_leader_only: bool = False,
    bet_config: dict | None = None,
    min_odds: float = 1.0,
    max_odds: float = 9999,
) -> list[BettingResult]:
    cfg_source = bet_config if bet_config else BET_CONFIG
    cfg = cfg_source.get(bet_type, BET_CONFIG.get(bet_type, (0.05, 0.15, 3, 10)))
    min_edge, kelly_frac, max_combos, top_n = cfg

    df = pred_df.copy()
    if line_leader_only and bet_type in ("win", "place"):
        if "is_line_leader" in df.columns:
            leaders = df[(df["is_line_leader"] == 1) | (df.get("line_no", 0) == 0)]
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
        odds = odds_dict.get(sel) or odds_dict.get(sel[0] if len(sel) == 1 else sel)
        if odds is None or odds <= 1.0:
            continue
        if not (min_odds <= odds <= max_odds):
            continue

        implied = 1.0 / odds
        edge = pred_prob - implied
        ev = pred_prob * odds

        if edge < min_edge or ev <= 1.0:
            continue

        amount = calc_bet_amount(bankroll, pred_prob, odds, kelly_frac)
        if amount < 100:
            continue

        bets.append(BettingResult(
            race_id="",
            bet_type=bet_type,
            selections=sel,
            predicted_prob=pred_prob,
            implied_prob=implied,
            edge=edge,
            odds=odds,
            bet_amount=amount,
            expected_value=ev,
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
    return False


def simulate_session(
    races: list[dict],
    initial_bankroll: float = 50000.0,
    bet_types: list[str] | None = None,
    line_leader_only: bool = False,
    no_col: str = "car_no",
    strategy: dict | None = None,
) -> SessionResult:
    """
    バックテスト
    races 各要素:
      pred_df:       win_prob 付きDF
      odds:          {bet_type: {sel_tuple: odds}}
      finish_order:  [1着車番, 2着車番, 3着車番, ...]
      race_id:       文字列
    """
    # strategy が指定されていればそちらの設定を優先
    s_bet_types = bet_types
    s_line_leader = line_leader_only
    s_bet_config = None
    s_min_odds = 1.0
    s_max_odds = 9999.0
    if strategy:
        s_bet_types = strategy.get("bet_types", bet_types or ALL_BET_TYPES)
        s_line_leader = strategy.get("line_leader_only", line_leader_only)
        s_bet_config = strategy.get("bet_config")
        s_min_odds = strategy.get("min_odds", 1.0)
        s_max_odds = strategy.get("max_odds", 9999.0)
    if s_bet_types is None:
        s_bet_types = ALL_BET_TYPES

    bankroll = initial_bankroll
    session = SessionResult(initial_bankroll=initial_bankroll, final_bankroll=bankroll)
    fixed_amount = strategy.get("fixed_bet") if strategy else None

    for race in races:
        pred_df = race["pred_df"]
        odds_all = race["odds"]
        finish_order = race["finish_order"]
        race_id = race.get("race_id", "")

        for bt in s_bet_types:
            odds_dict = odds_all.get(bt, {})
            if not odds_dict:
                continue

            new_bets = pick_bets(
                pred_df, odds_dict, bankroll, bt, no_col, s_line_leader,
                bet_config=s_bet_config, min_odds=s_min_odds, max_odds=s_max_odds,
            )

            for bet in new_bets:
                if fixed_amount:
                    bet.bet_amount = int(fixed_amount)
                bet.race_id = race_id
                bankroll -= bet.bet_amount
                bet.win_flag = check_win(bet, finish_order)

                if bet.win_flag:
                    bankroll += bet.bet_amount * bet.odds
                    session.wins += 1
                else:
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
            by_type[bt]["payout"] += b.bet_amount * b.odds

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
