"""
競艇ベッティング戦略モジュール
単勝 / 複勝 / 2連単 / 2連複 / 3連単 / 3連複 に対応
Kelly基準 + エッジフィルタで賭け金を決定する
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import permutations, combinations

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from prob import top_combinations, BET_TYPE_NAMES, ALL_BET_TYPES

DEDUCTION_RATE = 0.25

# 賭け式ごとのデフォルト設定
BET_CONFIG = {
    #          min_edge  kelly_frac  max_combos  top_n候補数
    "win":      (0.05,   0.25,  1,   6),
    "place":    (0.05,   0.20,  1,   6),
    "exacta":   (0.07,   0.15,  3,  12),
    "quinella": (0.07,   0.15,  3,   6),
    "trifecta": (0.08,   0.10,  6,  20),
    "trio":     (0.07,   0.12,  4,  10),
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
    max_amount: int = 3000,
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
    no_col: str = "boat_no",
) -> list[BettingResult]:
    """
    賭け式に応じたベットを選択して返す
    odds_dict のキー:
      単勝/複勝: {艇番: odds}
      2連単/2連複: {(i,j): odds}
      3連単/3連複: {(i,j,k): odds}
    """
    cfg = BET_CONFIG.get(bet_type, BET_CONFIG["win"])
    min_edge, kelly_frac, max_combos, top_n = cfg

    nos = pred_df[no_col].astype(int).tolist()
    probs = pred_df["win_prob"].values.astype(float)
    probs = probs / probs.sum()

    candidates = top_combinations(probs, bet_type, nos, top_n=top_n)

    bets = []
    for sel, pred_prob in candidates:
        if len(bets) >= max_combos:
            break
        odds = odds_dict.get(sel) or odds_dict.get(sel[0] if len(sel) == 1 else sel)
        if odds is None or odds <= 1.0:
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
    """着順リスト（1着から順）と照合して的中判定"""
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
    no_col: str = "boat_no",
) -> SessionResult:
    """
    バックテスト
    races 各要素:
      pred_df:       win_prob 付きDF
      odds:          {sel_key: odds} の dict（賭け式ごとに自動選択）
      finish_order:  [1着艇番, 2着艇番, 3着艇番, ...]
      race_id:       文字列
    bet_types: 使用する賭け式リスト（None で全式）
    """
    if bet_types is None:
        bet_types = ALL_BET_TYPES

    bankroll = initial_bankroll
    session = SessionResult(initial_bankroll=initial_bankroll, final_bankroll=bankroll)

    for race in races:
        pred_df = race["pred_df"]
        odds_all = race["odds"]          # {bet_type: {sel: odds}}
        finish_order = race["finish_order"]
        race_id = race.get("race_id", "")

        for bt in bet_types:
            odds_dict = odds_all.get(bt, {})
            if not odds_dict:
                continue

            new_bets = pick_bets(pred_df, odds_dict, bankroll, bt, no_col)

            for bet in new_bets:
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

    # 賭け式別集計
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
    """
    モック用オッズ生成（市場確率に少しノイズ）
    戻り値: {bet_type: {sel_tuple: odds}}
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from prob import (
        win_prob, place_prob, exacta_prob, quinella_prob,
        trifecta_prob, trio_prob,
    )
    from itertools import permutations, combinations

    idx = {no: i for i, no in enumerate(nos)}
    n = len(nos)

    # 市場確率（ノイズ付き）
    mp = probs + np.random.normal(0, noise, n)
    mp = np.clip(mp, 0.005, 1)
    mp = mp / mp.sum()

    def to_odds(p: float) -> float:
        raw = (1 - DEDUCTION_RATE) / max(p, 0.005)
        return round(raw * 10) / 10

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
    COURSE_PROBS = np.array([0.553, 0.178, 0.118, 0.082, 0.044, 0.025])

    races = []
    for i in range(200):
        nos = [1, 2, 3, 4, 5, 6]
        true_p = COURSE_PROBS.copy()
        finish = list(np.random.choice(nos, size=6, replace=False, p=true_p))

        pred_p = true_p + np.random.normal(0, 0.015, 6)
        pred_p = np.clip(pred_p, 0.005, 1)
        pred_p /= pred_p.sum()

        pred_df = pd.DataFrame({"boat_no": nos, "win_prob": pred_p,
                                 "player_name": [f"選手{n}" for n in nos]})
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
