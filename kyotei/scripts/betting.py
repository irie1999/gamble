"""
ベッティング戦略モジュール
Kelly基準で賭け金を最適化し、期待値プラスのレースのみ購入する
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


DEDUCTION_RATE = 0.25  # 競艇の控除率（約25%）


@dataclass
class BettingResult:
    race_id: str
    bet_type: str       # "win" / "exacta" / "trifecta"
    selections: tuple   # (艇番, ...) の組み合わせ
    predicted_prob: float
    implied_prob: float    # オッズから計算した確率 1/odds
    edge: float            # predicted_prob - implied_prob
    odds: float
    bet_amount: int        # 賭け金（円）
    expected_value: float  # 期待値


@dataclass
class SessionResult:
    initial_bankroll: float
    final_bankroll: float
    bets: list[BettingResult] = field(default_factory=list)
    wins: int = 0
    losses: int = 0

    @property
    def roi(self) -> float:
        total_bet = sum(b.bet_amount for b in self.bets)
        return (self.final_bankroll - self.initial_bankroll) / total_bet if total_bet > 0 else 0.0

    @property
    def total_bet(self) -> int:
        return sum(b.bet_amount for b in self.bets)

    @property
    def profit(self) -> float:
        return self.final_bankroll - self.initial_bankroll


def kelly_fraction(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Kelly基準の賭け割合を計算
    fraction=0.25 はハーフケリーより保守的な1/4ケリー（実践推奨）
    b = オッズ（1円賭けたときの純利益）
    """
    b = odds - 1.0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    kelly = max(0.0, kelly)  # マイナスなら賭けない
    return kelly * fraction


def calc_bet_amount(
    bankroll: float,
    prob: float,
    odds: float,
    min_bet: int = 100,
    max_bet_ratio: float = 0.05,
    max_bet_amount: int = 3000,
    kelly_frac: float = 0.25,
) -> int:
    """
    bankrollに対するケリー基準での賭け金を計算（100円単位）
    max_bet_ratioとmax_bet_amountの小さい方で上限を制限（破産リスク管理）
    """
    fraction = kelly_fraction(prob, odds, kelly_frac)
    amount = bankroll * fraction
    upper = min(bankroll * max_bet_ratio, max_bet_amount)
    amount = min(amount, upper)
    amount = max(0.0, amount)
    rounded = (int(amount) // min_bet) * min_bet
    return rounded


def evaluate_bet(
    predicted_prob: float,
    odds: float,
    min_edge: float = 0.05,
) -> tuple[bool, float, float]:
    """
    賭けるかどうかと期待値を判定
    Returns: (should_bet, edge, expected_value)
    """
    implied_prob = 1.0 / odds
    edge = predicted_prob - implied_prob
    expected_value = predicted_prob * odds  # 1円あたり期待回収額
    should_bet = edge >= min_edge and expected_value > 1.0
    return should_bet, edge, expected_value


def pick_best_win_bet(
    pred_df: pd.DataFrame,
    odds_dict: dict[int, float],
    bankroll: float,
    min_edge: float = 0.05,
    kelly_frac: float = 0.25,
) -> list[BettingResult]:
    """
    単勝で期待値プラスの艇を選択してベット
    pred_df: boat_no, win_prob 列を持つDF
    odds_dict: {艇番: オッズ}
    """
    bets = []
    for _, row in pred_df.iterrows():
        boat_no = int(row["boat_no"])
        prob = float(row["win_prob"])
        odds = odds_dict.get(boat_no)
        if odds is None or odds <= 1.0:
            continue

        should_bet, edge, ev = evaluate_bet(prob, odds, min_edge)
        if not should_bet:
            continue

        amount = calc_bet_amount(bankroll, prob, odds, kelly_frac=kelly_frac, max_bet_amount=3000)
        if amount < 100:
            continue

        bets.append(BettingResult(
            race_id="",
            bet_type="win",
            selections=(boat_no,),
            predicted_prob=prob,
            implied_prob=1.0 / odds,
            edge=edge,
            odds=odds,
            bet_amount=amount,
            expected_value=ev,
        ))

    return bets


def pick_exacta_bets(
    pred_df: pd.DataFrame,
    exacta_odds: dict[tuple[int, int], float],
    bankroll: float,
    min_edge: float = 0.08,
    kelly_frac: float = 0.15,
    top_n: int = 2,
) -> list[BettingResult]:
    """
    2連単：上位艇の組み合わせで期待値プラスのものを選択
    """
    top_boats = pred_df.nlargest(top_n, "win_prob")["boat_no"].astype(int).tolist()
    bets = []

    for i, first in enumerate(top_boats):
        for second in top_boats:
            if first == second:
                continue
            key = (first, second)
            odds = exacta_odds.get(key)
            if odds is None:
                continue

            # 2連単の確率推定：P(1着=first) × P(2着=second | 1着=first)
            p1 = float(pred_df.loc[pred_df["boat_no"] == first, "win_prob"].values[0])
            p2_cond = float(pred_df.loc[pred_df["boat_no"] == second, "win_prob"].values[0])
            # 条件付き確率の近似
            combined_prob = p1 * (p2_cond / (1.0 - p1 + 1e-9))
            combined_prob = min(combined_prob, 0.99)

            should_bet, edge, ev = evaluate_bet(combined_prob, odds, min_edge)
            if not should_bet:
                continue

            amount = calc_bet_amount(bankroll, combined_prob, odds, kelly_frac=kelly_frac)
            if amount < 100:
                continue

            bets.append(BettingResult(
                race_id="",
                bet_type="exacta",
                selections=key,
                predicted_prob=combined_prob,
                implied_prob=1.0 / odds,
                edge=edge,
                odds=odds,
                bet_amount=amount,
                expected_value=ev,
            ))

    return bets


def simulate_session(
    races: list[dict],
    initial_bankroll: float = 10000.0,
    min_edge: float = 0.05,
    kelly_frac: float = 0.25,
    bet_type: str = "win",
) -> SessionResult:
    """
    過去レースデータでバックテスト
    races: [{"pred_df": ..., "odds": ..., "winner": int, ...}, ...]
    """
    bankroll = initial_bankroll
    session = SessionResult(initial_bankroll=initial_bankroll, final_bankroll=bankroll)

    for race in races:
        pred_df = race["pred_df"]
        odds = race["odds"]
        winner = race["winner"]
        race_id = race.get("race_id", "")

        if bet_type == "win":
            bets = pick_best_win_bet(pred_df, odds, bankroll, min_edge, kelly_frac)
        else:
            bets = []

        for bet in bets:
            bet.race_id = race_id
            bankroll -= bet.bet_amount

            if bet.bet_type == "win" and bet.selections[0] == winner:
                payout = bet.bet_amount * bet.odds
                bankroll += payout
                session.wins += 1
            else:
                session.losses += 1

            session.bets.append(bet)

    session.final_bankroll = bankroll
    return session


def print_session_report(session: SessionResult) -> None:
    total = session.wins + session.losses
    win_rate = session.wins / total if total > 0 else 0
    print(f"\n=== ベッティングセッション結果 ===")
    print(f"  初期資金:   {session.initial_bankroll:>10,.0f}円")
    print(f"  最終資金:   {session.final_bankroll:>10,.0f}円")
    print(f"  損益:       {session.profit:>+10,.0f}円")
    print(f"  ROI:        {session.roi:>+9.1%}")
    print(f"  総賭け金:   {session.total_bet:>10,.0f}円")
    print(f"  賭け回数:   {total}回 (的中{session.wins}回 / 外れ{session.losses}回)")
    print(f"  的中率:     {win_rate:.1%}")
    print(f"  平均賭け金: {session.total_bet/total:>10,.0f}円" if total > 0 else "")

    if session.bets:
        evs = [b.expected_value for b in session.bets]
        edges = [b.edge for b in session.bets]
        print(f"\n  平均期待値: {np.mean(evs):.3f}倍")
        print(f"  平均エッジ: {np.mean(edges):+.1%}")


if __name__ == "__main__":
    np.random.seed(0)

    # モックデータでバックテスト
    races = []
    for i in range(100):
        winner = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.553, 0.178, 0.118, 0.082, 0.044, 0.025])
        probs = np.array([0.55, 0.18, 0.12, 0.08, 0.04, 0.03])
        probs += np.random.normal(0, 0.02, 6)
        probs = np.clip(probs, 0.01, 1)
        probs /= probs.sum()

        pred_df = pd.DataFrame({
            "boat_no": [1, 2, 3, 4, 5, 6],
            "player_name": [f"選手{j}" for j in range(1, 7)],
            "win_prob": probs,
        })

        # オッズ（控除率25%を反映した逆数より少し低いオッズ）
        true_probs = np.array([0.553, 0.178, 0.118, 0.082, 0.044, 0.025])
        raw_odds = 1.0 / true_probs
        fair_odds = raw_odds * (1 - DEDUCTION_RATE)
        odds_dict = {i + 1: round(fair_odds[i], 1) for i in range(6)}

        races.append({
            "race_id": f"mock_{i+1:03d}",
            "pred_df": pred_df,
            "odds": odds_dict,
            "winner": winner,
        })

    session = simulate_session(races, initial_bankroll=50000, min_edge=0.05, kelly_frac=0.25)
    print_session_report(session)

    if session.bets:
        print("\n最初の5ベット:")
        for b in session.bets[:5]:
            print(f"  {b.race_id} 艇{b.selections[0]}番 "
                  f"予測P={b.predicted_prob:.3f} オッズ={b.odds:.1f} "
                  f"エッジ={b.edge:+.3f} 賭け={b.bet_amount:,}円 EV={b.expected_value:.3f}")
