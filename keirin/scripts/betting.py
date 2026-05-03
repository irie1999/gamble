"""
競輪ベッティング戦略モジュール
Kelly基準で賭け金を最適化し、期待値プラスのレースのみ購入する
競輪は7〜9人出走のため、控除率・ランダム基準が競艇と異なる
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


DEDUCTION_RATE = 0.25  # 競輪の控除率（約25%）


@dataclass
class BettingResult:
    race_id: str
    bet_type: str       # "win" / "exacta"
    selections: tuple   # (車番, ...) の組み合わせ
    predicted_prob: float
    implied_prob: float
    edge: float
    odds: float
    bet_amount: int
    expected_value: float


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
    """1/4ケリー基準（保守的）"""
    b = odds - 1.0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    return max(0.0, kelly) * fraction


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
    Kelly基準賭け金計算（100円単位）
    max_bet_ratioとmax_bet_amountの小さい方で上限制限
    """
    fraction = kelly_fraction(prob, odds, kelly_frac)
    amount = bankroll * fraction
    upper = min(bankroll * max_bet_ratio, max_bet_amount)
    amount = min(amount, upper)
    return (int(max(0.0, amount)) // min_bet) * min_bet


def evaluate_bet(
    predicted_prob: float,
    odds: float,
    min_edge: float = 0.05,
) -> tuple[bool, float, float]:
    implied_prob = 1.0 / odds
    edge = predicted_prob - implied_prob
    expected_value = predicted_prob * odds
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
    単勝で期待値プラスの選手を選択してベット
    pred_df: car_no, win_prob 列を持つDF（競輪は car_no）
    """
    bets = []
    for _, row in pred_df.iterrows():
        car_no = int(row["car_no"])
        prob = float(row["win_prob"])
        odds = odds_dict.get(car_no)
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
            selections=(car_no,),
            predicted_prob=prob,
            implied_prob=1.0 / odds,
            edge=edge,
            odds=odds,
            bet_amount=amount,
            expected_value=ev,
        ))

    return bets


def pick_line_leader_bet(
    pred_df: pd.DataFrame,
    odds_dict: dict[int, float],
    bankroll: float,
    min_edge: float = 0.05,
    kelly_frac: float = 0.25,
) -> list[BettingResult]:
    """
    ライン先頭選手のみを対象に単勝ベット（競輪固有戦略）
    ライン先頭が最も有利なため、先頭に絞ることで的中率を上げる
    """
    if "is_line_leader" not in pred_df.columns:
        return pick_best_win_bet(pred_df, odds_dict, bankroll, min_edge, kelly_frac)

    leaders = pred_df[(pred_df["is_line_leader"] == 1) | (pred_df.get("line_no", 0) == 0)]
    return pick_best_win_bet(leaders, odds_dict, bankroll, min_edge, kelly_frac)


def simulate_session(
    races: list[dict],
    initial_bankroll: float = 10000.0,
    min_edge: float = 0.05,
    kelly_frac: float = 0.25,
    bet_type: str = "win",
    line_leader_only: bool = False,
) -> SessionResult:
    """
    バックテスト
    races: [{"pred_df": ..., "odds": ..., "winner": int, ...}, ...]
    line_leader_only: Trueのときラインリーダーのみ対象
    """
    bankroll = initial_bankroll
    session = SessionResult(initial_bankroll=initial_bankroll, final_bankroll=bankroll)

    for race in races:
        pred_df = race["pred_df"]
        odds = race["odds"]
        winner = race["winner"]
        race_id = race.get("race_id", "")

        if line_leader_only:
            bets = pick_line_leader_bet(pred_df, odds, bankroll, min_edge, kelly_frac)
        else:
            bets = pick_best_win_bet(pred_df, odds, bankroll, min_edge, kelly_frac)

        for bet in bets:
            bet.race_id = race_id
            bankroll -= bet.bet_amount

            if bet.bet_type == "win" and bet.selections[0] == winner:
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
    print(f"\n=== ベッティングセッション結果 ===")
    print(f"  初期資金:   {session.initial_bankroll:>10,.0f}円")
    print(f"  最終資金:   {session.final_bankroll:>10,.0f}円")
    print(f"  損益:       {session.profit:>+10,.0f}円")
    print(f"  ROI:        {session.roi:>+9.1%}")
    print(f"  総賭け金:   {session.total_bet:>10,.0f}円")
    print(f"  賭け回数:   {total}回 (的中{session.wins}回 / 外れ{session.losses}回)")
    print(f"  的中率:     {win_rate:.1%}")
    if total > 0:
        print(f"  平均賭け金: {session.total_bet/total:>10,.0f}円")
    if session.bets:
        evs = [b.expected_value for b in session.bets]
        edges = [b.edge for b in session.bets]
        print(f"\n  平均期待値: {np.mean(evs):.3f}倍")
        print(f"  平均エッジ: {np.mean(edges):+.1%}")


if __name__ == "__main__":
    np.random.seed(0)

    # 競輪モックデータ（8人出走）
    # 現実に近い設定：市場（オッズ）は群衆の推測、モデルは真の確率に近い
    N_RIDERS = 8
    races = []
    for i in range(200):
        # 真の勝率（実際に起こる確率）
        true_probs = np.array([0.20, 0.16, 0.14, 0.13, 0.12, 0.10, 0.09, 0.06])
        true_probs /= true_probs.sum()
        winner = np.random.choice(range(1, N_RIDERS + 1), p=true_probs)

        # 市場確率（群衆の見立て）：人気馬を過大評価しがち
        market_probs = true_probs + np.random.normal(0, 0.03, N_RIDERS)
        market_probs = np.clip(market_probs, 0.01, 1)
        market_probs /= market_probs.sum()

        # オッズは市場確率から決まる（控除率25%込み）
        market_odds = {j: round((1 - DEDUCTION_RATE) / market_probs[j - 1], 1)
                       for j in range(1, N_RIDERS + 1)}

        # モデル予測（真の確率に近い・ノイズは市場より小さい）
        pred_probs = true_probs + np.random.normal(0, 0.015, N_RIDERS)
        pred_probs = np.clip(pred_probs, 0.005, 1)
        pred_probs /= pred_probs.sum()

        pred_df = pd.DataFrame({
            "car_no": range(1, N_RIDERS + 1),
            "player_name": [f"選手{j}" for j in range(1, N_RIDERS + 1)],
            "win_prob": pred_probs,
            "is_line_leader": [1, 0, 0, 1, 0, 1, 0, 1],
            "line_no": [1, 1, 1, 2, 2, 3, 3, 0],
        })

        races.append({
            "race_id": f"mock_{i+1:03d}",
            "pred_df": pred_df,
            "odds": market_odds,
            "winner": winner,
        })

    print("--- 全選手対象 ---")
    session = simulate_session(races, initial_bankroll=50000, min_edge=0.05)
    print_session_report(session)

    print("\n--- ライン先頭のみ ---")
    session2 = simulate_session(races, initial_bankroll=50000, min_edge=0.05, line_leader_only=True)
    print_session_report(session2)
