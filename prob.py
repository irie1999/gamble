"""
賭け式別の確率計算モジュール（競艇・競輪共通）

Harville公式を使って単勝確率から多着確率を推定する。
  P(j が2着 | i が1着) = P(j) / (1 - P(i))
  P(k が3着 | i→j) = P(k) / (1 - P(i) - P(j))
"""

from itertools import permutations, combinations
import numpy as np


def harville_p2(probs: np.ndarray, i: int, j: int) -> float:
    """P(i=1着, j=2着)"""
    pi = probs[i]
    pj = probs[j]
    return pi * (pj / (1.0 - pi + 1e-9))


def harville_p3(probs: np.ndarray, i: int, j: int, k: int) -> float:
    """P(i=1着, j=2着, k=3着)"""
    pi = probs[i]
    pj = probs[j]
    pk = probs[k]
    p2 = pj / (1.0 - pi + 1e-9)
    p3 = pk / (1.0 - pi - pj + 1e-9)
    return pi * p2 * p3


def win_prob(probs: np.ndarray, i: int) -> float:
    """単勝：i が1着"""
    return float(probs[i])


def place_prob(probs: np.ndarray, i: int) -> float:
    """複勝：i が3着以内（Harville近似）"""
    n = len(probs)
    p = 0.0
    # i が1着
    p += probs[i]
    # i が2着 = Σ_{j≠i} P(j=1着, i=2着)
    for j in range(n):
        if j != i:
            p += harville_p2(probs, j, i)
    # i が3着 = Σ_{j≠i} Σ_{k≠i,k≠j} P(j=1着, k=2着, i=3着)
    for j in range(n):
        if j == i:
            continue
        for k in range(n):
            if k == i or k == j:
                continue
            p += harville_p3(probs, j, k, i)
    return min(float(p), 1.0)


def exacta_prob(probs: np.ndarray, i: int, j: int) -> float:
    """2連単：i=1着, j=2着"""
    return harville_p2(probs, i, j)


def quinella_prob(probs: np.ndarray, i: int, j: int) -> float:
    """2連複：{i,j} が1・2着（順不同）"""
    return harville_p2(probs, i, j) + harville_p2(probs, j, i)


def trifecta_prob(probs: np.ndarray, i: int, j: int, k: int) -> float:
    """3連単：i=1着, j=2着, k=3着"""
    return harville_p3(probs, i, j, k)


def trio_prob(probs: np.ndarray, i: int, j: int, k: int) -> float:
    """3連複：{i,j,k} が1・2・3着（順不同）"""
    p = 0.0
    for perm in permutations([i, j, k]):
        p += harville_p3(probs, *perm)
    return min(float(p), 1.0)


def wide_prob(probs: np.ndarray, i: int, j: int) -> float:
    """ワイド：{i,j} が3着以内に両方入る"""
    n = len(probs)
    p = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        for perm in permutations([i, j, k]):
            p += harville_p3(probs, *perm)
    return min(float(p), 1.0)


def top_combinations(
    probs: np.ndarray,
    bet_type: str,
    nos: list,
    top_n: int = 3,
) -> list[tuple]:
    """
    bet_type に応じて上位確率の組み合わせを返す
    nos: 艇番 or 車番のリスト（0-indexed ではなく実際の番号）
    戻り値: [(選択tuple, 確率), ...]  確率降順
    """
    idx = {no: i for i, no in enumerate(nos)}
    p = probs  # 0-indexed

    results = []

    if bet_type == "win":
        for no in nos:
            i = idx[no]
            results.append(((no,), win_prob(p, i)))

    elif bet_type == "place":
        for no in nos:
            i = idx[no]
            results.append(((no,), place_prob(p, i)))

    elif bet_type == "exacta":
        for a in nos:
            for b in nos:
                if a == b:
                    continue
                results.append(((a, b), exacta_prob(p, idx[a], idx[b])))

    elif bet_type == "quinella":
        for a, b in combinations(nos, 2):
            results.append(((a, b), quinella_prob(p, idx[a], idx[b])))

    elif bet_type == "trifecta":
        for a, b, c in permutations(nos, 3):
            results.append(((a, b, c), trifecta_prob(p, idx[a], idx[b], idx[c])))

    elif bet_type == "trio":
        for combo in combinations(nos, 3):
            a, b, c = combo
            results.append((combo, trio_prob(p, idx[a], idx[b], idx[c])))

    elif bet_type == "wide":
        for a, b in combinations(nos, 2):
            results.append(((a, b), wide_prob(p, idx[a], idx[b])))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n] if top_n else results


# 賭け式の日本語名
BET_TYPE_NAMES = {
    "win":      "単勝",
    "place":    "複勝",
    "exacta":   "2連単",
    "quinella": "2連複",
    "trifecta": "3連単",
    "trio":     "3連複",
    "wide":     "ワイド",
}

ALL_BET_TYPES = list(BET_TYPE_NAMES.keys())

# keirin.jpで実際に販売される賭け式（競輪標準）
KEIRIN_BET_TYPES = ["trifecta", "trio", "wide"]


if __name__ == "__main__":
    # サンプル：6艇、1号艇が圧倒的に強いケース
    nos = [1, 2, 3, 4, 5, 6]
    raw = np.array([0.55, 0.18, 0.12, 0.08, 0.04, 0.03])
    raw = raw / raw.sum()

    print("入力確率:", {no: f"{raw[i]:.3f}" for i, no in enumerate(nos)})
    print()

    for bt in ALL_BET_TYPES:
        top = top_combinations(raw, bt, nos, top_n=3)
        print(f"【{BET_TYPE_NAMES[bt]}】上位3通り")
        for sel, prob in top:
            print(f"  {sel}  P={prob:.4f} ({prob*100:.2f}%)")
        print()
