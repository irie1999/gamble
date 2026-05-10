"""当日の全場・全レースから lane1_value/lane1_kelly のベット候補を抽出する。

朝〜昼に実行して、当日賭ける候補レース一覧を出力する。
出走表→確率予測→単勝オッズ取得→Benterブレンド→EVフィルタ→Kellyステーク。

使い方:
    # 今日の全場をスキャン、Kelly 0.25倍で推奨ベット
    python -m scripts.recommend_today

    # 日付指定、ev_threshold 1.10 と Kelly 0.5倍、フラット ¥1,000
    python -m scripts.recommend_today --date 2026-05-10 --ev-threshold 1.10 --kelly-fraction 0.5

    # 場フィルタ（大村24除外、デフォルト）
    python -m scripts.recommend_today --exclude-venues 24

    # フラット ¥1,000 で出したい場合
    python -m scripts.recommend_today --flat 1000

各レースについて、現在オッズ（締切時ではない）で評価するため、直前の変動で
EVが下がる可能性がある点に注意。締切直前にもう一度回すのが理想。
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.preprocessor import build_dataset
from src.features.feature_engineering import build_features
from src.models.predict import predict_win_probability
from src.scraper.boatrace_scraper import BoatraceScraper, RaceResult
from src.scraper.odds_scraper import OddsScraper
from src.strategy.blending import add_blended_probability
from src.strategy.kelly import kelly_stake
from src.utils.config import VENUE_CODES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _scan_venues_for_date(
    scraper: BoatraceScraper, target: date, venues: list[str]
) -> list[tuple[str, int]]:
    """その日に開催のある (venue, race_no) 一覧を返す。"""
    out: list[tuple[str, int]] = []
    for jcd in venues:
        try:
            rnos = scraper.fetch_race_index(jcd, target)
        except Exception as e:
            logger.warning("fetch_race_index failed jcd=%s: %s", jcd, e)
            continue
        for r in rnos:
            out.append((jcd, r))
    return out


def _build_pred_for_race(
    scraper: BoatraceScraper, jcd: str, race_no: int, target: date
) -> Optional[pd.DataFrame]:
    try:
        card = scraper.fetch_race_card(jcd, race_no, target)
    except Exception as e:
        logger.warning("racelist failed jcd=%s rno=%s: %s", jcd, race_no, e)
        return None
    empty = RaceResult(race_date=card.race_date, venue_code=card.venue_code, race_no=card.race_no)
    df = build_dataset([(card, empty)])
    feats = build_features(df)
    return predict_win_probability(feats)


def _attach_odds_and_blend(
    pred: pd.DataFrame, odds_scraper: OddsScraper, jcd: str, race_no: int,
    target: date, blend_alpha: float, takeout: float,
) -> Optional[pd.DataFrame]:
    try:
        wo = odds_scraper.fetch_win_odds(jcd, race_no, target)
    except Exception as e:
        logger.warning("odds fetch failed jcd=%s rno=%s: %s", jcd, race_no, e)
        return None
    if not wo.odds or len(wo.odds) < 6:
        return None
    pred = pred.copy()
    pred["odds_win"] = pred["lane"].map(wo.odds).astype(float)
    if pred["odds_win"].isna().any():
        return None
    blended = add_blended_probability(pred, alpha=blend_alpha, takeout=takeout)
    pred["blended_win_prob"] = blended["blended_win_prob"].values
    return pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD（デフォルト今日）")
    p.add_argument("--venues", nargs="*", default=None,
                   help="対象場コード。省略時は全24場")
    p.add_argument("--exclude-venues", nargs="*", default=["24"],
                   help="除外場コード（デフォルト 24=大村）")
    p.add_argument("--ev-threshold", type=float, default=1.05)
    p.add_argument("--blend-alpha", type=float, default=0.7)
    p.add_argument("--takeout", type=float, default=0.25)
    p.add_argument("--bankroll", type=float, default=100_000.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25,
                   help="0 にすると --flat のフラットベットになる")
    p.add_argument("--flat", type=float, default=1_000.0,
                   help="kelly_fraction=0 の時のステーク額")
    p.add_argument("--out", default=None, help="CSV 出力先（任意）")
    args = p.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    all_codes = list(VENUE_CODES.keys())
    venues = [v for v in (args.venues or all_codes) if v not in set(args.exclude_venues)]
    logger.info("対象日=%s 対象場=%s", target, venues)

    scraper = BoatraceScraper()
    odds_scraper = OddsScraper()

    # 全場の開催レース取得
    targets = _scan_venues_for_date(scraper, target, venues)
    logger.info("候補レース総数: %d", len(targets))
    if not targets:
        print("(本日は対象レースなし)")
        return

    rows = []
    for jcd, rno in targets:
        pred = _build_pred_for_race(scraper, jcd, rno, target)
        if pred is None:
            continue
        merged = _attach_odds_and_blend(
            pred, odds_scraper, jcd, rno, target,
            blend_alpha=args.blend_alpha, takeout=args.takeout,
        )
        if merged is None:
            continue
        # 1号艇の EV 評価
        lane1 = merged[merged["lane"] == 1].iloc[0]
        ev = float(lane1["blended_win_prob"]) * float(lane1["odds_win"])
        if ev <= args.ev_threshold:
            continue
        if args.kelly_fraction > 0:
            stake = kelly_stake(
                bankroll=args.bankroll,
                p=float(lane1["blended_win_prob"]),
                odds=float(lane1["odds_win"]),
                fraction=args.kelly_fraction,
            )
        else:
            stake = args.flat
        if stake <= 0:
            continue
        rows.append({
            "venue": jcd,
            "venue_name": VENUE_CODES.get(jcd, "?"),
            "race_no": rno,
            "lane": 1,
            "racer": lane1.get("name", ""),
            "p_model": float(lane1["pred_win_prob"]),
            "p_blend": float(lane1["blended_win_prob"]),
            "odds_win": float(lane1["odds_win"]),
            "ev": ev,
            "stake_yen": int(stake),
        })

    if not rows:
        print(f"(EV>{args.ev_threshold} を満たすレースなし)")
        return

    out = pd.DataFrame(rows).sort_values("ev", ascending=False)
    print(f"\n=== {target} 推奨ベット ({len(out)}件) ===")
    print(out.to_string(index=False, formatters={
        "p_model": "{:.3f}".format,
        "p_blend": "{:.3f}".format,
        "odds_win": "{:.2f}".format,
        "ev": "{:.3f}".format,
    }))
    print(f"\n合計ステーク: ¥{out['stake_yen'].sum():,}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
