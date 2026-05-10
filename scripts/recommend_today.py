"""当日の全場・全レースから lane1_value/lane1_kelly のベット候補を抽出する。

朝〜昼に実行して、当日賭ける候補レース一覧を出力する。
出走表→確率予測→単勝オッズ取得→Benterブレンド→EVフィルタ→Kellyステーク。

使い方:
    python -m scripts.recommend_today
    python -m scripts.recommend_today --date 2026-05-10 --ev-threshold 1.10
    python -m scripts.recommend_today --workers 8        # 並列度を上げる
    python -m scripts.recommend_today --venues 04 12     # 特定場のみ
    python -m scripts.recommend_today --flat 1000 --kelly-fraction 0

各レースについて、現在オッズで評価する。締切直前にもう一度回すのが理想。
"""
from __future__ import annotations

import argparse
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import local
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import build_dataset
from src.features.feature_engineering import build_features
from src.models.predict import _softmax
from src.scraper.boatrace_scraper import BoatraceScraper, RaceResult
from src.scraper.http_client import HttpClient
from src.scraper.odds_scraper import OddsScraper
from src.strategy.blending import add_blended_probability
from src.strategy.kelly import kelly_stake
from src.utils.config import MODELS_DIR, VENUE_CODES
from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")  # LightGBM の warning 抑制
logging.getLogger("lightgbm").setLevel(logging.ERROR)


# スレッドごとに独立した HttpClient/Scraper を持たせる（HttpClient の throttle が共有されないように）
_TLS = local()


def _scrapers() -> tuple[BoatraceScraper, OddsScraper]:
    if not hasattr(_TLS, "br"):
        client_a = HttpClient(interval_sec=0.3)
        client_b = HttpClient(interval_sec=0.3)
        _TLS.br = BoatraceScraper(client=client_a)
        _TLS.od = OddsScraper(client=client_b)
    return _TLS.br, _TLS.od


def _predict_with_bundle(features_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """predict_win_probability の bundle 受け取り版。joblib.load を毎回しない。"""
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    feat_cols = bundle["feature_columns"]
    X = features_df[feat_cols]
    raw = model.predict_proba(X)[:, 1]
    score = calibrator.transform(raw) if calibrator is not None else raw
    out = features_df.copy()
    out["raw_score"] = score
    out["pred_win_prob"] = (
        out.groupby("race_id")["raw_score"]
        .transform(lambda s: _softmax(s.to_numpy()))
    )
    return out


def _process_race(
    jcd: str, race_no: int, target: date, bundle: dict,
    blend_alpha: float, takeout: float,
) -> Optional[dict]:
    br, od = _scrapers()
    try:
        card = br.fetch_race_card(jcd, race_no, target)
    except Exception as e:
        logger.debug("racelist failed jcd=%s rno=%s: %s", jcd, race_no, e)
        return None
    empty = RaceResult(race_date=card.race_date, venue_code=card.venue_code, race_no=card.race_no)
    df = build_dataset([(card, empty)])
    feats = build_features(df)
    pred = _predict_with_bundle(feats, bundle)

    try:
        wo = od.fetch_win_odds(jcd, race_no, target)
    except Exception as e:
        logger.debug("odds failed jcd=%s rno=%s: %s", jcd, race_no, e)
        return None
    if not wo.odds or len(wo.odds) < 6:
        return None
    pred = pred.copy()
    pred["odds_win"] = pred["lane"].map(wo.odds).astype(float)
    if pred["odds_win"].isna().any():
        return None
    blended = add_blended_probability(pred, alpha=blend_alpha, takeout=takeout)
    pred["blended_win_prob"] = blended["blended_win_prob"].values

    lane1 = pred[pred["lane"] == 1].iloc[0]
    return {
        "venue": jcd,
        "venue_name": VENUE_CODES.get(jcd, "?"),
        "race_no": race_no,
        "lane": 1,
        "racer": lane1.get("name", ""),
        "p_model": float(lane1["pred_win_prob"]),
        "p_blend": float(lane1["blended_win_prob"]),
        "odds_win": float(lane1["odds_win"]),
        "ev": float(lane1["blended_win_prob"]) * float(lane1["odds_win"]),
    }


def _scan_targets(target: date, venues: list[str], workers: int) -> list[tuple[str, int]]:
    """各場の開催レース番号を並列に取得。"""
    out: list[tuple[str, int]] = []
    def task(jcd: str) -> tuple[str, list[int]]:
        br, _ = _scrapers()
        try:
            return jcd, br.fetch_race_index(jcd, target)
        except Exception as e:
            logger.warning("raceindex failed jcd=%s: %s", jcd, e)
            return jcd, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for jcd, rnos in ex.map(task, venues):
            for r in rnos:
                out.append((jcd, r))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD（デフォルト今日）")
    p.add_argument("--venues", nargs="*", default=None, help="対象場コード。省略時は全24場")
    p.add_argument("--exclude-venues", nargs="*", default=["24"],
                   help="除外場コード（デフォルト 24=大村）")
    p.add_argument("--ev-threshold", type=float, default=1.05)
    p.add_argument("--max-odds", type=float, default=None,
                   help="このオッズ超の1号艇は除外（高オッズ1号艇=構造的に弱いレース対策）")
    p.add_argument("--blend-alpha", type=float, default=0.7)
    p.add_argument("--takeout", type=float, default=0.25)
    p.add_argument("--bankroll", type=float, default=100_000.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25,
                   help="0 にすると --flat のフラットベットになる")
    p.add_argument("--flat", type=float, default=1_000.0,
                   help="kelly_fraction=0 の時のステーク額")
    p.add_argument("--workers", type=int, default=6, help="並列度（HTTP同時接続数）")
    p.add_argument("--out", default=None, help="CSV 出力先（任意）")
    args = p.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    all_codes = list(VENUE_CODES.keys())
    venues = [v for v in (args.venues or all_codes) if v not in set(args.exclude_venues)]
    print(f"対象日={target} 対象場={','.join(venues)} workers={args.workers}")

    bundle = joblib.load(MODELS_DIR / "lgb_win_model.joblib")

    # ① 各場の開催レース番号
    targets = _scan_targets(target, venues, args.workers)
    print(f"開催レース総数: {len(targets)} → 各レースで racelist+odds の2リクエスト")
    if not targets:
        print("(本日は対象レースなし)")
        return

    # ② レース毎に並列処理
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_process_race, jcd, rno, target, bundle,
                      args.blend_alpha, args.takeout): (jcd, rno)
            for jcd, rno in targets
        }
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if done % 20 == 0 or done == len(futures):
                print(f"  進捗 {done}/{len(futures)}", flush=True)
            if res is None or res["ev"] <= args.ev_threshold:
                continue
            if args.max_odds is not None and res["odds_win"] > args.max_odds:
                continue
            if args.kelly_fraction > 0:
                stake = kelly_stake(
                    bankroll=args.bankroll,
                    p=res["p_blend"], odds=res["odds_win"],
                    fraction=args.kelly_fraction,
                )
            else:
                stake = args.flat
            if stake <= 0:
                continue
            res["stake_yen"] = int(stake)
            rows.append(res)

    if not rows:
        print(f"\n(EV>{args.ev_threshold} を満たすレースなし)")
        return

    out = pd.DataFrame(rows).sort_values("ev", ascending=False)
    print(f"\n=== {target} 推奨ベット ({len(out)}件) ===")
    print(out.to_string(index=False, formatters={
        "p_model": "{:.3f}".format, "p_blend": "{:.3f}".format,
        "odds_win": "{:.2f}".format, "ev": "{:.3f}".format,
    }))
    print(f"\n合計ステーク: ¥{out['stake_yen'].sum():,}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
