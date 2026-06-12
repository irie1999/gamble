"""HTMLからレース結果を取得して races_payouts.parquet にマージする。

公式LZH の K-file は当日夜遅くまで未公開なので、日中に結果を反映したい場合に使う。
races.parquet にあるが races_payouts.parquet に payout が無いレースを対象に
boatrace.jp HTML から結果ページを取得し、勝者と払戻を更新する。

使い方:
    # 当日（今）終わっているレースの結果だけ取得
    python -m scripts.scrape_results_html --date 2026-05-12

    # 並列度を調整
    python -m scripts.scrape_results_html --date 2026-05-12 --workers 4

    # 強制全取得（既存も再取得して上書き）
    python -m scripts.scrape_results_html --date 2026-05-12 --force
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import Lock, local
from typing import Optional

import pandas as pd

from src.data.preprocessor import payouts_to_rows
from src.scraper.boatrace_scraper import BoatraceScraper
from src.scraper.http_client import HttpClient
from src.utils.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TLS = local()


def _scraper() -> BoatraceScraper:
    if not hasattr(_TLS, "br"):
        _TLS.br = BoatraceScraper(client=HttpClient(interval_sec=0.3, timeout_sec=10))
    return _TLS.br


def _fetch_one(jcd: str, race_no: int, target: date,
               max_attempts: int = 3) -> Optional[list[dict]]:
    """指定レースの結果を取得し、payouts行のリストを返す。失敗 or 未終了なら None。"""
    br = _scraper()
    last_err = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            result = br.fetch_race_result(jcd, race_no, target,
                                          max_retries=1, read_timeout=10.0)
        except Exception as e:
            last_err = type(e).__name__
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
            continue
        if not result.rows:
            return None  # 未開催/未終了
        if not result.payouts:
            return None  # 結果はあるが払戻取れず
        return payouts_to_rows(result)
    logger.warning("結果取得失敗 jcd=%s rno=%s after %d attempts (%s)",
                   jcd, race_no, max_attempts, last_err)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="対象日 YYYY-MM-DD")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--races-parquet", default=str(RAW_DIR / "races.parquet"))
    p.add_argument("--payouts-parquet", default=str(RAW_DIR / "races_payouts.parquet"))
    p.add_argument("--force", action="store_true",
                   help="既に payouts に登録済みのレースも再取得して上書き")
    p.add_argument("--only-bets", default=None,
                   help="bets CSV のパス。指定すると、その CSV にある race_id だけ取得"
                        "（ベット候補のみ取得＝高速・成功率高）")
    p.add_argument("--abort-on-failure-rate", type=float, default=0.7,
                   help="失敗率が閾値超になったら早期 abort（サイト障害時の救済）")
    args = p.parse_args()

    target = date.fromisoformat(args.date)
    races_path = Path(args.races_parquet)
    payouts_path = Path(args.payouts_parquet)

    if not races_path.exists():
        logger.error("races.parquet が無い: %s", races_path)
        return

    races = pd.read_parquet(races_path)
    races["d"] = races["race_date"].astype(str).str.replace("-", "").str[:8]
    target_str = target.strftime("%Y%m%d")
    races_today = races[races["d"] == target_str]
    if races_today.empty:
        logger.error("races.parquet に %s のレースがありません", target)
        return

    # 対象 race_id × race_no を抽出
    targets = (
        races_today[["race_id"]].drop_duplicates()
        .assign(jcd=lambda d: d["race_id"].str.split("-").str[1],
                rno=lambda d: d["race_id"].str.split("-").str[2].astype(int))
    )

    # --only-bets: ベット候補の race_id だけに絞る
    if args.only_bets:
        bets_path = Path(args.only_bets)
        if bets_path.exists():
            bets = pd.read_csv(bets_path, usecols=["race_id"])
            wanted = set(bets["race_id"].astype(str).unique())
            before = len(targets)
            targets = targets[targets["race_id"].isin(wanted)]
            logger.info("--only-bets フィルタ: %d → %d レース", before, len(targets))
        else:
            logger.warning("--only-bets で指定された CSV が無い: %s", bets_path)

    # 既存 payouts の race_id を取得
    existing: set[str] = set()
    if payouts_path.exists() and not args.force:
        old_payouts = pd.read_parquet(payouts_path)
        existing = set(old_payouts["race_id"].unique())
        logger.info("既存 payouts: %d レース分", len(existing))

    todo = targets[~targets["race_id"].isin(existing)]
    logger.info("対象 %s: %d レース（既存スキップ %d / 全 %d）",
                target, len(todo), len(targets) - len(todo), len(targets))
    if todo.empty:
        logger.info("取得対象なし。終了")
        return

    new_rows: list[dict] = []
    rows_lock = Lock()
    success = 0
    failed = 0
    counter_lock = Lock()
    aborted = False

    def task(row) -> None:
        nonlocal success, failed, aborted
        if aborted:
            return
        rows = _fetch_one(row.jcd, row.rno, target)
        with counter_lock:
            if rows:
                success += 1
            else:
                failed += 1
            done = success + failed
            # 失敗率による早期 abort（一定数試行した後のみ判定）
            if done >= 10 and failed / done > args.abort_on_failure_rate:
                if not aborted:
                    logger.warning("失敗率 %d/%d 超過。残りの取得を中止します（サイト遅延と判断）",
                                   failed, done)
                    aborted = True
        if rows:
            with rows_lock:
                new_rows.extend(rows)
        if done % 20 == 0:
            logger.info("progress %d/%d (success=%d, failed=%d)",
                        done, len(todo), success, failed)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(task, todo.itertuples()))

    logger.info("完了: success=%d failed=%d (aborted=%s)", success, failed, aborted)

    if not new_rows:
        logger.info("追加 payout なし。書き込みスキップ")
        return

    new_df = pd.DataFrame(new_rows)
    if payouts_path.exists():
        old = pd.read_parquet(payouts_path)
        if args.force:
            # 同 race_id の旧データを削除して新規で置換
            old = old[~old["race_id"].isin(new_df["race_id"].unique())]
        merged = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
            subset=["race_id", "bet_type", "combo"], keep="last"
        )
    else:
        merged = new_df

    payouts_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(payouts_path, index=False)
    logger.info("保存: %s rows=%d (+%d)", payouts_path, len(merged), len(new_df))


if __name__ == "__main__":
    main()
