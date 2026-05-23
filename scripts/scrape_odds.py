"""指定期間の単勝オッズをスクレイプし CSV/parquet で保存する。

使い方:
    # 単純に期間指定
    python -m scripts.scrape_odds --from 2024-08-15 --to 2024-08-31

    # LZHで取得済みのレース一覧から (date, venue, race_no) を絞ってスクレイプ
    # （休場日や非開催場のリクエストを完全にスキップできるため最大3倍速い）
    python -m scripts.scrape_odds --from 2024-04-01 --to 2024-09-30 \
        --races-from data/raw/races.parquet

    # 中断時の再開（既に取得済みのrace_idはスキップ）
    python -m scripts.scrape_odds --from ... --to ... --resume

    # 並列度を上げて高速化（4並列で約4倍速）
    python -m scripts.scrape_odds --from ... --to ... --workers 4

注意:
    - boatrace.jp HTML から取得するため遅い。
    - 過去レースは「締切時オッズ」が表示される。
    - レート制限を守って節度ある利用を。並列度を上げすぎるとブロックされる。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock, local
from typing import Optional

import pandas as pd

from src.scraper.odds_scraper import OddsScraper
from src.utils.config import RAW_DIR, VENUE_CODES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _race_id(d_str: str, jcd: str, rno: int) -> str:
    return f"{d_str}-{jcd}-{int(rno):02d}"


def _parse_race_id(rid: str) -> Optional[tuple[date, str, int]]:
    parts = rid.split("-")
    if len(parts) != 3:
        return None
    d_str, jcd, rno = parts
    try:
        d = date(int(d_str[:4]), int(d_str[4:6]), int(d_str[6:8]))
        return d, jcd, int(rno)
    except (ValueError, IndexError):
        return None


def _read_table(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _build_target_set(
    args: argparse.Namespace,
) -> list[tuple[date, str, int]]:
    """スクレイプ対象 (date, venue, race_no) のタプル列を構築。

    races_from が指定されていればそこから抽出（実開催のみ）。
    無ければ期間×場×R の総当たり。
    """
    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_to)
    venues = set(args.venues) if args.venues else set(VENUE_CODES.keys())
    races = set(args.races)

    if args.races_from:
        df = _read_table(Path(args.races_from))
        targets: list[tuple[date, str, int]] = []
        seen: set[str] = set()
        for rid in df["race_id"].dropna().unique():
            parsed = _parse_race_id(str(rid))
            if not parsed:
                continue
            d, jcd, rno = parsed
            if jcd not in venues or rno not in races:
                continue
            if not (start <= d <= end):
                continue
            if rid in seen:
                continue
            seen.add(rid)
            targets.append((d, jcd, rno))
        targets.sort(key=lambda t: (t[0], t[1], t[2]))
        logger.info("races_from=%s から実開催 %d レース抽出", args.races_from, len(targets))
        return targets

    # 総当たり
    targets = []
    for d in daterange(start, end):
        for jcd in sorted(venues):
            for rno in sorted(races):
                targets.append((d, jcd, rno))
    return targets


def _filter_past_deadline(
    targets: list[tuple[date, str, int]],
    schedule_path: Path,
    now: object = None,
) -> list[tuple[date, str, int]]:
    """schedule CSV を元に「現在時刻より過去の締切」を持つレースを除外。

    schedule が無い・パース失敗のレースは安全側で残す（再取得対象にする）。
    schedule に該当 race_id が無いレースも残す。
    """
    from datetime import datetime, time
    if not schedule_path.exists():
        logger.debug("schedule not found: %s — フィルタ無し", schedule_path)
        return targets
    try:
        sched_df = pd.read_csv(schedule_path)
    except Exception as e:
        logger.warning("schedule 読込失敗: %s — フィルタ無し", e)
        return targets
    sched_map = dict(zip(
        sched_df["race_id"].astype(str),
        sched_df["deadline_time"].astype(str),
    ))
    cur_now = now if isinstance(now, datetime) else datetime.now()

    kept: list[tuple[date, str, int]] = []
    for (d, jcd, rno) in targets:
        rid = _race_id(d.strftime("%Y%m%d"), jcd, rno)
        deadline_str = sched_map.get(rid, "")
        if not deadline_str or ":" not in deadline_str:
            kept.append((d, jcd, rno))  # 不明 → 残す
            continue
        try:
            hh, mm = deadline_str.split(":")[:2]
            deadline = datetime.combine(d, time(int(hh), int(mm)))
        except (ValueError, AttributeError):
            kept.append((d, jcd, rno))  # パース失敗 → 残す
            continue
        if deadline > cur_now:
            kept.append((d, jcd, rno))
        # 締切過ぎたものは除外
    return kept


def _load_existing_race_ids(out_path: Path) -> set[str]:
    """既存 CSV から「有効なオッズが少なくとも1件ある race_id」を返す。

    全レーンが無効値（0 や NaN）の race_id は「未取得相当」とみなし、
    --resume 時にも再取得対象とする。これにより発売前で 0 を引いてしまった
    レースが永久にスキップされる問題を防ぐ。
    """
    if not out_path.exists():
        return set()
    df = _read_table(out_path)
    if "race_id" not in df.columns or "odds_win" not in df.columns:
        return set()
    df["odds_win"] = pd.to_numeric(df["odds_win"], errors="coerce")
    valid = df[df["odds_win"] >= 1.0]
    return set(valid["race_id"].astype(str).unique())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--venues", nargs="*", default=None, help="場コード (例: 12)")
    parser.add_argument("--races", nargs="*", type=int, default=list(range(1, 13)))
    parser.add_argument(
        "--races-from",
        default=None,
        help="LZH スクレイプ済みの races.parquet/csv を渡すと実開催レースだけ取得",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既に出力ファイルにある race_id はスキップして追記する",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="リクエスト間隔(秒)。サーバー側が遅い時は 3〜5 に上げる",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="N レース毎に部分結果を保存（中断耐性）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列スクレイパー数。2〜6 推奨。1 で従来のシリアル動作",
    )
    parser.add_argument(
        "--skip-past-deadline",
        default=None,
        help="race_schedule.csv のパス。指定すると現在時刻より過去の締切レースを除外。"
             "終了したレースのオッズは固定値なので再取得しても無駄、という最適化用。",
    )
    parser.add_argument("--out", default=str(RAW_DIR / "odds_win.csv"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets = _build_target_set(args)
    skip_ids: set[str] = set()
    if args.resume:
        skip_ids = _load_existing_race_ids(out_path)
        logger.info("resume: 既存 %d レースをスキップ", len(skip_ids))
        targets = [(d, jcd, rno) for (d, jcd, rno) in targets
                   if _race_id(d.strftime("%Y%m%d"), jcd, rno) not in skip_ids]

    # 締切過ぎのレースを除外（オッズは確定済で再取得しても無駄）
    if args.skip_past_deadline:
        before = len(targets)
        targets = _filter_past_deadline(targets, Path(args.skip_past_deadline))
        skipped = before - len(targets)
        if skipped:
            logger.info("skip-past-deadline: 締切過ぎ %d レースをスキップ (残 %d)",
                        skipped, len(targets))

    logger.info("対象レース数: %d (workers=%d)", len(targets), args.workers)
    if not targets:
        logger.info("取得対象なし。終了。")
        return

    from src.scraper.http_client import HttpClient

    # スレッドローカルなスクレイパー（HttpClient のスロットリングをワーカー毎に独立させる）
    _TLS = local()

    def _scraper() -> OddsScraper:
        if not hasattr(_TLS, "od"):
            _TLS.od = OddsScraper(client=HttpClient(interval_sec=args.interval))
        return _TLS.od

    rows: list[dict] = []
    rows_lock = Lock()
    success = 0
    attempts = 0
    counter_lock = Lock()

    def _flush() -> None:
        with rows_lock:
            current = list(rows)
            rows.clear()
        if not current:
            return
        new_df = pd.DataFrame(current)
        if (args.resume and out_path.exists()) or out_path.exists():
            old = _read_table(out_path)
            new_df = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
                subset=["race_id", "lane"], keep="last"
            )
        if out_path.suffix == ".parquet":
            new_df.to_parquet(out_path, index=False)
        else:
            new_df.to_csv(out_path, index=False)

    def _fetch_one(triple: tuple[date, str, int]) -> tuple[Optional[str], list[dict]]:
        d, jcd, rno = triple
        od = _scraper()
        try:
            wo = od.fetch_win_odds(jcd, rno, d)
        except Exception as e:
            logger.warning("odds失敗 jcd=%s rno=%s d=%s err=%s", jcd, rno, d, e)
            return None, []
        if not wo.odds:
            return None, []
        rid = _race_id(d.strftime("%Y%m%d"), jcd, rno)
        # 1.0 未満（=0 含む）は無効値として CSV に保存しない。
        # （発売前/中止のページが 0.0 を返した場合の保険）
        rows = [{"race_id": rid, "lane": lane, "odds_win": odds}
                for lane, odds in wo.odds.items() if odds >= 1.0]
        if not rows:
            return None, []
        return rid, rows

    try:
        if args.workers <= 1:
            # 従来のシリアル動作
            for triple in targets:
                attempts += 1
                rid, new_rows = _fetch_one(triple)
                if rid is not None:
                    success += 1
                    with rows_lock:
                        rows.extend(new_rows)
                if attempts % 50 == 0:
                    logger.info("progress: %d/%d (success=%d)", attempts, len(targets), success)
                if attempts % args.checkpoint_every == 0:
                    _flush()
                    logger.info("checkpoint saved: %s", out_path)
        else:
            # 並列実行: スレッドローカル HttpClient で interval_sec をワーカー毎に独立。
            # 各ワーカーが interval_sec 間隔でリクエストするので、合算 RPS は workers/interval_sec。
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_fetch_one, t): t for t in targets}
                for fut in as_completed(futures):
                    with counter_lock:
                        attempts += 1
                        cur = attempts
                    rid, new_rows = fut.result()
                    if rid is not None:
                        with counter_lock:
                            success += 1
                        with rows_lock:
                            rows.extend(new_rows)
                    if cur % 50 == 0:
                        logger.info("progress: %d/%d (success=%d)", cur, len(targets), success)
                    if cur % args.checkpoint_every == 0:
                        _flush()
                        logger.info("checkpoint saved: %s", out_path)
    finally:
        _flush()

    logger.info("done: attempts=%d success=%d → %s", attempts, success, out_path)


if __name__ == "__main__":
    main()
