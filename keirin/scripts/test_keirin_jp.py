"""
keirin.jp dataplaza から払戻金データを取得できるか検証するスクリプト

URL形式: https://keirin.jp/pc/dfw/dataplaza/guest/raceresult?KCD={KCD}&KBI={YYYYMMDD}&RNO={RNO}

Usage:
  python keirin/scripts/test_keirin_jp.py
  python keirin/scripts/test_keirin_jp.py KCD KBI RNO
  例: python keirin/scripts/test_keirin_jp.py 26 20260504 1
"""

import sys
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://keirin.jp/",
}

BASE_URL = "https://keirin.jp/pc/dfw/dataplaza/guest/raceresult"

# kdreams.jp venue_code → keirin.jp KCD の対応表（要確認）
# keirin.jpのKCDはJKA標準場コード（2桁）
KDREAMS_TO_KCD = {
    # 関東
    "11": "11",  # 前橋
    "12": "12",  # 高崎
    "13": "13",  # 宇都宮
    "14": "14",  # 大宮
    "15": "15",  # 西武園
    "16": "16",  # 京王閣
    "17": "17",  # 立川
    "18": "18",  # 松戸
    "19": "19",  # 千葉
    "20": "20",  # 川崎
    "21": "21",  # 平塚
    "22": "22",  # 小田原
    "23": "23",  # 伊東
    "24": "24",  # 静岡
    # 中部
    "25": "25",  # 豊橋
    "26": "26",  # 岐阜(確認要)
    "27": "27",  # 大垣
    "28": "28",  # 四日市
    "29": "29",  # 松阪
    # 近畿
    "30": "30",  # 奈良
    "31": "31",  # 向日町
    "32": "32",  # 和歌山
    "33": "33",  # 岸和田
    # 中四国
    "34": "34",  # 玉野
    "35": "35",  # 広島
    "36": "36",  # 防府
    "37": "37",  # 高松
    "38": "38",  # 小松島
    "39": "39",  # 高知
    "40": "40",  # 松山
    # 九州
    "41": "41",  # 小倉
    "42": "42",  # 久留米
    "43": "43",  # 武雄 ← kdreams.jpでは岐阜? 要確認
    "44": "44",  # 佐世保
    "45": "45",  # 別府
    "46": "46",  # 熊本
    # 北海道・東北
    "52": "52",  # 函館
    "53": "53",  # 青森
    "54": "54",  # いわき平
    "55": "55",  # 弥彦
}


def fetch_payout(kcd: str, kbi: str, rno: int) -> dict | None:
    """
    keirin.jp から払戻金データを取得
    Returns: {bet_type: {sel_tuple: payout_per_100yen}} or None
    """
    url = f"{BASE_URL}?KCD={kcd}&KBI={kbi}&RNO={rno}"
    print(f"\nアクセス: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        print(f"ステータス: {resp.status_code}")
        if resp.status_code != 200:
            print(f"アクセス失敗: {resp.status_code}")
            return None

        import re as _re
        html = _re.sub(r'<\?xml[^>]+\?>', '', resp.content.decode("utf-8", errors="replace"))

        tables = pd.read_html(StringIO(html), flavor="lxml")
        print(f"テーブル数: {len(tables)}")

        payout_keywords = ["単勝", "複勝", "2車単", "2車複", "3連単", "3連複", "ワイド", "払戻"]
        for i, df in enumerate(tables):
            text = df.to_string()
            if any(kw in text for kw in payout_keywords):
                print(f"\n  ★払戻関連 Table[{i}]: shape={df.shape}")
                print(df.to_string())
            else:
                print(f"  Table[{i}]: shape={df.shape}  {str(list(df.columns))[:60]}")

        return tables

    except Exception as e:
        print(f"エラー: {e}")
        return None


def find_kcd_for_venue(venue_name: str, date: str) -> None:
    """指定会場のKCDを総当たりで探す"""
    print(f"\n{venue_name} のKCDを探索中 (日付: {date})")
    for kcd in range(11, 56):
        url = f"{BASE_URL}?KCD={kcd:02d}&KBI={date}&RNO=1"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200 and venue_name in resp.text:
                print(f"  → KCD={kcd:02d} で {venue_name} のデータが見つかりました!")
        except Exception:
            pass


def main():
    if len(sys.argv) == 4:
        kcd, kbi, rno = sys.argv[1], sys.argv[2], int(sys.argv[3])
    else:
        # デフォルト: 岐阜 2026-05-05 R1 (kdreams.jpで race_id=4320260505010001)
        # keirin.jpのKCDが不明なので複数試す
        print("=== keirin.jp アクセステスト ===")
        print("岐阜のKCDを探します（kdreams.jpでは venue_code=43）")
        test_date = "20260201"  # raw_data.jsonにある確定済みレース

        # 20260201の開催会場: 久留米(42?)・伊東(23?)・岐阜(26?)・松阪(29?) を試す
        for kcd in ["23", "26", "27", "29", "42", "40", "39"]:
            url = f"{BASE_URL}?KCD={kcd}&KBI={test_date}&RNO=1"
            print(f"\n試行: KCD={kcd} → {url}")
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                print(f"  ステータス: {resp.status_code}")
                if resp.status_code == 200:
                    print(f"  ページサイズ: {len(resp.text)} bytes")
                    if "岐阜" in resp.text:
                        print(f"  → 岐阜データ発見! KCD={kcd}")
                    if "払戻" in resp.text or "オッズ" in resp.text:
                        print(f"  → 払戻データあり")

                    import re as _re
                    clean = _re.sub(r'<\?xml[^>]+\?>', '', resp.content.decode("utf-8", errors="replace"))
                    tables = pd.read_html(StringIO(clean), flavor="lxml")
                    print(f"  テーブル数: {len(tables)}")

                    payout_keywords = ["単勝", "複勝", "2車単", "2車複", "3連単", "3連複", "ワイド", "払戻"]
                    found = False
                    for i, df in enumerate(tables):
                        text = df.to_string()
                        if any(kw in text for kw in payout_keywords):
                            print(f"  ★払戻関連 Table[{i}]: shape={df.shape}")
                            print(df.to_string())
                            found = True
                    if not found:
                        print(f"  (払戻テーブルなし)")
            except Exception as e:
                print(f"  エラー: {e}")
        return

    fetch_payout(kcd, kbi, rno)


if __name__ == "__main__":
    main()
