"""
fetch.py

Description:
    aqualog データベースから定期的にデータを取得し，正解ラベルとして場所名を付与して出力ファイルに保存する．

Usage:
    python fetch.py label [options]

Arguments:
    label: str
        場所名を指定する．
    ble_ids: list[int]
        BLE ビーコンの番号を指定する．

Options:
    --interval <int> : default=120
        データ取得間隔を指定する．
    --output <str> : default='data.csv'
        出力ファイル名を指定する．
    --append : default=False
        出力ファイルが存在する場合に追記する．
    --force : default=False
        出力ファイルが存在する場合に上書きする．
    --env <str> : default='.env'
        環境変数ファイル名を指定する．

Examples:
    python fetch.py 8-302
    python fetch.py 8-302 --interval 120 --output data.csv --env .env
"""

import os
import sys
import time
import dotenv
import mysql.connector
import argparse

# ble_id は label 列に対応する
QUERY_FOR_ACCOUNT = "SELECT * FROM ble_tag WHERE label IN (%s)"
QUERY_FOR_OBSERVATION = "SELECT * FROM room_log WHERE label IN (%s)"
LABELS = [
    "8-302",
    "8-303",
    "8-320",
    "8-322",
    "corridor",
]
HEADER = "label,id,timestamp,ble_id,place,proxi,detector,batt"

# コマンドライン引数を処理する
parser = argparse.ArgumentParser(description="fetch data from aqualog database")
parser.add_argument("label", type=str, choices=LABELS, help="label of the data")
parser.add_argument("ble_ids", type=int, nargs="+", help="BLE beacon IDs")
parser.add_argument("-i", "--interval", type=int, default=120, help="fetch interval")
parser.add_argument("-o", "--output", type=str, default="data.csv", help="output file")
parser.add_argument("-a", "--append", action="store_true", help="append to output file")
parser.add_argument("-f", "--force", action="store_true", help="force overwrite output file")
parser.add_argument("-e", "--env", type=str, default=".env", help="environment file")
args = parser.parse_args()

label = args.label
ble_ids = args.ble_ids
interval = args.interval
output_file = args.output if args.output.endswith(".csv") else args.output + ".csv"
append = args.append
force = args.force
env_file = args.env

# 出力ファイルが存在するか確認する
if os.path.exists(output_file):
    if force:
        print(f"Overwrite {output_file}", file=sys.stderr)
    elif append:
        print(f"Append to {output_file}", file=sys.stderr)
    else:
        print(f"{output_file} already exists. Use --force option to overwrite", file=sys.stderr)
        exit(1)
else:
    print(f"Create {output_file}", file=sys.stderr)

# 環境変数を読み込む
dotenv.load_dotenv(env_file)
host = os.getenv("HOST")
user = os.getenv("USER")
password = os.getenv("PASSWORD")

# データベースに接続する
with mysql.connector.connect(
    host=host,
    user=user,
    password=password,
    database="aqualog",

) as connection:
    if connection.is_connected():
        print("Connected to MySQL database", file=sys.stderr)
    else:
        print("Failed to connect to MySQL database", file=sys.stderr)
        exit(1)

    # ble_ids に対応するアカウントが存在するか確認する．
    with connection.cursor() as cursor:
        cursor.execute(QUERY_FOR_ACCOUNT % ",".join(map(str, ble_ids)))
        accounts = cursor.fetchall()
        for account in accounts:
            active = "active  " if account[6] else "inactive"
            print(f"{active}: {account[1]}: {account[3]}", file=sys.stderr)
        if len(accounts) != len(ble_ids):
            print("Some BLE beacon IDs are not registered", file=sys.stderr)
            exit(1)

    # 出力ファイルを開く
    with open(output_file, "a" if append else "w") as f:
        # ヘッダを書き込む
        if not append:
            f.write(HEADER + "\n")
        while True:
            # データベースを最新の状態に更新する
            connection.commit()
            
            # データを取得する
            with connection.cursor() as cursor:
                cursor.execute(QUERY_FOR_OBSERVATION % ",".join(map(str, ble_ids)))
                for row in cursor.fetchall():
                    f.write(f"{label}," + ",".join(map(str, row)) + "\n")
            f.flush()
            print("Fetched data from aqualog database", file=sys.stderr)
            # interval 秒待つ
            time.sleep(interval)

