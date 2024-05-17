"""
fetch.py

Description:
    aqualog データベースから定期的にデータを取得し，正解ラベルとして部屋番号を付与して出力ファイルに保存する．

Usage:
    python fetch.py room_label [options]

Arguments:
    room_label: str
        部屋番号を指定する．

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

QUERY = "SELECT * FROM room_log;"

# コマンドライン引数を処理する
parser = argparse.ArgumentParser(description="fetch data from aqualog database")
parser.add_argument("room_label", type=str, help="room number")
parser.add_argument("-i", "--interval", type=int, default=120, help="fetch interval")
parser.add_argument("-o", "--output", type=str, default="data.csv", help="output file")
parser.add_argument("-a", "--append", action="store_true", help="append to output file")
parser.add_argument("-f", "--force", action="store_true", help="force overwrite output file")
parser.add_argument("-e", "--env", type=str, default=".env", help="environment file")
args = parser.parse_args()

room_label = args.room_label
interval = args.interval
output_file = args.output
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
    database="aqualog"
) as connection:
    if connection.is_connected():
        print("Connected to MySQL database", file=sys.stderr)
    else:
        print("Failed to connect to MySQL database", file=sys.stderr)
        exit(1)

    # 出力ファイルを開く
    with open(output_file, "a" if append else "w") as f:
        while True:
            # データを取得する
            with connection.cursor() as cursor:
                cursor.execute(QUERY)
                for row in cursor.fetchall():
                    f.write(f"{room_label}," + ",".join(map(str, row)) + "\n")
            print("Fetched data from aqualog database", file=sys.stderr)
            # interval 秒待つ
            time.sleep(interval)

