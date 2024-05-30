"""
classify.py

Description:
    aqualog データベースから定期的にデータを取得し，深層学習モデルによりビーコンの居る場所を推定する．

Usage:
    python classify.py [options]

Arguments:
    model_path: str
        モデルファイルのパスを指定する．

Options:
    --interval <int> : default=10
        データ取得間隔を指定する．
    --time_window <int> : default=30
        時間窓を指定する．
    --env <str> : default='.env'
        環境変数ファイル名を指定する．

Examples:
    python classify.py model.h5
    python classify.py model.h5 --interval 30 --env .env
"""

import os
import sys
import time
import dotenv
import sqlalchemy
import argparse
import pandas as pd
import torch
import torch.nn as nn

QUERY = "SELECT label, place, d_id, AVG(proxi) FROM room_log WHERE timestamp > (SELECT MAX(timestamp) FROM room_log) - %s GROUP BY label, place, d_id ORDER BY label, place, d_id;"
DETECTORS = [
    ("8-302", "0"),
    ("8-302", "1"),
    ("8-303", "0"),
    ("8-303", "1"),
    ("8-303", "2"),
    ("8-320", "0"),
    ("8-320", "1"),
]
INPUT_DIM = 5
OUTPUT_DIM = 2
HIDDEN_DIM = 10

# 深層学習モデルの定義
class AquaBleClassifier(nn.Module):
    def __init__(self):
        super(AquaBleClassifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x

# コマンドライン引数を処理する
parser = argparse.ArgumentParser(description="classify location of BLE beacons")
parser.add_argument("model_path", type=str, help="path to the model file")
parser.add_argument("-i", "--interval", type=int, default=10, help="fetch interval")
parser.add_argument("-t", "--time_window", type=int, default=30, help="time window")
parser.add_argument("-e", "--env", type=str, default=".env", help="environment file")
args = parser.parse_args()

model_path = args.model_path
interval = args.interval
env_file = args.env

# 環境変数を読み込む
dotenv.load_dotenv(env_file)
host = os.getenv("HOST")
user = os.getenv("USER")
password = os.getenv("PASSWORD")

# モデルを読み込む
model = AquaBleClassifier()
model.load_state_dict(torch.load(model_path))


# データベースに接続する
engine = sqlalchemy.create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/aqualog")

# データベースからユーザ一覧を取得する
user_df = pd.read_sql_query("SELECT label, name FROM ble_tag", engine)

# 位置情報を推定する
while True:
    # interval 秒待つ
    time.sleep(interval)

    # データを取得する
    df = pd.read_sql_query(QUERY % args.time_window, engine)

    # データを処理する
    df["AVG(proxi)"] = df["AVG(proxi)"].map(lambda x: 100 if x > 100 else x)
    for ble_id, gdf in df.groupby("label"):
        # モデルに使用できる形式に変換する
        record = {f"{place}-{detector}": 100 for place, detector in DETECTORS}
        for row in gdf.itertuples():
            record[f"{row.place}-{row.d_id}"] = row._4
        x = torch.tensor([list(record.values())], dtype=torch.float32)

        # # モデルで推論する
        y = model(x[:, 0:5])
        argmax_y = torch.argmax(y).item()
        
        # 推論結果を表示する
        print(f"{["8-302", "8-303", "8-320"][argmax_y]}: {ble_id}, {user_df[user_df['label'] == ble_id]['name'].values[0]}")
    print()

    