"""
classify.py

Description:
    aqualog データベースから定期的にデータを取得し，深層学習モデルによりビーコンの居る場所を推定する．

Usage:
    python classify.py method model_path [options]

Arguments:
    method: ["nn", "knn"]
        使用するモデルを指定する．
    model_path: str
        モデルファイルのパスを指定する．

Options:
    --interval <int> : default=60
        データ取得間隔を指定する．
    --time_window <int> : default=180
        時間窓を指定する．
    --boxcox <str | None> : default=None
        指定されたファイルからパラメータを読み込み，Box-Cox 変換を行う．
    --reduce <str | None> : default=None
        指定されたファイルから次元削減モデルを読み込み，次元削減を行う．
    --weighted_average : default=False
        重複データの処理に加重平均を用いる．
    --update : default=False
        推論結果をデータベースに保存する．
    --env <str> : default='.env'
        環境変数ファイル名を指定する．

Examples:
    python classify.py model.pth
    python classify.py model.pth --interval 30 --env .env
"""

import os
import sys
import time
import dotenv
import sqlalchemy
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
from collections import namedtuple
import torch
import torch.nn as nn

VALUE_IF_UNDETECTED = 300.0
VALUE_IF_UNDETECTED_BOXCOX = 10.0
QUERY_FOR_ACCOUNT = "SELECT label, name FROM ble_tag"
QUERY_FOR_OBSERVATION = "SELECT * FROM room_log WHERE timestamp > (SELECT MAX(timestamp) FROM room_log) - {time_window} ORDER BY label, place, d_id;"
QUERY_FOR_UPSERT = "INSERT INTO beacon_status (label, place) VALUES ({ble_id}, '{place}') ON DUPLICATE KEY UPDATE place = '{place}';"
QUERY_FOR_DELETE = "DELETE FROM beacon_status WHERE label = {ble_id};"
QUERY_FOR_ALLDELETE = "DELETE FROM beacon_status;"
DETECTORS = [
    ("8-302", "0"),
    ("8-302", "1"),
    ("8-303", "0"),
    ("8-303", "1"),
    ("8-320", "0"),
    ("8-320", "1"),
    ("8-322", "0"),
    ("8-322", "1"),
    ("8-417", "0"),
]
PLACES = ["8-302", "8-303", "8-320", "8-322", "8-417"]

def parse_args():
    parser = argparse.ArgumentParser(description="classify location of BLE beacons")
    parser.add_argument("method", choices=["nn", "knn"], help="model selection")
    parser.add_argument("model_path", type=str, help="path to the model file")
    parser.add_argument("-i", "--interval", type=int, default=60, help="fetch interval")
    parser.add_argument("-t", "--time_window", type=int, default=180, help="time window")
    parser.add_argument("-x", "--boxcox", type=str, default=None, help="boxcox parameter file")
    parser.add_argument("-r", "--reduce", type=str, default=None, help="dimension reduction model file")
    parser.add_argument("-u", "--update", action="store_true", help="update database")
    parser.add_argument("-w", "--weighted_average", action="store_true", help="use weighted average for duplicate data")
    parser.add_argument("-e", "--env", type=str, default=".env", help="environment file")
    args = parser.parse_args()
    return args

def create_record(
    windowed_df: pd.DataFrame,
    time_window: int,
    value_if_undetected: float,
    weighted_average: bool,
) -> pd.DataFrame:
    record = {f"{place}-{detector}": value_if_undetected for place, detector in DETECTORS}
    
    if weighted_average:
        for (place, detector), df in windowed_df.groupby(["place", "d_id"]):
            # 最新時刻からの時間的距離が [0, time_window / 2] の範囲の重み : 1
            # 最新時刻からの時間的距離が (time_window / 2, time_window] の範囲の重み : 0.5
            latest = df["timestamp"].max()
            time_diff = latest - df["timestamp"]
            weights = time_diff.map(lambda x: 1 if x <= time_window / 2 else 0.5)
            record[f"{place}-{detector}"] = ((df["proxi"] * weights).sum()) / (
                weights.sum()
            )
    else:
        for (place, detector), df in windowed_df.groupby(["place", "d_id"]):
            record[f"{place}-{detector}"] = df["proxi"].mean()
    return pd.DataFrame([record])

if __name__ == "__main__":
    args = parse_args()

    # 環境変数を読み込む
    dotenv.load_dotenv(args.env)
    host = os.getenv("AQUATAN_HOST")
    user = os.getenv("AQUATAN_USER")
    password = os.getenv("AQUATAN_PASSWORD")
    database = os.getenv("AQUATAN_DATABASE")
    print(f"host: {host}, user: {user}, database: {database}")

    # モデルを読み込む
    if args.method == "nn":
        model = torch.jit.load(args.model_path)
        model.eval()
    elif args.method == "knn":
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)

    if args.boxcox is not None:
        # Box-Cox 変換のパラメータを読み込む
        with open(args.boxcox, "rb") as f:
            lambda_values = pickle.load(f)

    if args.reduce is not None:
        # 次元削減モデルを読み込む
        with open(args.reduce, "rb") as f:
            reducer = pickle.load(f)

    # データベースに接続する
    engine = sqlalchemy.create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

    # データベースからユーザ一覧を取得する
    user_df = pd.read_sql_query(QUERY_FOR_ACCOUNT, engine)

    if args.update:
        # 予測結果の記録用辞書
        Prediction = namedtuple("Prediction", ["place", "times", "flag"])
        prediction_dict: dict[int, Prediction] = {}
        pd.read_sql_query(QUERY_FOR_ALLDELETE, engine)

    # 位置情報を推定する
    while True:
        # Flag のリセット
        for ble_id in prediction_dict:
            prediction_dict[ble_id] = prediction_dict[ble_id]._replace(flag=False)

        # データを取得する
        query = QUERY_FOR_OBSERVATION.format(time_window=args.time_window)
        df = pd.read_sql_query(query, engine)

        if args.boxcox is not None:
            # データを Box-Cox 変換する
            for (place, detector), grouped_df in df.groupby(["place", "d_id"]):
                df.loc[grouped_df.index, "proxi"] = stats.boxcox(grouped_df["proxi"], lambda_values[f"{place}-{detector}"])

        # データを処理する
        for ble_id, windowed_df in df.groupby("label"):
            # ユーザテーブルに存在しない BLE ID は無視する
            if ble_id not in user_df["label"].values:
                continue

            # モデルに使用できる形式に変換する
            value_if_undetected = VALUE_IF_UNDETECTED if args.boxcox is None else VALUE_IF_UNDETECTED_BOXCOX
            x = create_record(windowed_df, args.time_window, value_if_undetected, args.weighted_average)

            if args.reduce is not None:
                # 次元削減する
                x = pd.DataFrame(reducer.transform(x[x.columns]), columns=[f"C{i}" for i in range(reducer.n_components)])

            # # モデルで推論する
            if args.method == "nn":
                x = torch.tensor(x.values, dtype=torch.float32)
                y = model(x)
                place = PLACES[y.argmax()]
            elif args.method == "knn":
                y = model.predict(x)
                place = y[0]
            
            if args.update:
                # 予測結果を記録する
                if ble_id in prediction_dict and prediction_dict[ble_id].place == place:
                    prediction_dict[ble_id] = prediction_dict[ble_id]._replace(times=min(3, prediction_dict[ble_id].times + 1), flag=True)
                else:
                    prediction_dict[ble_id] = Prediction(place, 1, True)
            else:
                # 推論結果を表示する
                print(f"{place}: {ble_id}, {user_df[user_df['label'] == ble_id]['name'].values[0]}")
                if args.method == "nn":
                    y_formatted = "[" + ", ".join([f"{y_:.6f}" for y_ in y.tolist()[0]]) + "]"
                    print(f"y: {y_formatted}")

        if args.update:
            with engine.connect() as connection:
                drop_ble_ids = []
                # 予測結果をデータベースに保存する
                for ble_id, prediction in prediction_dict.items():
                    if prediction.flag and prediction.times == 3:
                        query = QUERY_FOR_UPSERT.format(ble_id=ble_id, place=prediction.place)
                        connection.execute(sqlalchemy.text(query))
                        connection.commit()
                    if not prediction.flag:
                        query = QUERY_FOR_DELETE.format(ble_id=ble_id)
                        connection.execute(sqlalchemy.text(query))
                        connection.commit()
                        drop_ble_ids.append(ble_id)
                for ble_id in drop_ble_ids:
                    prediction_dict.pop(ble_id)

        # interval 秒待つ
        time.sleep(args.interval)
        