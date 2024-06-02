"""
format.py

Description:
    fetch.py で取得したデータを，機械学習モデルに入力可能な形式に変換する．
        
Usage:
    python format.py [options]

Options:
    --input <str> : default='data.csv'
        入力ファイル名を指定する．
    --output <str> : default='formatted.csv'
        出力ファイル名を指定する．
    --time_window <int> : default=30
        時間窓を指定する．
    --weighted_average : default=False
        重複データの処理に加重平均を用いる．
    --append : default=False
        出力ファイルが存在する場合に追記する．
    --force : default=False
        出力ファイルが存在する場合に上書きする．

Examples:
    python format.py --input data.csv --output formatted.csv --time_window 30
"""

import os
import sys
import argparse
import pandas as pd

DETECTORS = [
    ("8-302", "0"),
    ("8-302", "1"),
    ("8-303", "0"),
    ("8-303", "1"),
    ("8-303", "2"),
]
HEADER = "label," + ",".join([f"{place}-{detector}" for place, detector in DETECTORS])

# コマンドライン引数を処理する
parser = argparse.ArgumentParser(description="format data for machine learning model")
parser.add_argument("-i", "--input", type=str, default="data.csv", help="input file")
parser.add_argument("-o", "--output", type=str, default="formatted.csv", help="output file")
parser.add_argument("-t", "--time_window", type=int, default=30, help="time window")
parser.add_argument("-w", "--weighted_average", action="store_true", help="use weighted average for duplicate data")
parser.add_argument("-a", "--append", action="store_true", help="append to output file")
parser.add_argument("-f", "--force", action="store_true", help="force overwrite output file")
args = parser.parse_args()

input_file = args.input
output_file = args.output if args.output.endswith(".csv") else f"{args.output}.csv"
time_window = args.time_window
weighted_average = args.weighted_average
append = args.append
force = args.force

# 入力ファイルが存在するか確認する
if not os.path.exists(input_file):
    print(f"error: {input_file} not found", file=sys.stderr)
    sys.exit(1)

# 出力ファイルが存在するか確認する
if os.path.exists(output_file):
    if force:
        print(f"Overwrite {output_file}", file=sys.stderr)
    elif append:
        print(f"Append to {output_file}", file=sys.stderr)
    else:
        print(f"error: {output_file} already exists", file=sys.stderr)
        sys.exit(1)
else:
    print(f"Create {output_file}", file=sys.stderr)

# データを読み込む
original_df = pd.read_csv(input_file)
original_df["datetime"] = pd.to_datetime(original_df["timestamp"], unit="s")

# データを整形する
formatted_df = pd.DataFrame(columns=HEADER.split(","))
for (label, ble_id), df in original_df.groupby(["label", "ble_id"]):
    df = df.sort_values("datetime")
    tdf = df.rolling(f"{time_window}s", on="datetime" )
    tmp_df = pd.DataFrame(columns=formatted_df.columns)
    for dfi in tdf:
        record = {"label": label}
        if weighted_average:
            for (place, detector), tdfi in dfi.groupby(["place", "detector"]):
                # 最新時刻からの時間的距離が [0, time_window / 2] の範囲の重み : 1
                # 最新時刻からの時間的距離が (time_window / 2, time_window] の範囲の重み : 0.5
                latest = tdfi["timestamp"].max()
                time_diff = (latest - tdfi["timestamp"])
                weights = time_diff.map(lambda x: 1 if x <= time_window / 2 else 0.5)
                record[f"{place}-{detector}"] = ((tdfi["proxi"] * weights).sum()) / (weights.sum())
        else:    
            for (place, detector), tdfi in dfi.groupby(["place", "detector"]):
                record[f"{place}-{detector}"] = tdfi["proxi"].mean()
        tmp_df.loc[len(tmp_df)] = record
    formatted_df = pd.concat([formatted_df, tmp_df.loc[len(DETECTORS):]], ignore_index=True)

# Null データと基準値以上の値を 100 に変換する
formatted_df = formatted_df.fillna(100)
formatted_df[HEADER.split(",")[1:]] = formatted_df[HEADER.split(",")[1:]].map(lambda x: 100 if x > 100 else x)

# データを書き込む
if force:
    formatted_df.to_csv(output_file, index=False)
elif append:
    formatted_df.to_csv(output_file, mode="a", header=False, index=False)
else:
    formatted_df.to_csv(output_file, index=False)