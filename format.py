"""
format.py

Description:
    fetch.py で取得したデータを，機械学習モデルに入力可能な形式に変換する．
        
Usage:
    python format.py [options]

Options:
    --inputs <list[str]> : default=['data.csv']
        入力ファイル名を指定する．
    --output <str> : default='formatted.csv'
        出力ファイル名を指定する．
    --time_window <int> : default=30
        時間窓を指定する．
    --weighted_average : default=False
        重複データの処理に加重平均を用いる．
    --ble_ids_filter <list[int]> : default=None
        フォーマットに使用する BLE の ID を指定する．
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
from itertools import islice

DETECTORS = [
    ("8-302", "0"),
    ("8-302", "1"),
    ("8-303", "0"),
    ("8-303", "1"),
    ("8-320", "0"),
    ("8-320", "1"),
    ("8-322", "0"),
    ("8-322", "1"),
    ("8-417", "0")
]
HEADER = "label," + ",".join([f"{place}-{detector}" for place, detector in DETECTORS])
VALUE_IF_UNDETECTED = 300.0

pd.set_option('future.no_silent_downcasting', True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="format data for machine learning model")
    parser.add_argument("-i", "--inputs", type=str, nargs="+", default=["data.csv"], help="input files")
    parser.add_argument("-o", "--output", type=str, default="formatted.csv", help="output file")
    parser.add_argument("-t", "--time_window", type=int, default=30, help="time window")
    parser.add_argument("-w", "--weighted_average", action="store_true", help="use weighted average for duplicate data")
    parser.add_argument("-b", "--ble_ids_filter", type=int, nargs="+", help="BLE IDs to use for formatting")
    parser.add_argument("-a", "--append", action="store_true", help="append to output file")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite output file")
    args = parser.parse_args()
    return args

def create_record(windowed_df: pd.DataFrame, label: str, time_window: int, weighted_average: bool) -> dict:
    record = {"label": label}
    record.update({f"{place}-{detector}": VALUE_IF_UNDETECTED for place, detector in DETECTORS})
    if weighted_average:
        for (place, detector), df in windowed_df.groupby(["place", "detector"]):
            # 最新時刻からの時間的距離が [0, time_window / 2] の範囲の重み : 1
            # 最新時刻からの時間的距離が (time_window / 2, time_window] の範囲の重み : 0.5
            latest = df["timestamp"].max()
            time_diff = (latest - df["timestamp"])
            weights = time_diff.map(lambda x: 1 if x <= time_window / 2 else 0.5)
            record[f"{place}-{detector}"] = ((df["proxi"] * weights).sum()) / (weights.sum())
    else:    
        for (place, detector), df in windowed_df.groupby(["place", "detector"]):
            record[f"{place}-{detector}"] = df["proxi"].mean()
    return record

def format_data(original_df: pd.DataFrame, time_window: int, weighted_average: bool, ble_ids_filter: list[int] | None = None) -> pd.DataFrame:
    # データを整形する
    original_df["datetime"] = pd.to_datetime(original_df["timestamp"], unit="s")
    formatted_df = pd.DataFrame(columns=HEADER.split(","))

    for (label, ble_id), grouped_df in original_df.groupby(["label", "ble_id"]):
        if ble_ids_filter is not None and ble_id not in ble_ids_filter:
            continue
        grouped_df = grouped_df.sort_values("datetime")

        # 最初の len(DETECTORS) 行は無視して，レコードを作成する
        windowed_df_iter = grouped_df.rolling(f"{time_window}s", on="datetime").__iter__()
        windowed_df_iter = islice(windowed_df_iter, len(DETECTORS), None)
        for windowed_df in windowed_df_iter:
            record = create_record(windowed_df, label, time_window, weighted_average)
            formatted_df.loc[len(formatted_df)] = record

    return formatted_df

if __name__ == "__main__":
    args = parse_args()

    # 入力ファイルが存在するか確認する
    for input_file in args.inputs:
        if not os.path.exists(input_file):
            print(f"error: {input_file} does not exist", file=sys.stderr)
            sys.exit(1)

    # 出力ファイルが存在するか確認する
    if os.path.exists(args.output):
        if args.force:
            print(f"Overwrite {args.output}", file=sys.stderr)
        elif args.append:
            print(f"Append to {args.output}", file=sys.stderr)
        else:
            print(f"error: {args.output} already exists", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Create {args.output}", file=sys.stderr)


    # データを読み込む
    original_df = pd.concat([pd.read_csv(input_file) for input_file in args.inputs])

    # データを整形する
    formatted_df = format_data(original_df, args.time_window, args.weighted_average, args.ble_ids_filter)

    # データを書き込む
    if args.force:
        formatted_df.to_csv(args.output, index=False)
    elif args.append:
        formatted_df.to_csv(args.output, mode="a", header=False, index=False)
    else:
        formatted_df.to_csv(args.output, index=False)