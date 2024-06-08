"""
window.py

Description:
    fetch.py で取得したデータに時間窓を適用し，機械学習モデルに入力可能な形式に変換する．
        
Usage:
    python window.py [options]

Options:
    --inputs <list[str]> : default=['data.csv']
        入力ファイル名を指定する．
    --output <str> : default='windowed.csv'
        出力ファイル名を指定する．
    --time_window <int> : default=30
        時間窓を指定する．
    --boxcox <str | None> : default=None
        前処理として Box-Cox 変換を行い，パラメータを指定ファイルに保存する．
    --weighted_average : default=False
        重複データの処理に加重平均を用いる．
    --ble_ids_filter <list[int]> : default=None
        フォーマットに使用する BLE の ID を指定する．
    --append : default=False
        出力ファイルが存在する場合に追記する．
    --force : default=False
        出力ファイルが存在する場合に上書きする．

Examples:
    python window.py --input data.csv --output windowed.csv --time_window 30
"""

import os
import sys
import argparse
import pandas as pd
import scipy.stats as stats
import pickle
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
    ("8-417", "0"),
]
HEADER = "label,ble_id," + ",".join(
    [f"{place}-{detector}" for place, detector in DETECTORS]
)
VALUE_IF_UNDETECTED = 300.0
VALUE_IF_UNDETECTED_BOXCOX = 10.0

pd.set_option("future.no_silent_downcasting", True)

# fmt: off
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="window data for machine learning model")
    parser.add_argument("-i", "--inputs", type=str, nargs="+", default=["data.csv"], help="input files")
    parser.add_argument("-o", "--output", type=str, default="windowed.csv", help="output file")
    parser.add_argument("-t", "--time_window", type=int, default=30, help="time window")
    parser.add_argument("-x", "--boxcox", type=str, default=None, help="boxcox parameter file")
    parser.add_argument("-w", "--weighted_average", action="store_true", help="use weighted average for duplicate data")
    parser.add_argument("-b", "--ble_ids_filter", type=int, nargs="+", help="BLE IDs to use for time windowing")
    parser.add_argument("-a", "--append", action="store_true", help="append to output file")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite output file")
    args = parser.parse_args()
    return args
# fmt: on


def boxcox_transform(
    original_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    boxcox_df = original_df.copy()
    lambda_dict = {}
    for (place, detector), df in original_df.groupby(["place", "detector"]):
        transformed_proxi, lambda_ = stats.boxcox(df["proxi"])
        boxcox_df.loc[df.index, "proxi"] = transformed_proxi
        lambda_dict[f"{place}-{detector}"] = lambda_
    return boxcox_df, lambda_dict


def create_record(
    windowed_df: pd.DataFrame,
    label: str,
    ble_id: str,
    time_window: int,
    value_if_undetected: float,
    weighted_average: bool,
) -> dict:
    record = {"label": label, "ble_id": ble_id}
    record.update(
        {f"{place}-{detector}": value_if_undetected for place, detector in DETECTORS}
    )
    if weighted_average:
        for (place, detector), df in windowed_df.groupby(["place", "detector"]):
            # 最新時刻からの時間的距離が [0, time_window / 2] の範囲の重み : 1
            # 最新時刻からの時間的距離が (time_window / 2, time_window] の範囲の重み : 0.5
            latest = df["timestamp"].max()
            time_diff = latest - df["timestamp"]
            weights = time_diff.map(lambda x: 1 if x <= time_window / 2 else 0.5)
            record[f"{place}-{detector}"] = ((df["proxi"] * weights).sum()) / (
                weights.sum()
            )
    else:
        for (place, detector), df in windowed_df.groupby(["place", "detector"]):
            record[f"{place}-{detector}"] = df["proxi"].mean()
    return record


def window_data(
    original_df: pd.DataFrame,
    time_window: int,
    value_if_undetected: float,
    weighted_average: bool,
    ble_ids_filter: list[int] | None = None,
) -> pd.DataFrame:
    # データを整形する
    original_df["datetime"] = pd.to_datetime(original_df["timestamp"], unit="s")
    result_df = pd.DataFrame(columns=HEADER.split(","))

    for (label, ble_id), grouped_df in original_df.groupby(["label", "ble_id"]):
        if ble_ids_filter is not None and ble_id not in ble_ids_filter:
            continue
        grouped_df = grouped_df.sort_values("datetime")

        # 最初の len(DETECTORS) 行は無視して，レコードを作成する
        windowed_df_iter = grouped_df.rolling(
            f"{time_window}s", on="datetime"
        ).__iter__()
        windowed_df_iter = islice(windowed_df_iter, len(DETECTORS), None)
        for windowed_df in windowed_df_iter:
            record = create_record(
                windowed_df,
                label,
                ble_id,
                time_window,
                value_if_undetected,
                weighted_average,
            )
            result_df.loc[len(result_df)] = record

    return result_df


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

    if args.boxcox is not None and os.path.exists(args.boxcox):
        print(f"error: {args.boxcox} already exists", file=sys.stderr)
        sys.exit(1)

    # データを読み込む
    original_df = pd.concat(
        [pd.read_csv(input_file) for input_file in args.inputs], ignore_index=True
    )

    # データを前処理する
    if args.boxcox is not None:
        original_df, lambda_dict = boxcox_transform(original_df)
        with open(args.boxcox, "wb") as f:
            pickle.dump(lambda_dict, f)

    # データを整形する
    result_df = window_data(
        original_df,
        args.time_window,
        VALUE_IF_UNDETECTED if args.boxcox is None else VALUE_IF_UNDETECTED_BOXCOX,
        args.weighted_average,
        args.ble_ids_filter,
    )

    # データを書き込む
    if args.force:
        result_df.to_csv(args.output, index=False)
    elif args.append:
        result_df.to_csv(args.output, mode="a", header=False, index=False)
    else:
        result_df.to_csv(args.output, index=False)

    print(f"Write {len(result_df)} records to {args.output}", file=sys.stderr)
