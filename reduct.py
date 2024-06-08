"""
reduct.py

Description:
    window.py で整形したデータの特徴量の次元削減を行い，次元削減モデルを保存する．

Usage:
    python reduct.py method [options]

Arguments:
    method: ["pca", "lda"]
        次元削減手法を指定する．

Options:
    --input <str> : default='windowed.csv'
        入力ファイル名を指定する．
    --output <str> : default='reducted.csv'
        出力ファイル名を指定する．
    --model_output <str> : default='model.pkl'
        モデルファイル名を指定する．
    --records_per_ble <int> : default=100
        BLE ごとの学習に使用するデータ数を指定する．
    --n_components <int> : default=2
        次元数を指定する．
    --force : default=False
        出力ファイルが存在する場合に上書きする．

Examples:
    python reduct.py pca --input windowed.csv --output reducted.csv --n_components 2
"""

import os
import sys
import argparse
import pandas as pd
import sklearn.decomposition as skd
import sklearn.discriminant_analysis as skda
import pickle

METHODS = ["pca", "lda"]

# fmt: off
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="window data for machine learning model")
    parser.add_argument("method", choices=METHODS, help="dimension reduction method")
    parser.add_argument("-i", "--input", type=str, default="windowed.csv", help="input file")
    parser.add_argument("-o", "--output", type=str, default="reducted.csv", help="output file")
    parser.add_argument("-m", "--model_output", type=str, default="model.pkl", help="model file")
    parser.add_argument("-r", "--records_per_ble", type=int, default=100, help="number of records per BLE")
    parser.add_argument("-n", "--n_components", type=int, default=2, help="number of components")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite the output file")
    return parser.parse_args()
# fmt: on


if __name__ == "__main__":
    args = parse_args()

    # 入力ファイルが存在するか確認する
    if not os.path.exists(args.input):
        print(f"error: {args.input} does not exist", file=sys.stderr)
        sys.exit(1)

    # 出力ファイルが存在するか確認する
    if os.path.exists(args.output):
        if args.force:
            print(f"Overwrite {args.output}", file=sys.stderr)
        else:
            print(f"error: {args.output} already exists", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Create {args.output}", file=sys.stderr)

    if os.path.exists(args.model_output):
        if args.force:
            print(f"Overwrite {args.model_output}", file=sys.stderr)
        else:
            print(f"error: {args.model_output} already exists", file=sys.stderr)
            sys.exit(1)

    # データを読み込む
    df = pd.read_csv(args.input)
    sampled_df = df.groupby(["label", "ble_id"]).sample(n=args.records_per_ble, random_state=0)
    sampled_df.reset_index(drop=True, inplace=True)

    # 次元削減モデルの学習を行う
    match args.method:
        case "pca":
            model = skd.PCA(n_components=args.n_components)
            model.fit(sampled_df.drop(columns=["label", "ble_id"]))
        case "lda":
            model = skda.LinearDiscriminantAnalysis(n_components=args.n_components)
            model.fit(sampled_df.drop(columns=["label", "ble_id"]), sampled_df["label"])
        case _:
            print(f"error: invalid method {args.method}", file=sys.stderr)
            sys.exit(1)

    # 寄与率を表示する
    print(model.explained_variance_ratio_)

    # 次元削減を行う
    reducted_df = df[["label", "ble_id"]]
    reducted_df = pd.concat(
        [
            reducted_df,
            pd.DataFrame(
                model.transform(df.drop(columns=["label", "ble_id"])),
                columns=[f"C{i}" for i in range(args.n_components)],
            ),
        ],
        axis=1,
    )

    # データを保存する
    reducted_df.to_csv(args.output, index=False)

    # モデルを保存する
    with open(args.model_output, "wb") as f:
        pickle.dump(model, f)
