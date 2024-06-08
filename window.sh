VERSION=v1
TIME_WINDOWS=(120 180 240)
NEW_BLE_IDS=(15070 15158 15159 15160 15161)
OLD_BLE_IDS=(15153 15155 15162 15163 15164)

for TIME_WINDOW in ${TIME_WINDOWS[@]}; do
    python window.py -t $TIME_WINDOW -i $VERSION/data/*.csv -o $VERSION/windowed/t=$TIME_WINDOW.csv
    python window.py -t $TIME_WINDOW -i $VERSION/data/*.csv -o $VERSION/windowed_boxcox/t=$TIME_WINDOW.csv -x $VERSION/model/boxcox_lambda/t=$TIME_WINDOW.pkl
done

