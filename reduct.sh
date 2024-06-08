VERSION=v1
TIME_WINDOWS=(120 180 240)
METHOD=lda
RECORDS_PER_BLE=80

for TIME_WINDOW in ${TIME_WINDOWS[@]}; do 
    python reduct.py $METHOD \
    -i $VERSION/windowed/t=$TIME_WINDOW.csv \
    -o $VERSION/reducted/t=$TIME_WINDOW.csv \
    -r $RECORDS_PER_BLE \
    -m $VERSION/model/$METHOD/t=$TIME_WINDOW.pkl
    python reduct.py $METHOD \
    -i $VERSION/windowed_boxcox/t=$TIME_WINDOW.csv \
    -o $VERSION/reducted_boxcox/t=$TIME_WINDOW.csv \
    -r $RECORDS_PER_BLE \
    -m $VERSION/model/${METHOD}_boxcox/t=$TIME_WINDOW.pkl
done