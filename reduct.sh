VERSION=v1
TIME_WINDOWS=(180 240)
METHOD=lda
RECORDS_PER_BLE=75
N_COMPONENTS=4

for TIME_WINDOW in ${TIME_WINDOWS[@]}; do 
    python reduct.py $METHOD \
    -i $VERSION/windowed/t=$TIME_WINDOW.csv \
    -o $VERSION/reducted/t=$TIME_WINDOW.csv \
    -r $RECORDS_PER_BLE \
    -m $VERSION/model/$METHOD/t=$TIME_WINDOW.pkl \
    -n $N_COMPONENTS
    python reduct.py $METHOD \
    -i $VERSION/windowed+boxcox/t=$TIME_WINDOW.csv \
    -o $VERSION/reducted+boxcox/t=$TIME_WINDOW.csv \
    -r $RECORDS_PER_BLE \
    -m $VERSION/model/${METHOD}+boxcox/t=$TIME_WINDOW.pkl \
    -n $N_COMPONENTS
done