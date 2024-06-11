VERSION=v1
TIME_WINDOW=180
INTERVAL=5
PREPROCESS=lda+boxcox

python classify.py \
    nn \
    $VERSION/model/nn_t=${TIME_WINDOW}.pth \
    -i $INTERVAL \
    -t $TIME_WINDOW \
    -x $VERSION/model/boxcox_lambda/t=$TIME_WINDOW.pkl \
    -r $VERSION/model/${PREPROCESS}/t=$TIME_WINDOW.pkl \
    -u
