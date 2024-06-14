#!/bin/bash
TIME_WINDOW=180
INTERVAL=30
PREPROCESS=lda+boxcox

pipenv run python classify.py \
    knn \
    model/knn_t=${TIME_WINDOW}.pth \
    -i $INTERVAL \
    -t $TIME_WINDOW \
    -x model/boxcox_lambda/t=$TIME_WINDOW.pkl \
    -r model/${PREPROCESS}/t=$TIME_WINDOW.pkl \
    -u