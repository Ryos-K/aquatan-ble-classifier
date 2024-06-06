VERSION=v1
TIME_WINDOWS=(60 90 120 150)

for TIME_WINDOW in ${TIME_WINDOWS[@]}; do
    python format.py -i $VERSION/data/* -o $VERSION/formatted/t=$TIME_WINDOW.csv -t $TIME_WINDOW
    python format.py -i $VERSION/data/* -o $VERSION/formatted_w/t=$TIME_WINDOW.csv -t $TIME_WINDOW -w
done

