VERSION=v1
ROOM=8303
BLE_IDS=(15070 15153 15155 15158 15159 15160 15161 15162 15163 15164)
INTERVAL=300

echo fetch.py $ROOM ${BLE_IDS[@]} -o $VERSION/data/$ROOM.csv -i $INTERVAL
python fetch.py $ROOM ${BLE_IDS[@]} -o $VERSION/data/$ROOM.csv -i $INTERVAL