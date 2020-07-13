#!bin/bash

args=(
    'uiuc_video_rnn.py'
    'uiuc_video_lstm.py'
    'uiuc_video_esn.py'
)

for arg in ${args[@]}; do
    python3 -u classifier.py ${arg} > ${arg}.out
done

