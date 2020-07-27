#!bin/bash

args=(
    'uiuc_video_crnn.py'
    'uiuc_video_clstm.py'
    'uiuc_video_cesn.py'
)

for arg in ${args[@]}; do
    python3 -u classifier.py ${arg} > ${arg}.out
done

