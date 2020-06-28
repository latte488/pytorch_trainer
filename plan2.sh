#!bin/bash

args=(
    'uiuc_video_v1_conv_tanh_rnn.py'
    'uiuc_video_v2_conv_tanh_rnn.py'
    'uiuc_video_v3_conv_tanh_rnn.py'
)

for arg in ${args[@]}; do
    python3 -u classifier.py ${arg} > ${arg}.out
done

