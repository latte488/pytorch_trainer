#!bin/bash

origin='uiuc_video_cesn.py'

for i in {0..9}; do
    arg=No_${i}_${origin}
    cp ${origin} ${arg}
    python3 -u classifier.py ${arg} > ${arg}.out
done

