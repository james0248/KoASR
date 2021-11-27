#! /bin/bash
nsml run -e ./wav2vec2/main.py -d final_stt_2 \
    -c 2 -g 0 --memory 10G --shm-size 1G