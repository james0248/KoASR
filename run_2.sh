#! /bin/bash
sleep 0 # Be patient!
nsml run -e ./wav2vec2/new_main.py -d stt_2 \
    -c 8 -g 1 --memory 70G --shm-size 13G -a \
    "--output_dir ./wav2vec2-korean-1
    --model_name_or_path facebook/wav2vec2-large-xlsr-53 
    --num_train_epochs 3
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 4
    --evaluation_strategy steps 
    --eval_steps 1000
    --save_strategy no
    --save_total_limit 2 
    --learning_rate 1e-4 
    --warmup_steps 1000 
    --fp16 True 
    --preprocessing_num_workers 8
    --group_by_length True
    --length_column_name length
    --freeze_feature_extractor True 
    --mode train
    --split 0
    --max_split 1
    --gradient_accumulation_steps 10
    --eval_accumulation_steps 500
    --writer_batch_size 500" 
