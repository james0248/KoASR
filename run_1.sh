#! /bin/bash
sleep 0 # Be patient!
nsml run -e ./wav2vec2/new_main.py -d stt_1 \
    -c 8 -g 1 --memory 70G --shm-size 13G -a \
    "--output_dir ./wav2vec2-korean-1
<<<<<<< HEAD
    --num_train_epochs 2
    --per_device_train_batch_size 2
    --per_device_eval_batch_size 2
    --evaluation_strategy steps 
    --save_total_limit 3 
    --save_steps 700 
    --eval_steps 100 
    --learning_rate 5e-4 
    --warmup_steps 1000 
    --model_name_or_path facebook/wav2vec2-large-xlsr-53 
    --fp16 True 
    --preprocessing_num_workers 8
    --length_column_name length
    --group_by_length True 
    --freeze_feature_extractor True 
    --mode train
    --split 0
    --max_split 5
    --gradient_accumulation_steps 10" 
=======
    --num_train_epochs 5
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --evaluation_strategy steps 
    --save_total_limit 3 
    --save_steps 700 
    --eval_steps 700 
    --learning_rate 5e-5 
    --warmup_steps 3000 
    --model_name_or_path fleek/wav2vec-large-xlsr-korean
    --fp16 True 
    --preprocessing_num_workers 8
    --group_by_length False
    --freeze_feature_extractor True 
    --mode trai
    --gradient_accumulation_steps 10"
>>>>>>> c0ddbd2 (Fix submit errors)
