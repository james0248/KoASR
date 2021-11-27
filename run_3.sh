#! /bin/bash
sleep 0m # Be patient!
nsml run -e ./wav2vec2/main.py -d final_stt_3 \
    -c 8 -g 1 --memory 50G --shm-size 13G -a \
    "--output_dir ./wav2vec2-korean-3
    --model_name_or_path facebook/wav2vec2-large-xlsr-53 
    --num_train_epochs 5
    --per_device_train_batch_size 6
    --per_device_eval_batch_size 6
    --evaluation_strategy steps 
    --eval_steps 1000
    --save_strategy no
    --save_total_limit 2 
    --learning_rate 3e-5
    --warmup_steps 0 
    --attention_dropout 0.094
    --activation_dropout 0.055
    --feat_proj_dropout 0.04
    --hidden_dropout 0.047
    --layerdrop 0.041
    --mask_time_prob 0.082
    --fp16 True 
    --preprocessing_num_workers 8
    --dataloader_num_workers 8
    --group_by_length True
    --freeze_feature_extractor True 
    --mode train
    --gradient_accumulation_steps 10
    --eval_accumulation_steps 500
    --writer_batch_size 500
    --max_split 10
    --split 0
    --data_type 3
    --disable_tqdm True
    --logging_steps 10
    " 
