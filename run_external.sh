#! /bin/bash
sleep 0 # Be patient!
nsml run -e ./wav2vec2/main.py -d final_stt_1 \
    -c 8 -g 1 --memory 80G --shm-size 2G -a \
    "--output_dir ./wav2vec2-korean-ext
    --model_name_or_path facebook/wav2vec2-large-xlsr-53
    --num_train_epochs 5
    --per_device_train_batch_size 6
    --per_device_eval_batch_size 6
    --learning_rate 1e-4
    --gradient_accumulation_steps 40
    --evaluation_strategy epoch
    --save_strategy no
    --save_total_limit 2 
    --warmup_steps 100 
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
    --length_column_name length
    --freeze_feature_extractor True 
    --mode train
    --eval_accumulation_steps 500
    --writer_batch_size 500
    --data_type 2
    --disable_tqdm True
    --logging_steps 10
    --use_external_data True
    --gdrive_code 4/0AX4XfWihNVpt-SloOyKwT-AeXfa_dx3RH7iIU-SbHlQZiATKY6DbAFMP-74yPVwR3f9Egg
    " 
