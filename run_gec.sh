#! /bin/bash
sleep 0m # Be patient!
nsml run -e ./gec/main.py -d final_stt_2 \
    -c 8 -g 1 --memory 80G --shm-size 2G -a \
    "--output_dir ./wav2vec2-korean-1
    --model_name_or_path hyunwoongko/kobart
    --num_train_epochs 10
    --per_device_train_batch_size 256
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps 2
    --evaluation_strategy no
    --save_strategy no
    --save_total_limit 2 
    --learning_rate 5e-5
    --warmup_steps 10 
    --dropout 0.07
    --attention_dropout 0.094
    --activation_dropout 0.055
    --classifier_dropout 0.05
    --encoder_layerdrop 0.05
    --decoder_layerdrop 0.05
    --generation_num_beams 1
    --fp16 True 
    --preprocessing_num_workers 8
    --dataloader_num_workers 8
    --mode train
    --eval_accumulation_steps 500
    --data_type 1
    --disable_tqdm True
    --logging_steps 10
    --max_source_length 50
    --pad_to_max_length False
    --generation_max_length 50
    --max_target_length 50
    --ignore_pad_token_for_loss True
    --seed 777
    --use_processed_data True
    " 
