#! /bin/bash
sleep 0 # Be patient!
nsml run -e ./wav2vec2/main.py -d final_stt_1 \
    -c 4 -g 0 --memory 80G --shm-size 2G -a \
    "--output_dir ./wav2vec2-korean-1
    --model_name_or_path facebook/wav2vec2-large-xlsr-53
    --num_train_epochs 10
    --per_device_train_batch_size 6
    --per_device_eval_batch_size 6
    --evaluation_strategy steps 
    --save_strategy no
    --save_total_limit 2 
    --preprocessing_num_workers 4
    --dataloader_num_workers 4
    --group_by_length True
    --length_column_name length
    --freeze_feature_extractor True 
    --mode train
    --gradient_accumulation_steps 20
    --eval_accumulation_steps 500
    --writer_batch_size 500
    --data_type 2
    --disable_tqdm True
    --load_external_data True
    --use_external_data False
    --gdrive_code 4/0AX4XfWiMTTcmgQ2wPW67jWwLEOlLcFg5TQCr7Pu9ZBor7hxExsNXCe0jIWh3VSXSnEGJhw
    " 
