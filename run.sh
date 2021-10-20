nsml run -e ./wav2vec2/new_main.py -d stt_1 \
    -c 16 -g 2 --memory 150G --shm-size 16G -a \
    "--output_dir ./wav2vec2-korean 
    --num_train_epochs 2 
    --per_device_train_batch_size 8 
    --per_device_eval_batch_size 8 
    --evaluation_strategy steps 
    --save_total_limit 3 
    --save_steps 700 
    --eval_steps 700 
    --learning_rate 0.0005 
    --warmup_steps 3000 
    --model_name_or_path facebook/wav2vec2-large-xlsr-53 
    --fp16 True 
    --preprocessing_num_workers 16 
    --group_by_length True 
    --freeze_feature_extractor True 
    --mode train
    --split 0
    --max_split 5" 