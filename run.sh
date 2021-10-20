sleep 1m # Be patient!
nsml run -e ./wav2vec2/new_main.py -d stt_1 \
    -c 8 -g 1 --memory 75G --shm-size 8G -a \
    "--output_dir ./wav2vec2-korean 
    --num_train_epochs 2 
    --per_device_train_batch_size 32 
    --per_device_eval_batch_size 32 
    --evaluation_strategy steps 
    --save_total_limit 3 
    --save_steps 700 
    --eval_steps 700 
    --learning_rate 5e-5 
    --warmup_steps 0 
    --model_name_or_path facebook/wav2vec2-large-xlsr-53 
    --fp16 True 
    --preprocessing_num_workers 8 
    --group_by_length True 
    --freeze_feature_extractor True 
    --mode train
    --split 0" 