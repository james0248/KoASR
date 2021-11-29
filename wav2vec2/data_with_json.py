import json
import re
import gc
from typing import Iterable, Union
from hangul_utils import split_syllables
import librosa
import os
import numpy as np
import torch
from hangul_utils import join_jamos
from ctcdecode import CTCBeamDecoder
from nsml import DATASET_PATH


def prepare_dataset_with_json(org_train_data, org_val_data, args, processor):
    '''
    Function used in final_stt_3
    '''
    print('func called')
    print(len(org_train_data))

    json_train_data = org_train_data.map(
        read_json, batched=True, batch_size=1, remove_columns=org_train_data.column_names)
    json_val_data = org_val_data.map(
        read_json, batched=True, batch_size=1, remove_columns=org_val_data.column_names)

    # print(set(unique_chars))
    print(len(json_train_data))
    print(len(json_val_data))

    wav_train_data = json_train_data.map(
        read_wav,
        num_proc=args.preprocessing_num_workers,
    )
    wav_val_data = json_val_data.map(
        read_wav,
        num_proc=args.preprocessing_num_workers,
    )
    train_data = wav_train_data.map(
        preprocess_dataset,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        fn_kwargs={'processor': processor},
        writer_batch_size=args.writer_batch_size,
        batch_size=args.writer_batch_size
    )
    val_data = wav_val_data.map(
        preprocess_dataset,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        fn_kwargs={'processor': processor},
        writer_batch_size=args.writer_batch_size,
        batch_size=args.writer_batch_size
    )
    print("size of ./train_temp")
    os.system('du -sh ./train_temp')
    print("size of ./val_temp")
    os.system('du -sh ./val_temp')

    return train_data, val_data


def clean_text(text: str) -> Union[str, None]:
    '''
    Clean text of stt_3
    returns cleaned text or None
    '''
    if len(re.findall(r'&.+&', text)) > 0:
        return None
    text = re.sub(r'{.*}|\(\(\)\)', '', text)
    text = re.sub(r'\(\(.+\)\)', lambda match: match.group(0)[2:-2], text)
    text = re.sub(r'\(.+\)\/\(.+\)',
                  lambda match: match.group(0).split('/')[0][1:-1], text)
    text = re.sub(r'-', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)

    if len(re.findall(r'[(){}\-&xX@]', text)) > 0:
        return None

    if len(text) <= 3:
        return None
    return text


unique_chars = []


def read_json(batch):
    with open(batch["json_path"][0]) as json_file:
        data = json.load(json_file)

    return_dict = {"path": [], "text": [], "start": [], "length": []}
    for record in data["utterance"]:
        duration = record["end"]-record["start"]
        if duration > 15.5 or duration < 3:
            continue
        cleaned_text = clean_text(record["dialect_form"])
        if not cleaned_text:
            continue
        # print(cleaned_text)
        unique_chars.extend(cleaned_text)
        return_dict["path"].append(batch["path"][0])
        return_dict["text"].append(split_syllables(cleaned_text))
        return_dict["start"].append(record["start"])
        return_dict["length"].append(duration)
        # print(record["form"]==record["standard_form"])
        # print(record["form"]==record["dialect_form"])
    return return_dict


sampling_rates = []


def read_wav(batch):
    data, sampling_rate = librosa.load(
        batch["path"], offset=batch["start"], duration=batch["length"], sr=None)
    sampling_rates.append(sampling_rate)
    batch["data"] = librosa.resample(data,
                                     sampling_rate,
                                     16_000,
                                     res_type='polyphase')
    assert len(batch["data"]) > 2
    return batch


def preprocess_dataset(batch, processor):
    '''
    Normalize audio & convert tokens to ids
    ## Input batch
    - data
    - text

    ## Output batch
    - input_values
    - labels
    '''
    input_values = processor(
        batch["data"], sampling_rate=16_000).input_values
    if not isinstance(input_values[0], np.ndarray):
        batch["input_values"] = [np.asarray(
            array, dtype=np.float16) for array in input_values]
    elif (
        not isinstance(input_values, np.ndarray)
        and isinstance(input_values[0], np.ndarray)
        and input_values[0].dtype is not np.dtype(np.float16)
    ):
        batch["input_values"] = [array.astype(
            np.float16) for array in input_values]
    elif isinstance(input_values, np.ndarray) and input_values.dtype is not np.dtype(np.float16):
        batch["input_values"] = input_values.astype(np.float16)

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids

    del batch["data"]
    gc.collect()
    return batch


def predict_dialect(file, model, processor, device):
    '''
    return a concatenated string.
    '''
    model.eval()
    model.to(device)
    file = os.path.join(DATASET_PATH, 'test_data', file)
    total_length = librosa.get_duration(filename=file)
    length_per_batch = 15
    starts = np.arange(0, total_length, length_per_batch)
    data_list = []
    for start in starts:
        data, sampling_rate = librosa.load(file, offset=start, duration=min(
            length_per_batch, total_length-start), sr=None)
        data = librosa.resample(data,
                                sampling_rate,
                                16_000,
                                res_type='polyphase')
        data_list.append(data)
    batch_size = 4
    result_list = []
    decoder = CTCBeamDecoder(
        list(processor.tokenizer.get_vocab().keys()),
        # model_path='./model.arpa',
        model_path=None,
        alpha=0.1,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=8,
        blank_id=0,
        log_probs_input=True  # No softmax layer in Wav2Vec2ForCTC
    )
    for iter in range((len(data_list)-1)//batch_size + 1):
        input_values = processor(
            data_list[iter*batch_size:(iter+1)*batch_size],
            padding=True,
            sampling_rate=16_000,
            return_tensors="pt"
        ).input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
        pred_ids = [beam_results[i][0][:out_lens[i][0]]
                    for i in range(out_lens.shape[0])]
        decoded_strings = processor.batch_decode(pred_ids, group_tokens=False)

        for i in range(out_lens.shape[0]):
            pred_str = join_jamos(decoded_strings[i])
            pred_str = re.sub('<unk>|<s>|<\/s>|[ㄱ-ㅎ|ㅏ-ㅣ]|�', '', pred_str)
            result_list.append(pred_str)
    # print(f"debug : {result_list}")
    result_str = " ".join(result_list)
    result_str = re.sub(r' {2,}', ' ', result_str)
    # print(f"debug : {result_str}")
    return result_str
