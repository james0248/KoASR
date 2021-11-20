import gc
import pandas as pd
import numpy as np
import re
import json
import librosa
from hangul_utils import join_jamos, split_syllables
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

from datasets.utils.logging import set_verbosity_error, set_verbosity_info

from nsml import DATASET_PATH
import os
import time

not_kor = {}


def init_data():
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "|": 4,
        "ㄱ": 5,
        "ㄴ": 6,
        "ㄷ": 7,
        "ㄹ": 8,
        "ㅁ": 9,
        "ㅂ": 10,
        "ㅅ": 11,
        "ㅇ": 12,
        "ㅈ": 13,
        "ㅊ": 14,
        "ㅋ": 15,
        "ㅌ": 16,
        "ㅍ": 17,
        "ㅎ": 18,
        "ㄲ": 19,
        "ㄸ": 20,
        "ㅃ": 21,
        "ㅆ": 22,
        "ㅉ": 23,
        "ㅏ": 24,
        "ㅐ": 25,
        "ㅑ": 26,
        "ㅒ": 27,
        "ㅓ": 28,
        "ㅔ": 29,
        "ㅕ": 30,
        "ㅖ": 31,
        "ㅗ": 32,
        "ㅘ": 33,
        "ㅙ": 34,
        "ㅚ": 35,
        "ㅛ": 36,
        "ㅜ": 37,
        "ㅝ": 38,
        "ㅞ": 39,
        "ㅟ": 40,
        "ㅠ": 41,
        "ㅡ": 42,
        "ㅢ": 43,
        "ㅣ": 44,
        "ㄳ": 45,
        "ㄵ": 46,
        "ㄶ": 47,
        "ㄺ": 48,
        "ㄻ": 49,
        "ㄼ": 50,
        "ㄽ": 51,
        "ㄾ": 52,
        'ㄿ': 53,
        "ㅀ": 54,
        "ㅄ": 55,
        ",": 56,
        "?": 57,
        ".": 58,
        "!": 59,
    }
    os.makedirs('./kowav-processor', exist_ok=True)
    with open('./kowav-processor/vocab.json', 'w') as vocab_file:
        json.dump(vocab, vocab_file)


def remove_duplicate_tokens(token_list, processor):
    prev_token = -1
    clean_token_list = []
    for token in token_list:
        if token == processor.tokenizer.convert_tokens_to_ids('[PAD]'):
            prev_token = -1
        elif token != prev_token:
            prev_token = token
            clean_token_list.append(token)

    return [clean_token_list]


def decode_CTC(token_list, processor):
    clean_token_list = remove_duplicate_tokens(token_list, processor)
    raw_char_list = list(map(processor.convert, clean_token_list))
    joined_string = join_jamos(''.join(raw_char_list))
    return joined_string


# Currently not in use
def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def split_and_remove_special_characters(batch):
    symbolic_words_regex = r'\([A-Z]*\:|\)'
    batch["text"] = re.sub(symbolic_words_regex, '', batch["text"])

    chars_to_ignore_regex = '[\-\;\:\"\“\%\‘\”]'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"])
    
    batch["text"] = split_syllables(batch["text"])
    batch["text"] = batch["text"] + " " + "</s>"
    return batch


def map_to_array(batch, idx):
    try:
        data, sampling_rate = librosa.load(batch['path'], sr=None)
    except:
        # Method 1
        with open(batch['path'], 'rb') as opened_pcm_file:
            buf = opened_pcm_file.read()
            pcm_data = np.frombuffer(buf, dtype='int16')
            data = librosa.util.buf_to_float(pcm_data, 2)
            sampling_rate = 16_000

        # Method 2
        # data = np.memmap(batch['path'], dtype = 'h', mode = 'r').astype(np.float)
        # sampling_rate = 16_000

    resampled_data = librosa.resample(data,
                                      sampling_rate,
                                      16_000,
                                      res_type='polyphase'
                                      )

    # truncate files longer than 240_000 = 15s(22 files in final_stt_2)
    if(len(resampled_data) > 240_500):
        print(f"Long file detected: length = {len(resampled_data)}")
        resampled_data = resampled_data[:240_000]
    batch['data'] = resampled_data
    batch['length'] = len(resampled_data)
    batch['sampling_rate'] = 16_000
    batch['target_text'] = batch['text']
    del resampled_data, data

    if (idx % 5000 == 0):
        print(idx)
        gc.collect()
    return batch


def preprocess_dataset(batch, processor):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["data"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids

    del batch["target_text"], batch["data"], batch["sampling_rate"]
    gc.collect()
    return batch


def prepare_dataset(file_list, df, processor, args, val_size=0.1, val_df=None):
    if args.mode == 'train':

        set_verbosity_error()  # disable logging

        # Used for fast training (Only use some of the data)
        if args.split != None and args.max_split != None:
            length = int(len(df) / args.max_split)
            i = args.split
            df = df.iloc[i * length:(i + 1) * length, :]

        print(f"Number of soundfiles : {len(df)}")
        # print(df["text"][:50])
        # exit()

        # Initialize datasets
        train_data = Dataset.from_dict({})
        val_data = Dataset.from_dict({})

        if args.load_external_data:
            train_data = Dataset.from_pandas(
                df)  # THESE HAD TO BE USED VERY CAREFULLY.
            val_data = Dataset.from_pandas(
                val_df)  # IT LOADS EVERYTHING AFTER THIS IN MEMORY!!!
        else:
            train, val = train_test_split(df, test_size=val_size)

            train_data = Dataset.from_pandas(
                train)  # THESE HAD TO BE USED VERY CAREFULLY.
            val_data = Dataset.from_pandas(
                val)  # IT LOADS EVERYTHING AFTER THIS IN MEMORY!!!

        # first save this files to disk and reload
        train_data.save_to_disk('./train_temp')
        val_data.save_to_disk('./val_temp')

        train_data = load_from_disk('./train_temp')
        val_data = load_from_disk('./val_temp')

        # """
        train_data = train_data.map(
            split_and_remove_special_characters,
            num_proc=args.preprocessing_num_workers,
        )
        val_data = val_data.map(
            split_and_remove_special_characters,
            num_proc=args.preprocessing_num_workers,
        )
        # """
        # print(not_kor)
        # print(train_data[:10]['text'])

        # change data to array
        print("Start changing to array")
        tic = time.perf_counter()
        train_data = train_data.map(
            map_to_array,
            remove_columns=train_data.column_names,
            num_proc=args.preprocessing_num_workers,
            with_indices=True,
        )
        val_data = val_data.map(
            map_to_array,
            remove_columns=val_data.column_names,
            num_proc=args.preprocessing_num_workers,
            with_indices=True,
        )
        toc = time.perf_counter()
        print(f"Changing to array done in {toc-tic:.1f}s")

        # print("Filter long files")
        # This is slow, but is the only way to drop rows(not truncate)
        # tic = time.perf_counter()
        # train_data = train_data.filter(
        #     lambda length: [x < 240_000 for x in length],
        #     input_columns='length',
        #     num_proc=args.preprocessing_num_workers,
        #     batched=True
        # )
        # val_data = val_data.filter(
        #     lambda length: length < 240_000,
        #     input_columns='length',
        #     num_proc=args.preprocessing_num_workers,
        #     batched=True
        # )
        # toc = time.perf_counter()
        # print(f"Filter done in {toc-tic:.1f}s")
        # """
        print("Start preprocess")
        tic = time.perf_counter()

        train_data = train_data.map(
            preprocess_dataset,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            fn_kwargs={'processor': processor},
            writer_batch_size=args.writer_batch_size,
            batch_size=args.writer_batch_size
        )
        val_data = val_data.map(
            preprocess_dataset,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            fn_kwargs={'processor': processor},
            writer_batch_size=args.writer_batch_size,
            batch_size=args.writer_batch_size
        )
        toc = time.perf_counter()
        print(f"Preprocess done in {toc-tic:.1f}s")
        # """

        set_verbosity_info()

        return train_data, val_data

    else:
        set_verbosity_error()  # disable logging
        data = pd.DataFrame({'file_name': file_list})
        print(len(data))
        data['path'] = data['file_name'].apply(
            lambda row: os.path.join(DATASET_PATH, 'test_data', row))
        data['text'] = None
        test_data = Dataset.from_pandas(data)
        print("Start changing to array")
        test_data = test_data.map(
            map_to_array,
            remove_columns=test_data.column_names,
            num_proc=args.preprocessing_num_workers,
            with_indices=True,
        )
        print("Finished changing to array")
        set_verbosity_info()

        return test_data
