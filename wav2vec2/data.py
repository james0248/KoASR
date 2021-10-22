import pandas as pd
import re
import json
import librosa
from hangul_utils import join_jamos, split_syllables
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

from datasets.utils.logging import set_verbosity_error, set_verbosity_info

from nsml import DATASET_PATH
import os


def init_data():
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "|": 2,
        "ㄱ": 3,
        "ㄴ": 4,
        "ㄷ": 5,
        "ㄹ": 6,
        "ㅁ": 7,
        "ㅂ": 8,
        "ㅅ": 9,
        "ㅇ": 10,
        "ㅈ": 11,
        "ㅊ": 12,
        "ㅋ": 13,
        "ㅌ": 14,
        "ㅍ": 15,
        "ㅎ": 16,
        "ㄲ": 17,
        "ㄸ": 18,
        "ㅃ": 19,
        "ㅆ": 20,
        "ㅉ": 21,
        "ㅏ": 22,
        "ㅐ": 23,
        "ㅑ": 24,
        "ㅒ": 25,
        "ㅓ": 26,
        "ㅔ": 27,
        "ㅕ": 28,
        "ㅖ": 29,
        "ㅗ": 30,
        "ㅘ": 31,
        "ㅙ": 32,
        "ㅚ": 33,
        "ㅛ": 34,
        "ㅜ": 35,
        "ㅝ": 36,
        "ㅞ": 37,
        "ㅟ": 38,
        "ㅠ": 39,
        "ㅡ": 40,
        "ㅢ": 41,
        "ㅣ": 42,
        "ㄳ": 43,
        "ㄵ": 44,
        "ㄶ": 45,
        "ㄺ": 46,
        "ㄻ": 47,
        "ㄼ": 48,
        "ㄽ": 49,
        "ㄾ": 50,
        "ㄿ": 51,
        "ㅀ": 52,
        "ㅄ": 53,
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
    return list(joined_string)


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def split_and_remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”]'
    batch["text"] = split_syllables(batch["text"])
    batch["text"] = re.sub(chars_to_ignore_regex, '',
                           batch["text"]).lower() + " "
    return batch


def map_to_array(batch, index):
    data, sampling_rate = librosa.load(batch['path'], sr=None)
    resampled_data = librosa.resample(data,
                                      sampling_rate,
                                      16_000,
                                      res_type='kaiser_fast')
    batch['data'] = resampled_data
    batch['sampling_rate'] = 16_000
    batch['target_text'] = batch['text']
    # del resampled_data, data

    if (index % 1000 == 0):
        print(index)
    return batch


def preprocess_dataset(batch, processor):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["data"], sampling_rate=batch["sampling_rate"][0]).input_values

    batch["length"] = [len(x) for x in batch["input_values"]]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    #print(batch["target_text"])
    #print(batch["labels"])
    return batch


def prepare_dataset(file_list, df, processor, args, val_size=0.1):
    if args.mode == 'train':
        set_verbosity_error()  #disable logging

        df['path'] = df['file_name'].apply(
            lambda row: os.path.join(DATASET_PATH, 'train', 'train_data', row))
        if args.split != None and args.max_split != None:
            length = int(len(df) / args.max_split)
            i = args.split
            df = df.iloc[i * length:(i + 1) * length, :]
        # df['text_split'] = df['text'].apply(split_syllables)
        # df = df.loc[:30000, :]
        print(f"Number of soundfiles : {len(df)}")
        # print(df.head())

        train, val = train_test_split(df, test_size=val_size)

        train_data = Dataset.from_pandas(
            train)  # THESE HAD TO BE USED VERY CAREFULLY.
        val_data = Dataset.from_pandas(
            val)  #  IT LOADS EVERYTHING AFTER THIS IN MEMORY!!!

        train_data = train_data.map(
            split_and_remove_special_characters,
            num_proc=args.preprocessing_num_workers,
        )
        val_data = val_data.map(
            split_and_remove_special_characters,
            num_proc=args.preprocessing_num_workers,
        )

        # first save this files to disk and reload
        train_data.save_to_disk('./train_temp')
        val_data.save_to_disk('./val_temp')

        train_data = load_from_disk('./train_temp')
        val_data = load_from_disk('./val_temp')

        # change data to array
        print("Start changing to array")

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

        print("Start preprocess")

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

        set_verbosity_info()

        return train_data, val_data

    else:
        set_verbosity_error()  #disable logging
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
