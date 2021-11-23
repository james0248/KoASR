import gc
import pandas as pd
import numpy as np
import re
import json
import librosa
from hangul_utils import split_syllables
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

from nsml import DATASET_PATH
import os
import time
from pathlib import Path


def init_data():
    old_vocab = {
        "[PAD]": 0, "[UNK]": 1, "|": 2, "ㄱ": 3, "ㄴ": 4, "ㄷ": 5, "ㄹ": 6, "ㅁ": 7, "ㅂ": 8, "ㅅ": 9,
        "ㅇ": 10, "ㅈ": 11, "ㅊ": 12, "ㅋ": 13, "ㅌ": 14, "ㅍ": 15, "ㅎ": 16, "ㄲ": 17, "ㄸ": 18,
        "ㅃ": 19, "ㅆ": 20, "ㅉ": 21, "ㅏ": 22, "ㅐ": 23, "ㅑ": 24, "ㅒ": 25, "ㅓ": 26, "ㅔ": 27,
        "ㅕ": 28, "ㅖ": 29, "ㅗ": 30, "ㅘ": 31, "ㅙ": 32, "ㅚ": 33, "ㅛ": 34, "ㅜ": 35, "ㅝ": 36,
        "ㅞ": 37, "ㅟ": 38, "ㅠ": 39, "ㅡ": 40, "ㅢ": 41, "ㅣ": 42, "ㄳ": 43, "ㄵ": 44, "ㄶ": 45,
        "ㄺ": 46, "ㄻ": 47, "ㄼ": 48, "ㄽ": 49, "ㄾ": 50, "ㄿ": 51, "ㅀ": 52, "ㅄ": 53, ",": 54,
        "?": 55, ".": 56, "!": 57
    }
    vocab = {
        "<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "|": 4, "ㄱ": 5, "ㄴ": 6, "ㄷ": 7, "ㄹ": 8,
        "ㅁ": 9, "ㅂ": 10, "ㅅ": 11, "ㅇ": 12, "ㅈ": 13, "ㅊ": 14, "ㅋ": 15, "ㅌ": 16, "ㅍ": 17,
        "ㅎ": 18, "ㄲ": 19, "ㄸ": 20, "ㅃ": 21, "ㅆ": 22, "ㅉ": 23, "ㅏ": 24, "ㅐ": 25, "ㅑ": 26,
        "ㅒ": 27, "ㅓ": 28, "ㅔ": 29, "ㅕ": 30, "ㅖ": 31, "ㅗ": 32, "ㅘ": 33, "ㅙ": 34, "ㅚ": 35,
        "ㅛ": 36, "ㅜ": 37, "ㅝ": 38, "ㅞ": 39, "ㅟ": 40, "ㅠ": 41, "ㅡ": 42, "ㅢ": 43, "ㅣ": 44,
        "ㄳ": 45, "ㄵ": 46, "ㄶ": 47, "ㄺ": 48, "ㄻ": 49, "ㄼ": 50, "ㄽ": 51, "ㄾ": 52, 'ㄿ': 53,
        "ㅀ": 54, "ㅄ": 55, ",": 56, "?": 57, ".": 58, "!": 59,
    }
    os.makedirs('./kowav-processor', exist_ok=True)
    with open('./kowav-processor/vocab.json', 'w') as vocab_file:
        # json.dump(old_vocab, vocab_file)
        json.dump(vocab, vocab_file)

# Currently not in use


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def clean_text(batch):
    '''
    preprocess text using regex & split syllables
    '''
    symbolic_words_regex = r'\([A-Z]*\:|\)'
    batch["text"] = re.sub(symbolic_words_regex, '', batch["text"])

    chars_to_ignore_regex = r'[\r\n\-\;\:\'\"\%\‘\’\“\”]'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"])

    # 띄어쓰기 안되어 있는거 해줌
    batch["text"] = re.sub(pattern="[.,!?][^\s.\n0-9]",
                           repl=lambda match: '. ' + match.group(0)[1], string=batch["text"])
    batch["text"] = re.sub(pattern="\s+", repl=' ', string=batch["text"])
    batch["text"] = re.sub(pattern="\s+[.]", repl='.', string=batch["text"])
    batch["text"] = re.sub(pattern="[.]+", repl='.', string=batch["text"])
    batch["text"] = batch["text"].strip()
    return batch


def clean_text_ext(batch):
    '''
    preprocess text using regex & split syllables for external data
    '''
    batch["text"] = re.sub(pattern=r"[a-zA-Z]/|/[a-zA-Z]+",
                           repl='', string=batch["text"])
    batch["text"] = re.sub(pattern="\([^/()]+\)\/\([^/()]+\)",
                           repl=lambda match: match.group(0).split('/')[1][1:-1], string=batch["text"])
    batch["text"] = re.sub(
        pattern=r"[₂₁…州♡~<>:‘’'“”ㆍ^`;​/+*​​=♤&@「」{}\u0200\[\]\-\\\"]", repl='', string=batch["text"])
    batch["text"] = batch["text"].replace('%', '퍼센트')
    batch["text"] = batch["text"].replace('フ', 'ㄱ')
    batch["text"] = batch["text"].replace('．', '.')
    batch["text"] = batch["text"].replace('！', '!')
    batch["text"] = batch["text"].replace('¸', ',')
    batch["text"] = batch["text"].replace('·', ' ')
    # 띄어쓰기 안되어 있는거 해줌
    batch["text"] = re.sub(pattern="[.,!?][^\s.\n0-9]",
                           repl=lambda match: '. ' + match.group(0)[1], string=batch["text"])
    batch["text"] = re.sub(pattern="\s+", repl=' ', string=batch["text"])
    batch["text"] = re.sub(pattern="\s+[.]", repl='.', string=batch["text"])
    batch["text"] = re.sub(pattern="[.]+", repl='.', string=batch["text"])
    batch["text"] = batch["text"].strip()
    return batch


def split_syllables_text(batch):
    batch["text"] = split_syllables(batch["text"])
    return batch


def not_long_file(batch):
    '''
    return True if file duration is not longer than max_length=15.5s
    '''
    max_length = 15.5
    if 'no_header' not in batch or not batch['no_header']:
        return librosa.get_duration(filename=batch['path']) < max_length
    else:
        return Path(batch['path']).stat().st_size // 2 < max_length * 16_000


def not_bad_string(batch):
    '''
    return True if label contains unknown vocab
    '''
    return re.search("[^가-힣\s.,!?]", batch["text"]) == None


def not_long_or_short_string(batch):
    '''
    return True if label is to short or long
    '''
    return len(batch["text"]) < 11 or len(batch["text"]) > 50


def file_exists(batch):
    '''
    return True if file exists in file system, in case of download errors
    '''
    return Path(batch['path']).is_file()


def map_to_array(batch, idx):
    '''Load audio from 'path' and resamples to 16kHz
    - handle two type of .wav files: with header/ without header(16bit, 16kHz)
    ## Input batch
    - path
    - text
    - no_header(optional)

    ## Output batch
    - data
    - length
    - sampling_rate
    - target_text
    '''
    if 'no_header' not in batch or not batch['no_header']:
        data, sampling_rate = librosa.load(batch['path'], sr=None)

    else:
        with open(batch['path'], 'rb') as opened_pcm_file:
            buf = opened_pcm_file.read()
            pcm_data = np.frombuffer(buf, dtype='int16')
            data = librosa.util.buf_to_float(pcm_data, 2)
            sampling_rate = 16_000

    batch["data"] = librosa.resample(data,
                                     sampling_rate,
                                     16_000,
                                     res_type='polyphase'
                                     )
    batch["target_text"] = batch["text"]
    batch['length'] = len(batch["data"])
    batch['sampling_rate'] = 16_000
    del data

    if (idx % 5000 == 0):
        print(idx)
        gc.collect()
    return batch


def preprocess_dataset(batch, processor):
    '''
    Normalize audio & convert tokens to ids
    ## Input batch
    - data
    - sampling_rate
    - target_text

    ## Output batch
    - input_values
    - labels
    '''
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["data"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids

    del batch["data"], batch["sampling_rate"]
    gc.collect()
    return batch


def prepare_dataset(file_list, df, processor, args, val_size=0.1, val_df=None):
    '''
    return train/val dataset or test dataset depending on args.mode
    ## Arguments
    - file_list : list of files. Used in test mode
    - df : dataframe containing columns "path", "text"
    - processor : processor to use in audio/text preprocessing
    - args : class DataTrainingArguments
    - val_size : validation split size (0~1)
    - val_df : use if external dataset

    ## Return
    dataset object with columns
    - input_values
    - labels
    '''
    if args.mode == 'train':

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

        if args.use_external_data:
            train_data = Dataset.from_pandas(
                df)  # THESE HAD TO BE USED VERY CAREFULLY.
            val_data = Dataset.from_pandas(
                val_df)  # IT LOADS EVERYTHING AFTER THIS IN MEMORY!!!
            train_data = train_data.filter(
                file_exists,
                num_proc=args.preprocessing_num_workers,
            )
            val_data = val_data.filter(
                file_exists,
                num_proc=args.preprocessing_num_workers,
            )
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

        print(f"Number of data before filter: {len(train_data)+len(val_data)}")

        if args.use_external_data:
            train_data = train_data.map(
                clean_text_ext,
                num_proc=args.preprocessing_num_workers,
            )
            val_data = val_data.map(
                clean_text_ext,
                num_proc=args.preprocessing_num_workers,
            )
            train_data = train_data.filter(not_long_or_short_string)
            val_data = val_data.filter(not_long_or_short_string)
        else:
            train_data = train_data.map(
                clean_text,
                num_proc=args.preprocessing_num_workers,
            )
            val_data = val_data.map(
                clean_text,
                num_proc=args.preprocessing_num_workers,
            )

        train_data = train_data.filter(not_bad_string)
        val_data = val_data.filter(not_bad_string)
        train_data = train_data.map(
            split_syllables_text,
            num_proc=args.preprocessing_num_workers)
        val_data = val_data.map(
            split_syllables_text,
            num_proc=args.preprocessing_num_workers)

        print(
            f"Number of data after text filter: {len(train_data)+len(val_data)}")

        train_data = train_data.filter(not_long_file)
        val_data = val_data.filter(not_long_file)

        print(
            f"Number of data after audio length filter: {len(train_data)+len(val_data)}")
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

        return train_data, val_data

    else:
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
        return test_data
