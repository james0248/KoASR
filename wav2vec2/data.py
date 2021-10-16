import pandas as pd
import re
import json
import soundfile as sf
from hangul_utils import split_syllables
from sklearn.model_selection import train_test_split
from datasets import Dataset
from nsml import DATASET_PATH
import os


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”]'
    batch["text"] = re.sub(chars_to_ignore_regex, '',
                           batch["text"]).lower() + " "
    return batch


def prepare_dataset(file_list, df, val_size=0.01, isTest=False):
    if isTest == True:
        data = pd.DataFrame({'file_name': file_list})
        df['path'] = data['path'].apply(
            lambda row: os.path.join(DATASET_PATH, 'test', 'test_data', row))
        data['text'] = None
        test_data = Dataset.from_pandas(data)
        test_data.map(speech_file_to_array,
                      remove_columns=test_data.column_names)

        return test_data

    else:
        df['path'] = df['file_name'].apply(
            lambda row: os.path.join(DATASET_PATH, 'train', 'train_data', row))
        df['text'] = df['text'].apply(split_syllables)
        train, val = train_test_split(df, test_size=val_size)

        train_data = Dataset.from_pandas(train)
        val_data = Dataset.from_pandas(val)

        train_data = train_data.map(remove_special_characters)
        val_data = val_data.map(remove_special_characters)
        # generate vocab.json
        vocab_train = train_data.map(extract_all_chars,
                                     batched=True,
                                     batch_size=-1,
                                     keep_in_memory=True,
                                     remove_columns=train_data.column_names)
        vocab_val = val_data.map(extract_all_chars,
                                 batched=True,
                                 batch_size=-1,
                                 keep_in_memory=True,
                                 remove_columns=val_data.column_names)

        vocab_list = list(
            set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        print(vocab_dict)
        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        # change data to array
        train_data = train_data.map(speech_file_to_array,
                                    remove_columns=train_data.column_names)
        val_data = val_data.map(speech_file_to_array,
                                remove_columns=val_data.column_names)

        return train_data, val_data


def speech_file_to_array(batch):
    data, sampling_rate = sf.read(batch['path'])
    batch['data'] = data
    batch['sampling_rate'] = sampling_rate
    batch['target_text'] = batch['text']
    return batch
