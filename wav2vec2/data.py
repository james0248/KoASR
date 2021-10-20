import pandas as pd
import re
import json
import librosa
from hangul_utils import join_jamos
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk

from datasets.utils.logging import set_verbosity_error, set_verbosity_info
set_verbosity_error()   #disable logging

from nsml import DATASET_PATH
import os


def init_data():
    vocab = {
        "[PAD]": 0,
        "[UNK]": 3,
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
        "ㄿ": 53,
        "ㅀ": 54,
        "ㅄ": 55,
    }
    os.mkdir('./kowav-processor')
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

    return clean_token_list


def decode_CTC(token_list, processor):
    clean_token_list = remove_duplicate_tokens(token_list, processor)
    raw_char_list = list(map(processor.convert, clean_token_list))
    joined_string = join_jamos(''.join(raw_char_list))
    return list(joined_string)


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”]'
    batch["text"] = re.sub(chars_to_ignore_regex, '',
                           batch["text"]).lower() + " "
    return batch


def prepare_dataset(file_list, df, args, val_size=0.1):
    if args.mode == 'train':
        df['path'] = df['file_name'].apply(
            lambda row: os.path.join(DATASET_PATH, 'train', 'train_data', row))
        # df['text_split'] = df['text'].apply(split_syllables)
        # df = df.loc[:1000, :]
        print(len(df))
        print(df.head())
        
        train, val = train_test_split(df, test_size=val_size)

        train_data = Dataset.from_pandas(train) # THESE HAD TO BE USED VERY CAREFULLY.
        val_data = Dataset.from_pandas(val) #  IT LOADS EVERYTHING AFTER THIS IN MEMORY!!!

        train_data = train_data.map(
            remove_special_characters,
            num_proc=args.preprocessing_num_workers,
        )
        val_data = val_data.map(
            remove_special_characters,
            num_proc=args.preprocessing_num_workers,
        )

        # first save this files to disk and reload
        train_data.save_to_disk('./train_temp')
        val_data.save_to_disk('./val_temp')

        train_data = load_from_disk('./train_temp')
        val_data = load_from_disk('./val_temp')

        # change data to array
        print("Start changing to array")
        global count
        count = 0

        train_data = train_data.map(
            map_to_array,
            remove_columns=train_data.column_names,
            num_proc=args.preprocessing_num_workers,
            with_indices= True
        )
        val_data = val_data.map(
            map_to_array,
            remove_columns=val_data.column_names,
            num_proc=args.preprocessing_num_workers,
            with_indices= True
        )

        print("Finished changing to array")
        


        # print("Start Saving to Disk")
        # train_data_path = "./train_data"
        # val_data_path = "./val_data"
        
        # train_data.save_to_disk(train_data_path)
        # val_data.save_to_disk(val_data_path)

        # del train_data
        # del val_data

        # print("Saved to Disk")
        
        #set_verbosity_info()

        return train_data, val_data

    else:
        data = pd.DataFrame({'file_name': file_list})
        data['path'] = data['path'].apply(
            lambda row: os.path.join(DATASET_PATH, 'test', 'test_data', row))
        data['target_text'] = None
        test_data = Dataset.from_pandas(data)
        print("Start changing to array")
        test_data.map(
            map_to_array,
            remove_columns=test_data.column_names,
        )
        print("Finished changing to array")

        return test_data


def map_to_array(batch, index):
    data, sampling_rate = librosa.load(batch['path'], sr=None)
    resampled_data = librosa.resample(data, sampling_rate, 16_000, res_type='kaiser_fast')
    batch['data'] = resampled_data
    batch['sampling_rate'] = 16_000
    batch['target_text'] = batch['text']
    del resampled_data, data
    if(index<113347//8 and index%100==0):
        print(index)
    return batch


# class KoWavDataset(Dataset):
#     def __init__(self, file_list, target_list, mode='train'):
#         self.mode = mode

#         if self.mode == 'train':
#             self.path_list = np.array([
#                 os.path.join(DATASET_PATH, 'train', 'train_data', files)
#                 for files in file_list
#             ])
#             self.text_list = target_list
#         else:
#             self.path_list = np.array([
#                 os.path.join(DATASET_PATH, 'test', 'test_data', files)
#                 for files in file_list
#             ])

#     def __len__(self):
#         return len(self.path_list)

#     def __getitem__(self, i):
#         data, sampling_rate = sf.read(self.path_list[i])
#         if self.mode == 'train':
#             text = self.text_list[i]
#             return {
#                 'data': torch.tensor(data, dtype=torch.float32),
#                 'text': torch.tensor(text, dtype=torch.long)
#             }
#         else:
#             return {'data': torch.tensor(data, dtype=torch.float32)}
