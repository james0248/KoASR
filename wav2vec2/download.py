import re
from datasets.arrow_dataset import Dataset
import pandas as pd
import gdown
import os
from pathlib import Path
import nsml
import shutil

import os
from pathlib import Path

from scipy.sparse import data

from wav2vec2.data import prepare_dataset


class DatasetWrapper():
    def __init__(self, dataset):
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset(self) -> Dataset:
        return self.dataset


def bind_dataset(train: DatasetWrapper, val: DatasetWrapper):
    def save(dir_name, *parser):
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'external_data')
        os.makedirs(save_dir, exist_ok=True)
        train.get_dataset().to_parquet(save_dir + '/train_dataset')
        val.get_dataset().to_parquet(save_dir + '/val_dataset')
        print("데이터 저장 완료!")

    def load(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'external_data')
        train.set_dataset(Dataset.from_parquet(save_dir + '/train_dataset'))
        val.set_dataset(Dataset.from_parquet(save_dir + '/val_dataset'))
        print("데이터 로딩 완료!")

    nsml.bind(save=save, load=load)


def download_aihub_file():
    '''
    -data

    -gdown
        -A.tar.gz
        -B.tar.gz...

    -extract
        -data
            -remote ....
    -data.tar -> for save
    '''

    train_dir = Path('./train')
    train_dir.mkdir(exist_ok=True)
    val_dir = Path('./val')
    val_dir.mkdir(exist_ok=True)
    extract_dir = Path('./extract')
    extract_dir.mkdir(exist_ok=True)
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    # up to 50files

    # Training data: folder containes ~220GB
    gdown.download_folder(id="10DYHdoizNokrb6WIIi-d0o7xILgFe2NH",
                          quiet=False, use_cookies=False, output='./train')
    # Validation data: folder containes ~13GB
    gdown.download_folder(id="1vOdPuSCL7IjhGOYkzfO_aIxiKifGdnxd",
                          quiet=False, use_cookies=False, output='./val')

    print(f"ls -l {str(train_dir)}")
    os.system(f"ls -l {str(train_dir)}")
    print(f"ls -l {str(val_dir)}")
    os.system(f"ls -l {str(val_dir)}")

    # i = 0
    # for gzfile in train_dir.iterdir():
    #     print(f"checkpoint {100+i} binding to {str(gzfile)}")
    #     bind_dataset(str(gzfile))
    #     nsml.save(100+i)
    #     i += 1

    for gzfile in train_dir.iterdir():
        print(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
        os.system(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
    print(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")
    os.system(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")

    # for gzfile in train_dir.iterdir():
    #     print(f"checkpoint {100+i} binding to {str(gzfile)}")
    #     bind_dataset(str(gzfile))
    #     nsml.save(100+i)
    #     i += 1

    for gzfile in val_dir.iterdir():
        print(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
        os.system(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
    print(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")
    os.system(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")

    print(f"ls -l {str(data_dir)}")
    os.system(f"ls -l {str(data_dir)}")

    # for dataset in ['dialog_02','dialog_03','dialog_04','hobby_01']:
    #     try:
    #         for key in ['label']:
    #             id = id_dict[dataset][key]
    #             output = Path('./gdown/data.tar.gz')
    #             # print(f"downloading {id} to {str(output)}")
    #             gdown.download(id = id, output = str(output), use_cookies= False)
    #             os.system(f"ls -l {str(output.parent)}")
    #             print(f"Start unzip .tar.gz")
    #             os.system(f"tar -zxf {str(output)} -C {str(output.parent)}")
    #             os.remove(str(output))
    #             os.system(f"ls -l {str(output.parent)}")
    #     except:
    #         print("Error occured")
    #         pass
    # os.system(f"mv {str(output.parent/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data')} .")

    print("Remove downloaded tar files and decompressed files")
    shutil.rmtree(str(train_dir))
    shutil.rmtree(str(val_dir))
    shutil.rmtree(str(extract_dir))
    print("Done removing files")


def repl_function(match):
    return match.group(0).split('/')[1][1:-1]


def clean_label(string):
    string = re.sub(pattern="\([^/()]+\)\/\([^/()]+\)",
                    repl=repl_function, string=string)
    string = re.sub(pattern="[+*/]", repl='', string=string)
    return string


def parse_label(path):
    df = pd.read_csv(path, header=None, names=['raw'])
    df[['path', 'raw_text']] = df['raw'].str.split(' :: ', expand=True)

    df['text'] = df['raw_text'].map(clean_label)
    df['bad'] = df['text'].map(
        lambda x: len(x)-len(re.findall("[ㄱ-ㅎ|ㅏ-ㅣ|가-힣|.?! ]", x)) > 0
        or len(x) < 10 or len(x) > 50 or len(re.findall("[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]"), x) == 0
    )
    df['path'] = df['path'].map(
        lambda row: os.path.join('./data', row))
    # print(df['bad'].sum())
    # print(df[df['bad']]['text'][:5])
    clean_df = df[df['bad'] == False][['path', 'text']]
    print(f'Total dataset length {len(clean_df)}')
    return clean_df


def save_external_data(processor, args):
    # Download AI Hub data
    download_aihub_file()

    print("Parse labels from text file")
    # data/1.Training/1.라벨링데이터/1.방송/broadcast_01/broadcast_01_scripts.txt
    # data/2.Validation/1.라벨링데이터/1.방송/broadcast_01/broadcast_01_scripts.txt
    train_label_path = Path('./data/1.Training/1.라벨링데이터/*/*/')
    train_df = pd.DataFrame(columns=['path', 'text'])
    for txt_file in train_label_path.iterdir():
        if('scripts.txt' in txt_file.name):
            dataset_df = parse_label(txt_file)
            train_df = train_df.append(dataset_df)

    train_df = train_df.drop_duplicates(subset='text')
    print(train_df['text'].value_counts().value_counts())

    val_label_path = Path('./data/2.Validation/1.라벨링데이터/*/*/')
    val_df = pd.DataFrame(columns=['path', 'text'])
    for txt_file in val_label_path.iterdir():
        if('scripts.txt' in txt_file.name):
            dataset_df = parse_label(txt_file)
            val_df = val_df.append(dataset_df)

    val_df = val_df.drop_duplicates(subset='text')
    print(val_df['text'].value_counts().value_counts())

    train_dataset, val_dataset = prepare_dataset(
        None, train_df, processor, args, val_df=val_df)
    train_dataset_wrapper = DatasetWrapper(train_dataset)
    val_dataset_wrapper = DatasetWrapper(val_dataset)
    bind_dataset(train_dataset_wrapper, val_dataset_wrapper)

    print("Save external data to nsml checkpoint 1000...")
    nsml.save(1000)
    print("save complete! now you can safely exit this session")
    shutil.rmtree('./data')
