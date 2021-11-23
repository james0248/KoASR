import re
import gdown
import pandas as pd
import os
from pathlib import Path
import nsml
import shutil
import requests
from urllib import parse
import time
import os
from datasets import Dataset

from data import prepare_dataset

train_ids = [
    "1mAxVgZYFZYwgZS92c7qzOtCEM-5REEHv",
    "1m5uxPJxdVHSdMQsZpIdeLwlXt3TxtLxJ",
    "1M0f9RCzuJYf_jTOebda7_4YQunGsbinS",
    "1sdLfEsbPC1-YJVCc2ZNMIZWxzZd4ILwJ",
    "1HVG7pSKyBc0DpqudEMbrlXZ9pjnqD18l",
    "1TcDIL4BncyvAdJtNCl5o4FKUZrT5YnAa",
    "1NgA40b13DqaqYYPdntKUPfbeVUVnFd62",
    "1Vi7VX9DhEtfNVtWCvr0M985-gyfBatjB",
    "1BQGzdy-5PE5O7qI_JQEU4sBmYFcaNFGH",
    "16iJ6LSsYdTSBX4AaqlMpIS1JJItZXGQu",
    "1qIQqWr1aECq8koH-A5bw02Trc7i5R9iv",
    "1aZIz3VyOWjaWCMeqHJtFbJ30br9bkJzL",
    "1TvIr9H6PxwXeJBSAmfijBM_gaIxCOzpc",
    "1mWD39GFh-aybP113qqEr_IsZShy56ybL",
    "1wt6J7gL_3si9xpeH2Uu7CLJro4V92OHP",
    "1MUbXbOwvbNpETAt05wL0uk3yDw7pEfUL",
    "1im14Fs7koNzCwCS20W1JqBxFODYq42BZ",
    "1bvos60SmpdtD6Y9Hro9_Tp1T9WwV1-NP",
    "1HJis4SehdjxiZvoodNZ1ZyEeoaO9m0B1",
    "1hVYk-hsXlr1hgIXjCyIMuE9sOQNcOkxe",
    "1svgp5bOik8_vnemj32lj0zV2b2pJqHMO",
    "1V-a8B8U-fZXCexyNg_78rPQwUGmIdwGd",
    "12PmyJRKpBHcDeNNrW_8CMfAWzBrsg9aL",
    "1FjolWaNPxMlwv2jL_7iav4sd7m-VYrFY",
    "1q9KHNLGqfATa9HG_oQNC6I6z_uq3svgN",
    "1jXCZ71K1QwINsNsKzrA-9F6cuzW086cx",
    "1xj4ibpMND399U9DjAapGseCkZsZYJTVK",
    "1WrrAqyCbgeTKET24ycdCeJ7j-eYyzhu1",
    "19HybaPmEZ-AZvGh1ix_b1a_R2ZPeIA3N",
    "1sWaRge-4TgCGPgI81IAmAba8Oxcw94Y2",
    "1d0II93SNCBR3w0Ei7T0woQ4KlC_gaHSE",
    "1DNyvu2dkgpawjYxrYq9Tomrm_4SxOihF",
]
train_names = [
    "train_broadcast_01.tar.gz",
    "train_broadcast_02.tar.gz",
    "train_broadcast_03.tar.gz",
    "train_broadcast_04.tar.gz",
    "train_broadcast_05.tar.gz",
    "train_dialog_01.tar.gz",
    "train_dialog_02.tar.gz",
    "train_dialog_03.tar.gz",
    "train_dialog_04.tar.gz",
    "train_economy_01.tar.gz",
    "train_economy_02.tar.gz",
    "train_economy_03.tar.gz",
    "train_economy_04.tar.gz",
    "train_hobby_01.tar.gz",
    "train_label.tar.gz",
    "train_life_01.tar.gz",
    "train_life_02.tar.gz",
    "train_life_03.tar.gz",
    "train_life_04.tar.gz",
    "train_life_05.tar.gz",
    "train_life_06.tar.gz",
    "train_life_07.tar.gz",
    "train_life_08.tar.gz",
    "train_life_09.tar.gz",
    "train_life_10.tar.gz",
    "train_play_01.tar.gz",
    "train_play_02.tar.gz",
    "train_shopping_01.tar.gz",
    "train_shopping_02.tar.gz",
    "train_weather_01.tar.gz",
    "train_weather_02.tar.gz",
    "train_weather_03.tar.gz",
]
val_ids = [
    "1HAX3DmFKRXwFpi1iCNJVI1ysf_ZC_Vet",
    "1i7cgTei6nHEZN-rEy4_YXuyJ5vg08ego",
    "1NvmR9kNXyK35st5XDmpLycKrlPNzbqG9",
    "1WM7Ba0bLAKYmdeJRx3A9kbnityEEBL9O",
    "1LW8I8-VYWPD9LxZFC8gqQH8Smjug-om3",
    "1f94Nci9akLgh87JEONEpXsTWi88ZnseJ",
    "1FqtlvHH7MatIBkAp4Nr3eUI6oQLXamgi",
    "1DyOJFHL7rebtU5DEPqfmLVHZMoE6A5_-",
    "17V1wgv9y7LoWcgLx0tnDsjsOpptYDcQC",
]
val_names = [
    "val_broadcast_01.tar.gz",
    "val_dialog_01.tar.gz",
    "val_economy_01.tar.gz",
    "val_hobby_01.tar.gz",
    "val_label.tar.gz",
    "val_life_01.tar.gz",
    "val_play_01.tar.gz",
    "val_shopping_01.tar.gz",
    "val_weather_01.tar.gz",
]

# ------------------------ For testing ---------------------------
# val_ids = [
#     "14Xu6MjCCn2FWECjxCQhktynoBJsb-Xwj",
#     "1OZGTgCeoS_fTpT0jcKQQNRyotIkzBIq7"
# ]
# val_names = [
#     "val.tar.gz",
#     "label.tar.gz"
# ]
# train_ids = [
#     "1t_bfjUmeCxV4S8SgpfugzCNFauMTIyRE",
#     "1WCKh9-E06l2m_sf3gINTZ9H_xljECUxY",
#     # "1dJqagEaXTUihM02wruZEPhZSQcpXeJMC"
# ]
# train_names = [
#     "train.tar.gz",
#     "label.tar.gz",
#     # "extra_label.tar.gz"
# ]


# Not in use
class DatasetWrapper():
    def __init__(self, dataset):
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset


def bind_file(file):
    def save(dir_name, *parser):
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'checkpoint')
        os.makedirs(save_dir, exist_ok=True)
        os.system(
            f"cp {str(Path(file))} {str(Path(save_dir) / (Path(file)))}")
        print("파일 저장 완료!")

    def load(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'checkpoint')
        print(save_dir)
        os.system(
            f"cp {str(Path(save_dir) / (Path(file)))} {str(Path(file))}")
        print("파일 로딩 완료!")
    nsml.bind(save=save, load=load)


def bind_dataset(train: DatasetWrapper, val: DatasetWrapper):
    def save(dir_name, *parser):
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'external_data')
        os.makedirs(save_dir, exist_ok=True)
        train.dataset.to_parquet(os.path.join(save_dir, 'train_dataset'))
        val.dataset.to_parquet(os.path.join(save_dir, 'val_dataset'))
        print("데이터셋 저장 완료!")

    def load(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'external_data')
        train.set_dataset(Dataset.from_parquet(
            os.path.join(save_dir, 'train_dataset')))
        val.set_dataset(Dataset.from_parquet(
            os.path.join(save_dir, 'val_dataset')))
        print("데이터셋 로딩 완료!")

    nsml.bind(save=save, load=load)


def download_gdrive(token, file_id, path):
    print(f'Downloading {path}: {file_id}')
    os.system(
        f"curl -H 'Authorization: Bearer {token}' 'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media' -o {path}")


def get_access_token(code):
    client_id = '98428905431-v81lasstvu3fohhhk2hn7ti2ecj0g7a8.apps.googleusercontent.com'
    client_secret = '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'
    code = parse.quote(code)
    client_id = parse.quote(client_id)
    client_secret = parse.quote(client_secret)

    # Working!!
    # os.system(
    #     f'curl -X POST -d \
    #         "code={code}&client_id={client_id}&client_secret={client_secret}&redirect_uri=https%3A//localhost&grant_type=authorization_code" \
    #             -H "Content-Type: application/x-www-form-urlencoded" \
    #             "https://oauth2.googleapis.com/token"')
    data = f'code={code}&client_id={client_id}&client_secret={client_secret}&redirect_uri=https%3A//localhost&grant_type=authorization_code'
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(
        'https://oauth2.googleapis.com/token', data=data, headers=headers)
    data = response.json()
    return data['access_token'], data['refresh_token']


def refresh_token(refresh):
    client_id = '98428905431-v81lasstvu3fohhhk2hn7ti2ecj0g7a8.apps.googleusercontent.com'
    client_secret = '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'
    client_id = parse.quote(client_id)
    client_secret = parse.quote(client_secret)
    refresh = parse.quote(refresh)

    data = f'client_id={client_id}&client_secret={client_secret}&refresh_token={refresh}&grant_type=refresh_token'
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(
        'https://oauth2.googleapis.com/token', data=data, headers=headers)
    data = response.json()
    return data['access_token']


def download_aihub_file(code):
    train_dir = Path('./train')
    train_dir.mkdir(exist_ok=True)
    val_dir = Path('./val')
    val_dir.mkdir(exist_ok=True)
    extract_dir = Path('./extract')
    extract_dir.mkdir(exist_ok=True)
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    # get code from this url
    # https://accounts.google.com/o/oauth2/v2/auth?scope=https%3A//www.googleapis.com/auth/drive.readonly&access_type=offline&include_granted_scopes=true&redirect_uri=https%3A//localhost&response_type=code&client_id=98428905431-v81lasstvu3fohhhk2hn7ti2ecj0g7a8.apps.googleusercontent.com
    token, refresh = get_access_token(code)
    print(f'GOT TOKEN: {token}')
    print(f'GOT REFRESH_TOKEN: {refresh}')

    start = time.time()
    for ids, name in list(zip(train_ids, train_names)):
        download_gdrive(token, ids, str(train_dir/name))
        if (time.time() - start > 3000):
            token = refresh_token(refresh)
            print(f'NEW TOKEN: {token}')
            start = time.time()

    for ids, name in list(zip(val_ids, val_names)):
        download_gdrive(token, ids, str(val_dir/name))
        if (time.time() - start > 3000):
            token = refresh_token(refresh)
            print(f'NEW TOKEN: {token}')
            start = time.time()

    print(f"ls -l {str(train_dir)}")
    os.system(f"ls -l {str(train_dir)}")
    print(f"ls -l {str(val_dir)}")
    os.system(f"ls -l {str(val_dir)}")

    for gzfile in train_dir.iterdir():
        print(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
        os.system(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
    print(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")
    os.system(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")

    for gzfile in val_dir.iterdir():
        print(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
        os.system(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
    print(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")
    os.system(
        f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/*')} ./data")

    print(f"ls -l {str(data_dir)}")
    os.system(f"ls -l {str(data_dir)}")

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
    df[['path', 'text']] = df['raw'].str.split(' :: ', expand=True)

    # df['text'] = df['raw_text'].map(clean_label)
    # df['bad'] = df['text'].map(
    #     lambda x: len(x)-len(re.findall("[ㄱ-ㅎ|ㅏ-ㅣ|가-힣|.?! ]", x)) > 0
    #     or len(x) < 10 or len(x) > 50 or len(re.findall("[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]", x)) == 0
    # )
    df['path'] = df['path'].map(
        lambda row: os.path.join('./data', row[1:]))
    # print(df['bad'].sum())
    # print(df[df['bad']]['text'][:5])
    # clean_df = df[df['bad'] == False][['path', 'text']]
    # print(f'Total dataset length {len(clean_df)}')
    return df


def get_external_data(processor, args):
    # Download AI Hub data
    print("Download aihub data...")
    download_aihub_file(args.gdrive_code)
    print("Download complete!")

    print("Parse labels from text file")
    # data/1.Training/1.라벨링데이터/1.방송/broadcast_01/broadcast_01_scripts.txt
    # data/2.Validation/1.라벨링데이터/1.방송/broadcast_01/broadcast_01_scripts.txt
    train_label_path = Path(os.path.join('./data', '1.Training', '1.라벨링데이터'))
    train_df = pd.DataFrame(columns=['path', 'text'])
    for subject in train_label_path.iterdir():
        for dataset in subject.iterdir():
            for txt_file in dataset.iterdir():
                if('scripts.txt' in txt_file.name):
                    dataset_df = parse_label(txt_file)
                    train_df = train_df.append(dataset_df)

    train_df = train_df.drop_duplicates(subset='text')
    print(train_df['text'].value_counts().value_counts())

    val_label_path = Path(os.path.join('./data', '2.Validation', '1.라벨링데이터'))
    val_df = pd.DataFrame(columns=['path', 'text'])
    for subject in val_label_path.iterdir():
        for dataset in subject.iterdir():
            for txt_file in dataset.iterdir():
                if('scripts.txt' in txt_file.name):
                    dataset_df = parse_label(txt_file)
                    val_df = val_df.append(dataset_df)

    val_df = val_df.drop_duplicates(subset='text')
    print(val_df['text'].value_counts().value_counts())

    train_dataset, val_dataset = prepare_dataset(
        None, train_df, processor, args, val_df=val_df)
    print("Changing to dataset done.")

    print("Cleaning data...")
    shutil.rmtree('./data')
    print("Cleaning done!")

    return train_dataset, val_dataset


def download_kenlm():
    id6 = '1hARkNXFOFiHcy9DVWV6olwodWj0xJX7H'
    id10 = '1talCLBaHYLpkIrPMfff8Cm2ETTfBQYsE'
    path6 = Path('./model6.arpa')
    path10 = Path('./model10.arpa')
    gdown.download(id=id6, output=str(path6), use_cookies=False)
    gdown.download(id=id10, output=str(path10), use_cookies=False)
    print("Download model complete!")
    print("Starting save...")
    bind_file(str(path6))
    nsml.save(6)
    bind_file(str(path10))
    nsml.save(10)
    print("Save complete on 6 and 10!")
    os.system(f'rm {str(path6)}')
    os.system(f'rm {str(path10)}')
    exit(0)
