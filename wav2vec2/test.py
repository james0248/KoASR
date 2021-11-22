from hangul_utils import split_syllables
import tqdm
import sys
import re
import os
import requests
import json
from urllib import parse
from pathlib import Path
from pprint import pprint


class DatasetWrapper():
    def __init__(self, dataset):
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset


def get_access_token(code):
    client_id = '98428905431-v81lasstvu3fohhhk2hn7ti2ecj0g7a8.apps.googleusercontent.com'
    client_secret = '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'
    code = parse.quote(code)
    client_id = parse.quote(client_id)
    client_secret = parse.quote(client_secret)
    redirect_url = parse.quote("https://localhost")
    grant = parse.quote("authorization_code")

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
    print(response.text)
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
    ",": 54,
    "?": 55,
    ".": 56,
    "!": 57
}


def extract_text():
    with open('./data/scripts.txt') as script:
        with open('./data/scripts_orig.txt', 'w') as text:
            scripts = script.readlines()
            res = list(map(lambda s: s.split(' :: ')[1], scripts))
            text.writelines(res)


def print_notkor(s: str, pat: str):
    if re.search(pat, s) == None:
        return
    print(f'{s}: ', end='')
    for x in re.finditer(pat, s):
        print(x.group(0), end=', ')
    print()


# Filter
# #, a-zA-Z, 0-9, ()
# length: < 13


def clean_label(s: str):
    s = re.sub(pattern=r"[a-zA-Z]/|/[a-zA-Z]+", repl='', string=s)
    s = re.sub(pattern="\([^/()]+\)\/\([^/()]+\)",
               repl=lambda match: match.group(0).split('/')[1][1:-1], string=s)
    s = re.sub(
        pattern=r"[₂₁…州♡~<>:‘’“”ㆍ^`;​/+*​​=♤&@「」{}\u0200\[\]\-\\\"\']", repl='', string=s)
    s = s.replace('%', '퍼센트')
    s = s.replace('フ', 'ㄱ')
    s = s.replace('．', '.')
    s = s.replace('！', '!')
    s = s.replace('¸', ',')
    s = s.replace('·', ' ')
    s = re.sub(pattern="[.,!?][^\s.\n0-9]",
               repl=lambda match: '. ' + match.group(0)[1], string=s)
    s = re.sub(pattern="\s+", repl=' ', string=s)
    s = re.sub(pattern="\s+[.]", repl='.', string=s)
    s = re.sub(pattern="[.]+", repl='.', string=s)
    s = s.strip()
    return s


def preprocess():
    with open('./data/scripts_orig.txt') as script:
        with open('./data/scripts_clean.txt', 'w') as text:
            scripts = script.readlines()
            res = list(set(scripts))
            res = list(map(clean_label, res))
            res = list(filter(lambda x: re.search(
                '[^가-힣\s.,!?]', x) == None, res))
            res = list(set(res))
            res = list(filter(lambda s: len(s) > 9, res))
            text.write('\n'.join(res) + '\n')


def gen_train_data():
    with open('./data/scripts_clean.txt') as script, open('./data/order.txt') as order:
        with open('./data/train_data.txt', 'w') as text:
            scripts = script.readlines()
            orders = order.readlines()
            res = list(set(scripts + orders))
            res = list(filter(lambda x: re.search(
                '[^가-힣\s.,!?]', x) == None, res))
            res = list(map(lambda s: ' '.join(list(split_syllables(
                s).strip().replace(' ', '|'))), res))
            text.write('\n'.join(res) + '\n')


if __name__ == "__main__":
    print(Path('./data').is_file())
