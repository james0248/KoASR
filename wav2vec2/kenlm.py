import kss
import re
import sys
import tqdm
from hangul_utils import split_syllables


def repl_function(match):
    return match.group(0).split('/')[1][1:-1]


def repl_function_2(match):
    return '. ' + match.group(0)[1]


def clean_label(string: str):
    try:
        string = string.split(' :: ')[1]
    except:
        pass
    string = re.sub(pattern="\([^/()]+\)\/\([^/()]+\)",
                    repl=repl_function, string=string)
    string = re.sub(pattern="[a-zA-Z]", repl='', string=string)
    string = re.sub(pattern="[+*/]", repl='', string=string)
    string = re.sub(pattern="\.[^\s\.\n0-9]",
                    repl=repl_function_2, string=string)
    string = re.sub(pattern="\s+", repl=' ', string=string)
    string = re.sub(pattern="\s+\.", repl='.', string=string)
    string = re.sub(pattern="\.{2}", repl='.', string=string)
    string = string.replace('%', '퍼센트')
    string = string.replace('フ', 'ㄱ')
    string = string.replace('．', '.')
    string = string.replace('！', '!')
    string = string.replace('¸', ',')
    string = string.replace('·', ' ')
    string = re.sub(
        pattern=r"[₂₁…州♡~<>:‘’'“”ㆍ^`;​​​=♤&@「」{}#\u0200\[\]\-\\\"]", repl='', string=string)
    string = string.strip()
    try:
        if string[-1] == '\n':
            string = string[:-1]
        if string[0] == '\n':
            string = string[1:]
    except:
        None
    return string


def train(mode):
    if mode == 'terminal':
        for line in sys.stdin:
            sentence = split_syllables(line).replace(' ', '|')
            sentence = ' '.join(list(sentence))
            print(sentence)
    elif mode == 'file':
        with open('text.txt', 'r') as f, open('train.txt', 'w') as t:
            for line in tqdm.tqdm(f.readlines()):
                sentence = split_syllables(line).replace(' ', '|')
                sentence = ' '.join(list(sentence))
                t.write(sentence)


def prepare():
    with open('./scripts.txt') as script, open('./order.txt') as order:
        with open('./text.txt', 'w') as text:
            scripts = script.readlines()
            orders = order.readlines()
            res = list(set(orders + scripts))
            res = list(filter(lambda x: len(x) > 4, res))
            res = list(map(clean_label, res))
            res = list(filter(lambda x: re.search(
                '[0-9\(\)]', x) == None, res))
            res = list(filter(lambda x: len(x) > 5, res))
            text.write('\n'.join(res) + '\n')


if __name__ == "__main__":
    prepare()
    train('file')
