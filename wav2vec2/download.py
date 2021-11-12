import gdown
import os
from pathlib import Path
def download_aihub_file():
    id_dict = {
        'hobby_01' :{
            'label' : '1e63HuvqmXIJrGAV48Cl45PcNkk24N69u',
            'data' : '1LWMK7zq5VSmbohX3WAz2nZEhSwXa9NGL',
        },
        'dialog_01' :{
            'label' : '1eqpLDV4qAUIbudh-qzsGj95HwYcBV7B9',
            'data' : '1zkV9W2Cww0nlapYvPcGqjPrmm7W0NyPx',
        }
    }
    datadir = Path('./data')
    datadir.mkdir(exist_ok=True)
    Path('./temp').mkdir(exist_ok=True)
    id = id_dict['hobby_01']['label']
    for dataset in id_dict.values():
        for key in ['label','data']:
            id = dataset[key]
            output = Path('./temp/data.tar.gz')
            # print(f"downloading {id} to {str(output)}")
            gdown.download(id = id, output = str(output))
            os.system(f"ls -l {str(output.parent)}")
            print(f"Start unzip .tar.gz")
            os.system(f"tar -zxf {str(output)} -C {str(output.parent)}")
            os.remove(str(output))
            os.system(f"ls -l {str(output.parent)}")
        
    os.system(f"mv {str(output.parent/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data')} .")
    print("now remove everything")
    os.system(f"rm -rf {str(output.parent)}")
    print("done!")
    #gdown.download_folder("https://drive.google.com/drive/folders/1P5hPyHEWRqUHtFkLnYZxE21usqlKxXsX", quiet=True, no_cookies = True)


import pandas as pd
import re
def repl_function(match):
    return match.group(0).split('/')[1][1:-1]
def clean_label(string):
    string = re.sub(pattern = "\([^/()]+\)\/\([^/()]+\)", repl = repl_function, string=string)
    string = re.sub(pattern="[+*/]",repl='',string=string)
    return string
def parse_label(path):
    df = pd.read_csv(path, header = None, names = ['raw'])
    df[['path', 'raw_text']] = df['raw'].str.split(' :: ', expand = True)
    
    df['text'] = df['raw_text'].map(clean_label)
    df['bad'] = df['text'].map(lambda x: len(re.findall("[^(ㄱ-ㅎ|ㅏ-ㅣ|가-힣|.?! )]",x))>0 or len(x)<10 or len(x)>50)
    # print(df['bad'].sum())
    # print(df[df['bad']]['text'][:5])
    clean_df = df[df['bad']==False][['path','text']]
    print(len(clean_df))
    return clean_df
def aihub_path_loader():
    download_aihub_file()
    label_path = Path('./data/1.Training/1.라벨링데이터/')
    df = pd.DataFrame(columns=['path','text'])
    for subject in label_path.iterdir():
        for dataset in subject.iterdir():
            for txtfile in dataset.iterdir():
                if('scripts.txt' in txtfile.name):
                    df1 = parse_label(txtfile)
                    df = df.append(df1)
    df = df.drop_duplicates(subset='text')
    print(df['text'].value_counts().value_counts())
    df['path'] = df['path'].apply(
            lambda row: os.path.join('./data', row[1:]))
    return None, df

if __name__=='__main__':
    # print(re.sub(pattern = "\([^/()]+\)\/\([^/()]+\)", repl = repl_function, string="아/ 초파리 땜에 (죽겠네)/(죽것네) 짜증 난 그 이놈의 초파리들은 어디서 나오는 (거야)/(겨)."))
    import shutil
    shutil.rmtree('./data')
    df = aihub_path_loader()
    print(df['path'][0])
    df['new_path'] = df['path'].apply(
            lambda row: os.path.join('./data', row[1:]))
    print(df['new_path'][0])
    
