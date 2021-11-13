import gdown
import os
from pathlib import Path
from gdown.download import download
import nsml

import os
from pathlib import Path
def bind_dataset(file):
    def save(dir_name, *parser):
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'checkpoint')
        os.makedirs(save_dir, exist_ok=True)
        os.system(f"cp {str(Path(file))} {str(Path(save_dir) / (Path(file).name))}")
        print("데이터 저장 완료!")
    def load(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'checkpoint')
        os.system(f"cp {str(Path(save_dir) / (Path(file).name))} {str(Path(file))}")
        print("데이터 로딩 완료!")
    nsml.bind(save=save,load=load)    

def download_aihub_file():
    id_dict = {
        'hobby_01' :{
            'label' : '1e63HuvqmXIJrGAV48Cl45PcNkk24N69u',
            'data' : '1LWMK7zq5VSmbohX3WAz2nZEhSwXa9NGL',
        },
        'dialog_01' :{
            'label' : '1eqpLDV4qAUIbudh-qzsGj95HwYcBV7B9',
            #'data' : '1zkV9W2Cww0nlapYvPcGqjPrmm7W0NyPx',
            'data' : '14M5_CoAstn4bwGPh22jxTUOdVnje1XYZ' #not working for 24hours....
        },
        'dialog_02':{
            'label' : '1asv4xOQFXE1YbwKQ-K_44jgrEnkLXObK',
            'data' : '18Ni90odB1q0NPyeyAzi3S7MIDqQPFCxZ',
        },
        'dialog_03':{
            'label' : '12olZ23q8JGjMxf29cSVHSo4Mtq7ri5fE',
            'data' : '1h9soOHXatigMrpE5Y_sSrVcWeA2mADsj',
        },
        'dialog_04':{
            'label' : '1Vuk4oFbqV9hrj7BJGaJlj3kNaHfKGtWy',
            'data' : '1FJ4c9b-G0dMNGSGjOk8Mkip34gLtlIPG',
        }
    }


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

    datadir = Path('./data')
    datadir.mkdir(exist_ok=True)
    download_dir = Path('./gdown')
    download_dir.mkdir(exist_ok=True)
    extract_dir = Path('./extract')
    extract_dir.mkdir(exist_ok=True)
    #up to 50files
    #folder containes ~220GB
    gdown.download_folder(id = "10DYHdoizNokrb6WIIi-d0o7xILgFe2NH", quiet=False, use_cookies = False, output = './gdown')
    print(f"ls -l {str(download_dir)}")
    os.system(f"ls -l {str(download_dir)}")
    i=0
    for gzfile in download_dir.iterdir():
        print(f"checkpoint {100+i} binding to {str(gzfile)}")
        bind_dataset(str(gzfile))
        nsml.save(100+i)
        i+=1
    for gzfile in download_dir.iterdir():
        print(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
        os.system(f"tar -zxf {str(gzfile)} -C {str(extract_dir)}")
    print(f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data')} .")
    os.system(f"mv {str(extract_dir/'data/remote/PROJECT/AI학습데이터/KoreanSpeech/data')} .")


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
    print("now remove everything")
    os.system(f"rm -rf {str(download_dir)}")
    print("done!")
    print("tar file to save")
    os.system(f"tar -cf data.tar data")


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
    df['bad'] = df['text'].map(
        lambda x: len(x)-len(re.findall("[ㄱ-ㅎ|ㅏ-ㅣ|가-힣|.?! ]",x))>0 \
            or len(x)<10 or len(x)>50 or len(re.findall("[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]"),x)==0
    )
    # print(df['bad'].sum())
    # print(df[df['bad']]['text'][:5])
    clean_df = df[df['bad']==False][['path','text']]
    print(len(clean_df))
    return clean_df
def aihub_path_loader():
    # Download only in the first session
    bind_dataset("./data.tar")

    first_download = False
    if first_download:
        download_aihub_file()
        print("save to nsml checkpoint 1000...")
        bind_dataset("./data.tar")
        nsml.save(1000)
        os.remove("data.tar")
        print("save complete! now you can safely exit this session")
    else:    
        print("Fetching dataset from checkpoint...")
        bind_dataset("./data.tar")
        nsml.load(checkpoint = '1000', session='nia1030/final_stt_1/**')
        os.system(f"tar -xf data.tar")
        
    print("parse labels")
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
    _, df = aihub_path_loader()
    print(df['path'][0])
    df['new_path'] = df['path'].apply(
            lambda row: os.path.join('./data', row[1:]))
    print(df['new_path'][0])
    
