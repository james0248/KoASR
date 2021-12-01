# KoASR

Source code for the competition [한국어 2021 음성·자연어 인공지능 경진 대회](http://aihub-competition.or.kr/hangeul). Most of the code are written using the Hugginface [transformers](https://huggingface.co/transformers/) and [datasets](https://huggingface.co/docs/datasets/) library.

# Main pipeline

1. Data preprocessing
2. Wav2Vec2 + CTC makes an incomplete sentence containing some errors
3. GEC using BART as the base model fixes the incomplete sentence

# Code structure

## baseline

Basline code provided from the competition

## wav2vec2

Our code using [wav2vec2](https://arxiv.org/abs/2006.11477).

-   main.py

File handling all of the training

-   data.py

File handling all of hte data preprocessing. `prepare_dataset` function returns the preprocessed training dataset and validation dataset.

-   download.py

File handling download of external data from Google drive. Download is implemented via google drive API.

-   data_with_json.py

File handling preprocessing of data of the third dataset. Dataset format was in json file.

-   arguments.py

File containing all of the arguments definitions.

Other files are relatively less important or specific to the [NSML](https://ai.nsml.navercorp.com) platform.

## gec

Directory storing code for grammatical error correction. Basic sturcture is very similar to the wav2vec2 directory, except this one is specific to text data.
