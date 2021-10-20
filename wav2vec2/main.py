import argparse
import os
from glob import glob
import gc
import pickle

from transformers.models.wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from gpuinfo import get_gpu_info
import pprint

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

import nsml
from nsml import DATASET_PATH

from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from datasets import load_metric

from data import prepare_dataset

print('torch version: ', torch.__version__)


def map_to_result(batch):
    model.to("cuda")
    processor = dict_for_infer['processor']
    input_values = processor(batch["data"],
                             sampling_rate=batch["sampling_rate"],
                             return_tensors="pt").input_values.to("cuda")

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["text"] = processor.batch_decode(pred_ids)[0]

    return batch


def evaluate(model, input):
    model.to(device)
    # as the target is english, the first word to the transformer should be the
    # english start token.
    processor = dict_for_infer['processor']
    tokenizer = dict_for_infer['tokenizer']
    decoder_input = torch.tensor([tokenizer.convert_tokens_to_ids['[SOS]']] *
                                 input.size(0),
                                 dtype=torch.long).to(device)
    output = decoder_input.unsqueeze(1).to(device)

    result_list = []
    token_list = []

    for tokens in output:
        summary = tokenizer.convert(tokens)
        result_list.append(summary)
        token_list.append(tokens)

    return result_list, token_list


def train_step(batch_item, training):
    src = batch_item['magnitude'].to(device)
    tar = batch_item['target'].to(device)
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    if training is True:
        # transformer.train()
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output, _, _ = model([src, tar_inp, None])
            # output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        return loss, acc, round(lr, 10)
    else:
        # transformer.eval()
        model.eval()
        with torch.no_grad():
            output, _, _ = model([src, tar_inp, None])
            # output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)

        return loss, acc


def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, 0))
    loss_ = criterion(pred.permute(0, 2, 1), real)
    mask = torch.tensor(mask, dtype=loss_.dtype)
    loss_ = mask * loss_

    return torch.sum(loss_) / torch.sum(mask)


def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim=2))
    mask = torch.logical_not(torch.eq(real, 0))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = torch.tensor(accuracies, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)

    return torch.sum(accuracies) / torch.sum(mask)


def path_loader(root_path, is_test=False):

    if is_test:
        test_path = os.path.join(root_path, 'test')
        file_list = sorted(glob(os.path.join(test_path, 'test_data', '*')))

        return file_list

    if args.mode == 'train':
        train_path = os.path.join(root_path, 'train')
        file_list = sorted(glob(os.path.join(train_path, 'train_data', '*')))
        label = pd.read_csv(os.path.join(train_path, 'train_label'))

        return file_list, label


def save_checkpoint(checkpoint, dir):

    torch.save(checkpoint, os.path.join(dir))


def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'checkpoint')
        save_checkpoint(dict_for_infer, save_dir)

        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):

        save_dir = os.path.join(dir_name, 'checkpoint')

        global checkpoint
        checkpoint = torch.load(save_dir)

        model.load_state_dict(checkpoint['model'])

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), 'rb') as f:
            dict_for_infer = pickle.load(f)

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        device = checkpoint['device']
        test_file_list = path_loader(test_path, is_test=True)
        test_dataset = prepare_dataset(test_file_list, None, mode='test')
        test_data_loader = DataLoader(test_dataset, batch_size=10)
        result_list = []

        for step, batch in enumerate(test_data_loader):
            inp = batch['magnitude'].to(device)
            output, _ = evaluate(model, inp)
            result_list.extend(output)

        prob = [1] * len(result_list)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(prob, result_list))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def compute_metrics(pred):
    wer_metric = load_metric("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{
            "input_values": feature["input_values"]
        } for feature in features]
        label_features = [{
            "input_ids": feature["labels"]
        } for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    args = parser.parse_args()

    pprint.pprint(get_gpu_info())

    report_interval = 10
    total_step = -1
    print(f"nsml report interval = {report_interval}")

    epochs = args.epochs
    learning_rate = 5e-5  # 5e-5
    batch_size = 128  # 32
    device = torch.device("cuda:0")

    model = Wav2Vec2ForCTC.from_pretrained(
        "fleek/wav2vec-large-xlsr-korean",
        gradient_checkpointing=True,
        ctc_loss_reduction='mean',
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer))

    bind_model(model=model, parser=args)
    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == 'train':
        file_list, label = path_loader(DATASET_PATH)

        train_dataset, val_dataset = prepare_dataset(file_list, label)

        train_dataset.set_format('torch', columns=['data', 'target_text'])
        val_dataset.set_format('torch', columns=['data', 'target_text'])

        tokenizer = Wav2Vec2CTCTokenizer('./vocab.json',
                                         unk_token="[UNK]",
                                         pad_token="[PAD]",
                                         word_delimiter_token="|")

        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False)

        processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer)

        data_collator = DataCollatorCTCWithPadding(processor=processor,
                                                   padding=True)

        # load model from session checkpoint
        #nsml.load(checkpoint='0', session='nia1030/stt_1/5')
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

        valid_dataloader = DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        print("Training start")

        for epoch in range(args.epochs):
            gc.collect()
            total_train_loss, total_valid_loss = 0, 0
            total_train_acc, total_valid_acc = 0, 0

            training = True
            for step, batch in enumerate(train_dataloader):
                if (step == total_step): break

                batch_loss, batch_acc, lr = train_step(batch, training)
                total_train_loss += batch_loss
                total_train_acc += batch_acc
                if step == 0:
                    pprint.pprint(get_gpu_info())
                if step % report_interval == 0:
                    nsml.report(step=step,
                                batch_loss=batch_loss.detach().item())
                    print(
                        f"[{step}/{len(train_dataloader)}] nsml.report batch_loss = {batch_loss.detach().item()}"
                    )

            training = False
            for step, batch in enumerate(valid_dataloader):
                batch_loss, batch_acc = train_step(batch, training)
                total_valid_loss += batch_loss
                total_valid_acc += batch_acc

            print('=================loss=================')
            print(f'total_train_loss: {total_train_loss}')
            print(f'total_valid_loss: {total_valid_loss}')
            print('\n')

            print('=================acc=================')
            print(f'total_train_acc : {total_train_acc}')
            print(f'total_valid_acc : {total_valid_acc}')
            print(
                f'average_train_acc : {total_train_acc/len(train_dataloader)}')
            print(
                f'average_valid_acc : {total_valid_acc/len(valid_dataloader)}')
            print('\n')

            dict_for_infer = {
                'model': model.state_dict(),
                'epochs': epochs,
                'learning_rate': learning_rate,
                'processor': processor,
                'device': device
            }

            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
