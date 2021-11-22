#!/usr/bin/env python3
from ctcdecode import CTCBeamDecoder
import logging
from glob import glob
import pickle
import os
import shutil
from dataclasses import dataclass, field
from datasets.arrow_dataset import Dataset

from hangul_utils import join_jamos
import re
from gpuinfo import get_gpu_info
from typing import Any, Callable, Dict, List, Optional, Set, Union

import nsml
import datasets
import numpy as np
import pandas as pd
import torch
from packaging import version
from transformers.trainer_callback import TrainerControl, TrainerState

from transformers import (HfArgumentParser, Trainer,
                          Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor, is_apex_available,
                          trainer_utils, TrainerCallback, AutoTokenizer,
                          AutoModelForPreTraining)

from data import init_data, prepare_dataset
from download import DatasetWrapper, bind_dataset, download_kenlm, get_external_data
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from nsml import DATASET_PATH

import warnings

warnings.filterwarnings(action='ignore')

if is_apex_available():
    from apex import amp  # type: ignore

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


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


class NSMLCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        global dict_for_infer
        dict_for_infer = {
            'model': model.state_dict(),
            'epochs': state.epoch,
            'learning_rate': args.learning_rate,
            'tokenizer': tokenizer,
            'processor': processor,
            'device': device,
        }
        #nsml.save(int(state.epoch))

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics, **kwargs):
        report_dict = {
            'step': state.epoch,
            'eval_loss': metrics['eval_loss'],
            'wer': metrics['eval_wer'],
            'cer': metrics['eval_cer'],
        }
        nsml.report(**report_dict)

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if state.is_local_process_zero and 'loss' in logs:
            report_dict = {
                'step': state.epoch,
                'train_loss': logs['loss'],
                'learning_rate' : logs['learning_rate'], 
            }
            nsml.report(**report_dict)


def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))


def predict(test_dataset):
    model.to(device)
    model.eval()

    result_list = []
    decoder = CTCBeamDecoder(
        list(processor.tokenizer.get_vocab().keys()),
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=True  # No softmax layer in Wav2Vec2ForCTC
    )

    def map_to_result(batch):

        input_values = processor(
            batch["data"],
            sampling_rate=batch["sampling_rate"],
            return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        # pred_ids = torch.argmax(logits, dim=-1)
        # pred_ids = remove_duplicate_tokens(pred_ids.cpu().numpy()[0],
        #                                    processor)
        # pred_str = join_jamos(processor.batch_decode(pred_ids))

        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
        # select best predection
        pred_ids = beam_results[0][0][:out_lens[0][0]]

        decoded_str = processor.batch_decode(pred_ids)
        pred_str = ""
        for char in decoded_str[:-1]:
            pred_str += " " if char == "" else char
        if decoded_str[-1] != "":
            pred_str += decoded_str[-1]
        pred_str = join_jamos(pred_str)
        pred_str = re.sub('<unk>|<s>|<\/s>', '', pred_str)
        result_list.append(pred_str)
        return None

    test_dataset.map(map_to_result)
    return result_list


def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)

        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), 'rb') as f:
            dict_for_infer = pickle.load(f)

        model.load_state_dict(dict_for_infer['model'])

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        device = dict_for_infer['device']
        test_file_list = path_loader(test_path)
        test_dataset = prepare_dataset(test_file_list, None, processor,
                                       data_args)

        result_list = predict(test_dataset)
        prob = [1] * len(result_list)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(prob, result_list))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def path_loader(root_path):
    if data_args.mode == 'train':
        train_path = os.path.join(root_path, 'train')
        file_list = sorted(glob(os.path.join(train_path, 'train_data', '*')))
        label = pd.read_csv(os.path.join(train_path, 'train_label'))
        label['path'] = label['file_name'].apply(
            lambda row: os.path.join(DATASET_PATH, 'train', 'train_data', row))
        return file_list, label

    else:
        file_list = sorted(glob(os.path.join(root_path, 'test_data', '*')))

        return file_list


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda:0")

    # For first time run
    init_data()

    tokenizer = Wav2Vec2CTCTokenizer('./kowav-processor/vocab.json',
                                     unk_token="<unk>",
                                     pad_token="<pad>",
                                     word_delimiter_token="|",
                                     bos_token="<s>",
                                     eos_token="</s>")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=False)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

    if data_args.load_external_data:
        get_external_data(processor, args=data_args)
        shutil.rmtree('./train_temp')
        shutil.rmtree('./val_temp')
        print('Cleaning done!')
        exit(0)
    if data_args.mode == 'test':
        pass
        # download_kenlm()
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        gradient_checkpointing=model_args.gradient_checkpointing_,
        layerdrop=model_args.layerdrop,
        vocab_size=len(processor.tokenizer),
    )
    print(model)
    

    bind_model(model, training_args)
    if model_args.pause:
        nsml.paused(scope=locals())

    if data_args.mode == 'train':
        if model_args.data_type == 1:
            print("No pretrained model yet")
            # nsml.load(checkpoint='5', session='nia1030/final_stt_2/46')
        elif model_args.data_type == 2:
            print("No pretrained model yet")
            # nsml.load(checkpoint='4', session='nia1030/final_stt_1/188')
        # nsml.save(0)
        # exit()

        print("Dataset preparation begin!")
        train_dataset = Dataset.from_dict({})
        val_dataset = Dataset.from_dict({})

        if data_args.use_external_data:
            train_dataset_wrapper = DatasetWrapper(Dataset.from_dict({}))
            val_dataset_wrapper = DatasetWrapper(Dataset.from_dict({}))
            bind_dataset(train_dataset_wrapper, val_dataset_wrapper)
            print("Loading saved external data...")
            nsml.load(checkpoint='1000', session='nia1030/final_stt_1/122')
            train_dataset = train_dataset_wrapper.dataset
            val_dataset = val_dataset_wrapper.dataset

        else:
            file_list, label = path_loader(DATASET_PATH)
            if model_args.data_type == 2:
                label['no_header'] = label['file_name'].apply(lambda row: int(row[3:])<118681)
            print("Loading competition data...")
            train_dataset, val_dataset = prepare_dataset(file_list,
                                                         label,
                                                         processor,
                                                         args=data_args)
        print("Finished dataset preparation")

        # print(train_dataset[0])

        bind_model(model, training_args)

        wer_metric = datasets.load_metric("wer")
        cer_metric = datasets.load_metric("cer")

        data_collator = DataCollatorCTCWithPadding(processor=processor, pad_to_multiple_of=128,
                                                   padding=True)

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids ==
                           -100] = processor.tokenizer.pad_token_id

            pred_str = processor.batch_decode(pred_ids)
            # we do not want to group tokens when computing the metrics
            label_str = processor.batch_decode(pred.label_ids,
                                               group_tokens=False)
            print(f"pred : {[join_jamos(x) for x in pred_str[:5]]}")
            print(f"label : {[join_jamos(x) for x in label_str[:5]]}")
            wer = wer_metric.compute(predictions=pred_str,
                                     references=label_str)
            cer = cer_metric.compute(predictions=pred_str,
                                     references=label_str)
            return {"wer": wer, "cer": cer}

        if model_args.freeze_feature_extractor:
            model.freeze_feature_extractor()

        # optimizer = transformers.Adafactor(model.parameters(), scale_parameter=True, relative_step=True, lr = None, warmup_init = True)
        # lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)

        optimizer = torch.optim.AdamW(model.parameters(),
         lr = training_args.learning_rate, amsgrad=True)
        lr_scheduler = None

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.feature_extractor,
            callbacks=[NSMLCallback],
            optimizers=(optimizer,
                        lr_scheduler
            )
        )

        print("Training start")
        try:
            trainer.train()
        except Exception as error:
            logging.exception(error)
            print('error occured')
            
        print("Training done!")
        # clear disk
        train_dataset.cleanup_cache_files()
        val_dataset.cleanup_cache_files()

        shutil.rmtree('./train_temp')
        shutil.rmtree('./val_temp')
        print('Cleaning done!')
