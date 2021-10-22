#!/usr/bin/env python3
from gc import callbacks
import gc
import logging
import re
from glob import glob
import pickle
import sys
import os
from dataclasses import dataclass, field
from datasets.search import NearestExamplesResults

from hangul_utils import join_jamos
from gpuinfo import get_gpu_info
from typing import Any, Callable, Dict, List, Optional, Set, Union

import nsml
import datasets
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import nn
from transformers.trainer_callback import TrainerControl, TrainerState

from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor, is_apex_available,
                          trainer_utils, TrainerCallback, AutoTokenizer,
                          AutoModelForPreTraining)

from data import init_data, remove_duplicate_tokens, prepare_dataset
from nsml import DATASET_PATH

import warnings

warnings.filterwarnings(action='ignore')

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True,
        metadata={
            "help":
            "Whether to freeze the feature extractor layers of the model."
        })
    gradient_checkpointing_2: Optional[bool] = field(
        default=False,
        metadata={
            "help":
            "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    pause: Optional[int] = field(
        default=0,
        metadata={"help": "Whether to submit or not"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    mode: Optional[str] = field(
        default="train",
        metadata={
            "help": "Set mode for training or testing. Defaults to 'train'"
        },
    )
    split: Optional[int] = field(
        default=None,
        metadata={
            "help": "Which split to use",
        },
    )
    max_split: Optional[int] = field(
        default=None,
        metadata={
            "help": "Which split to use",
        },
    )
    target_text_column: Optional[str] = field(
        default="target_text",
        metadata={
            "help":
            "Column in the dataset that contains label (target text). Defaults to 'text'"
        },
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={
            "help":
            "Column in the dataset that contains speech file path. Defaults to 'file'"
        },
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={
            "help":
            "Resample loaded audio to target feature extractor's sampling rate or not."
        },
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={
            "help":
            "Filters out examples longer than specified. Defaults to no filtering."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        })
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )


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
        nsml.save(int(state.epoch))
        #print(state.epoch)
        print(state.best_metric)


class CTCTrainer(Trainer):
    def __init__(self, **kwargs):
        super(CTCTrainer, self).__init__(**kwargs)
        self.cur_step = 0
        self.report_interval = 100

    def training_step(
            self, model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        self.cur_step += 1
        # print(self.cur_step)
        gc.collect()
        torch.cuda.empty_cache()
        # print(get_gpu_info())
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(
                    f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']"
                )

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        if self.cur_step % self.report_interval == 0:
            print(self.cur_step)
            nsml.report(step=self.cur_step, batch_loss=loss.detach().item())
        return loss.detach()


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
        test_file_list = path_loader(test_path)
        test_dataset = prepare_dataset(test_file_list, None, processor,
                                       data_args)
        result_list = []

        def map_to_result(batch):
            model.to(device)
            input_values = processor(
                batch["data"],
                sampling_rate=batch["sampling_rate"],
                return_tensors="pt").input_values.to("cuda")

            with torch.no_grad():
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            pred_ids = remove_duplicate_tokens(pred_ids.cpu().numpy()[0],
                                               processor)
            result_list.append(
                list(join_jamos(processor.batch_decode(pred_ids)[0])))

            return None

        results = test_dataset.map(map_to_result)

        prob = [1] * len(result_list)
        print(result_list[0])

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
                                     unk_token="[UNK]",
                                     pad_token="[PAD]",
                                     word_delimiter_token="|")
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=False)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
<<<<<<< HEAD
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=model_args.gradient_checkpointing_2,
        vocab_size=len(processor.tokenizer),
=======
        # cache_dir=model_args.cache_dir,
        # gradient_checkpointing=model_args.gradient_checkpointing,
        # vocab_size=len(processor.tokenizer),
>>>>>>> c0ddbd2 (Fix submit errors)
    )

    bind_model(model, training_args)
    nsml.save(0)
    if model_args.pause:
        nsml.paused(scope=locals())

    file_list, label = path_loader(DATASET_PATH)

    if data_args.mode == 'train':
        print("Dataset preparation begin!")
        train_dataset, val_dataset = prepare_dataset(file_list,
                                                     label,
                                                     processor,
                                                     args=data_args)
        print("Finished dataset preparation")

        wer_metric = datasets.load_metric("wer")

        data_collator = DataCollatorCTCWithPadding(processor=processor,
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
            wer = wer_metric.compute(predictions=pred_str,
                                     references=label_str)

            return {"wer": wer}

        if model_args.freeze_feature_extractor:
            model.freeze_feature_extractor()

        trainer = CTCTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.feature_extractor,
            callbacks=[NSMLCallback],
        )

        print("Training start")
        trainer.train()
        processor.save_pretrained('./kowav-processor')