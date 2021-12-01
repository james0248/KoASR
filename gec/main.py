#!/usr/bin/env python3
import warnings
from nsml import DATASET_PATH
from arguments import ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments
import logging
from glob import glob
import pickle
import os
import shutil
from pathlib import Path
import time
import re
from dataclasses import dataclass, field

from gpuinfo import get_gpu_info
from typing import Any, Callable, Dict, List, Optional, Set, Union

import nsml
import numpy as np
import torch
import datasets
from datasets import Dataset
from packaging import version
import transformers
from transformers.trainer_callback import TrainerControl, TrainerState

from transformers import (HfArgumentParser, is_apex_available, TrainerCallback,
                          BartForConditionalGeneration, BartTokenizerFast, set_seed,
                          Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer)

from data import prepare_dataset, DatasetWrapper, bind_dataset

warnings.filterwarnings(action='ignore')

if is_apex_available():
    from apex import amp  # type: ignore

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


data_dict = {
    'nia1030/final_stt_1/368': 10,
    'nia1030/final_stt_3/69': 5,
    'nia1030/final_stt_2/194': 5,
    'nia1030/final_stt_2/199': 3,
}


class NSMLCallback(TrainerCallback):
    def on_epoch_end(self, args: Seq2SeqTrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        global dict_for_infer
        dict_for_infer = {
            'model': model.state_dict(),
            'epochs': state.epoch,
            'learning_rate': args.learning_rate,
            'tokenizer': tokenizer,
            'device': device,
        }
        nsml.save(100 + int(state.epoch))

    def on_evaluate(self, args: Seq2SeqTrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics, **kwargs):
        report_dict = {
            'step': state.epoch,
            'eval_loss': metrics['eval_loss'],
            'wer': metrics['eval_wer'],
            'cer': metrics['eval_cer'],
        }
        nsml.report(**report_dict)

    def on_log(self, args: Seq2SeqTrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if state.is_local_process_zero and 'loss' in logs:
            report_dict = {
                'step': state.epoch,
                'train_loss': logs['loss'],
                'learning_rate': logs['learning_rate'],
            }
            nsml.report(**report_dict)


def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))


def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)

        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print("모델 저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), 'rb') as f:
            dict_for_infer = pickle.load(f)

        model.load_state_dict(dict_for_infer['model'])

        print("모델 로딩 완료!")

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load)


if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda:0")
    set_seed(training_args.seed)

    # Set seed before initializing model.

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = BartTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        dropout=model_args.dropout,
        attention_dropout=model_args.attention_dropout,
        activation_dropout=model_args.activation_dropout,
        classifier_dropout=model_args.classifier_dropout,
        encoder_layerdrop=model_args.encoder_layerdrop,
        decoder_layerdrop=model_args.decoder_layerdrop,
        vocab_size=len(tokenizer),
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    bind_model(model, training_args)
    if model_args.pause:
        nsml.paused(scope=locals())

    bind_model(model, training_args)
    if data_args.mode == 'train':
        print("Dataset preparation begin!")
        if data_args.use_processed_data:
            train_dataset_wrapper = DatasetWrapper(Dataset.from_dict({}))
            val_dataset_wrapper = DatasetWrapper(Dataset.from_dict({}))
            bind_dataset(train_dataset_wrapper, val_dataset_wrapper)
            nsml.load(checkpoint='1000', session='nia1030/final_stt_2/260')
            train_dataset = train_dataset_wrapper.dataset
            val_dataset = val_dataset_wrapper.dataset.select(range(5000))
            bind_dataset(DatasetWrapper(train_dataset),
                         DatasetWrapper(val_dataset))
            nsml.save(1000)
            print(train_dataset[:5])
        else:
            train_dataset, val_dataset = prepare_dataset(
                data_dict, tokenizer, data_args)
        print("Finished dataset preparation")

        bind_model(model, training_args)

        # Metric
        wer_metric = datasets.load_metric("wer")
        cer_metric = datasets.load_metric("cer")

        # Data collator
        label_pad_token_id = - \
            100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]
            preds = [re.sub('[^가-힣\s.,!?~]', '', pred) for pred in preds]
            labels = [re.sub('[^가-힣\s.,!?~]', '', label) for label in labels]

            return preds, labels

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            print(preds)
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels,
                                  tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels)

            print(f"pred : {decoded_preds[:5]}")
            print(f"label : {decoded_labels[:5]}")
            wer = wer_metric.compute(predictions=decoded_preds,
                                     references=decoded_labels)
            cer = cer_metric.compute(predictions=decoded_preds,
                                     references=decoded_labels)

            return {"wer": wer, "cer": cer}

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=training_args.learning_rate, amsgrad=True)
        lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            transformers.AdamW(model.parameters(),
                               lr=training_args.learning_rate),
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.num_train_epochs *
            (len(train_dataset) // training_args.per_device_train_batch_size //
                training_args.gradient_accumulation_steps // training_args.world_size),
            num_cycles=training_args.num_train_epochs,
        )

        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[NSMLCallback],
            optimizers=(optimizer, lr_scheduler),
        )

        # Training
        print("Training start")
        try:
            trainer.train()
        except Exception as error:
            logging.exception(error)
            print('error occured')

        print("Training done!")
