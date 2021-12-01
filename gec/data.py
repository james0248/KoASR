import gc
import pandas as pd
import numpy as np
import re
import json
import shutil
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk, load_dataset

from nsml import DATASET_PATH
import os
import time
import nsml

from pathlib import Path
import datasets
from datasets import Dataset
from jiwer import cer


class DatasetWrapper():
    def __init__(self, dataset):
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset


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


def filter_high_cer(batch):
    error = cer(batch['orig'], batch['noise'])
    return error < 0.15 and 0 < error


def filter_by_length(batch, tokenizer):
    inputs = batch["noise"]
    targets = batch["orig"]
    model_inputs = tokenizer(inputs, padding=False)
    with tokenizer.as_target_tokenizer():
        lables = tokenizer(targets, padding=False)
    if len(model_inputs["input_ids"]) < 5 or len(model_inputs["input_ids"]) > 50:
        return False
    if len(lables["input_ids"]) < 5 or len(lables["input_ids"]) > 50:
        return False
    if abs(len(model_inputs["input_ids"]) - len(lables["input_ids"])) > 10:
        return False
    return True


def preprocess_function(batch, tokenizer, data_args):
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = batch["noise"]
    targets = batch["orig"]
    model_inputs = tokenizer(
        inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def fetch_dataset(data_dict):
    train_datasets = []
    val_datasets = []
    for session in data_dict:
        for i in range(1, data_dict[session]):
            train_dataset_wrapper = DatasetWrapper(Dataset.from_dict({}))
            val_dataset_wrapper = DatasetWrapper(Dataset.from_dict({}))
            bind_dataset(train_dataset_wrapper, val_dataset_wrapper)
            nsml.load(checkpoint=str(1000+i), session=session)
            train_datasets.append(train_dataset_wrapper.dataset)
            val_datasets.append(val_dataset_wrapper.dataset)
    train_dataset = datasets.concatenate_datasets(train_datasets)
    val_dataset = datasets.concatenate_datasets(val_datasets)
    return train_dataset, val_dataset


def prepare_dataset(data_dict, tokenizer, args):
    train_dataset, val_dataset = fetch_dataset(data_dict)

    print(
        f"Number of data before filter: {len(train_dataset)+len(val_dataset)}")

    train_dataset = train_dataset.filter(
        filter_high_cer, num_proc=args.preprocessing_num_workers)
    val_dataset = val_dataset.filter(
        filter_high_cer, num_proc=args.preprocessing_num_workers)

    print(
        f"Number of data after cer filter: {len(train_dataset)+len(val_dataset)}")

    train_dataset = train_dataset.filter(
        filter_by_length,
        num_proc=args.preprocessing_num_workers,
        fn_kwargs={'tokenizer': tokenizer},
    )
    val_dataset = val_dataset.filter(
        filter_by_length,
        num_proc=args.preprocessing_num_workers,
        fn_kwargs={'tokenizer': tokenizer},
    )

    print(
        f"Number of data after text filter: {len(train_dataset)+len(val_dataset)}")
    print("Start preprocess...")

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        fn_kwargs={'tokenizer': tokenizer, 'data_args': args},
    )

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=val_dataset.column_names,
        fn_kwargs={'tokenizer': tokenizer, 'data_args': args},
    )

    print("Preprocess done!")

    bind_dataset(DatasetWrapper(train_dataset), DatasetWrapper(val_dataset))
    nsml.save(1000)

    return train_dataset, val_dataset
