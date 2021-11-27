#!/usr/bin/env python3
import warnings
from nsml import DATASET_PATH
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from data import init_data, prepare_dataset
from download import DatasetWrapper, bind_dataset, download_kenlm, get_external_data, bind_file
from ctcdecode import CTCBeamDecoder
import logging
from glob import glob
import pickle
import os
import time
import shutil
from pathlib import Path
import time
from dataclasses import dataclass, field
from datasets.arrow_dataset import Dataset

from hangul_utils import join_jamos
import re
from gpuinfo import get_gpu_info
from typing import Any, Callable, Dict, List, Optional, Set, Union
from torch import nn

import nsml
import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from packaging import version
from transformers.trainer_callback import TrainerControl, TrainerState

from transformers import (HfArgumentParser, Trainer,
                          Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor, is_apex_available,
                          trainer_utils, TrainerCallback, AutoTokenizer,
                          AutoModelForPreTraining, BartForConditionalGeneration, BartTokenizerFast)


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
        nsml.save(int(state.epoch))

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics, **kwargs):
        report_dict = {
            'step': state.epoch,
            'eval_loss': metrics['eval_loss'],
            'wer': metrics['eval_wer'],
            'cer': metrics['eval_cer'],
        }
        nsml.report(**report_dict)
        global gec_train_data
        global gec_val_data
        gec_train_dataset = DatasetWrapper(
            Dataset.from_dict(gec_train_data))
        gec_val_dataset = DatasetWrapper(Dataset.from_dict(gec_val_data))
        bind_dataset(gec_train_dataset, gec_val_dataset)
        nsml.save(1000+int(state.epoch))
        bind_model(model, training_args)
        try:
            print(gec_train_dataset.dataset[:5])
            print(gec_val_dataset.dataset[:5])
        except:
            pass
        gec_train_data = {'noise': [], 'orig': []}
        gec_val_data = {'noise': [], 'orig': []}

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if state.is_local_process_zero and 'loss' in logs:
            report_dict = {
                'step': state.epoch,
                'train_loss': logs['loss'],
                'learning_rate': logs['learning_rate'],
            }
            nsml.report(**report_dict)


class Wav2VecCTCTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        outputs = model(**inputs)

        # --------------- For future model training ------------------ #
        global gec_train_data
        global gec_val_data
        logits = outputs.logits
        pred_ids = np.argmax(logits.clone().detach().cpu(), axis=-1)
        decoded_strings = processor.batch_decode(pred_ids)
        decoded_labels = processor.batch_decode(labels)
        for noise, orig in zip(decoded_strings, decoded_labels):
            pred_str = join_jamos(noise)
            pred_str = re.sub('<unk>|<s>|<\/s>', '', pred_str)
            orig_str = join_jamos(orig)
            orig_str = re.sub('<unk>|<s>|<\/s>', '', orig_str)
            gec_train_data['noise'].append(
                pred_str + processor.tokenizer.eos_token)
            gec_train_data['orig'].append(
                orig_str + processor.tokenizer.eos_token)
        # beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
        # # select best predection
        # pred_ids = [beam_results[i][0][:out_lens[i][0]]
        #             for i in range(out_lens.shape[0])]
        # decoded_strings = processor.batch_decode(pred_ids)
        # decoded_labels = processor.batch_decode(labels)
        # for noise, orig in zip(decoded_strings, decoded_labels):
        #     pred_str = join_jamos(noise)
        #     pred_str = re.sub('<unk>|<s>|<\/s>', '', pred_str)
        #     orig_str = join_jamos(orig)
        #     orig_str = re.sub('<unk>|<s>|<\/s>', '', orig_str)
        #     gec_train_data['noise'].append(
        #         pred_str + processor.tokenizer.eos_token)
        #     gec_train_data['orig'].append(
        #         orig_str + processor.tokenizer.eos_token)
        # --------------- For future model training ------------------ #

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))


def predict(dataset, is_submit=True):
    model.to(device)
    model.eval()

    result_list = []
    decoder = CTCBeamDecoder(
        list(processor.tokenizer.get_vocab().keys()),
        # model_path='./model.arpa',
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=30,
        cutoff_prob=0.8,
        beam_width=40,
        num_processes=8,
        blank_id=0,
        log_probs_input=True  # No softmax layer in Wav2Vec2ForCTC
    )
    if not is_submit:
        print(f"Inference {len(dataset)} samples")

    def map_to_result(batch):
        if is_submit:
            input_values = processor(
                batch["data"],
                padding=True,
                sampling_rate=16_000,
                return_tensors="pt"
            ).input_values.to(device)
        else:
            input_features = [{"input_values": x}
                              for x in batch["input_values"]]
            input_values = processor.pad(
                input_features,
                padding=True,
                pad_to_multiple_of=128,
                return_tensors="pt",
            ).input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
        # select best predection
        pred_ids = [beam_results[i][0][:out_lens[i][0]]
                    for i in range(out_lens.shape[0])]
        decoded_strings = processor.batch_decode(pred_ids)

        for i in range(out_lens.shape[0]):
            pred_str = join_jamos(decoded_strings[i])
            pred_str = re.sub('<unk>|<s>|<\/s>', '', pred_str)
            result_list.append(pred_str)
        return
    tic = time.perf_counter()
    dataset.map(map_to_result, batched=True, batch_size=8)
    toc = time.perf_counter()
    print(f"Wav2Vec took {toc-tic:.1f}s")

    return result_list

    gec_result_list = []

    def apply_gec():
        for pred_str in result_list:
            inputs = gec_tokenizer([pred_str], return_tensors='pt')
            res_ids = gec_model.generate(
                inputs['input_ids'],
                max_length=30,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            res_str = [gec_tokenizer.decode(
                g, skip_special_tokens=True) for g in res_ids]

            gec_result_list.append(res_str[0])
    tic = time.perf_counter()
    apply_gec()
    toc = time.perf_counter()
    print(f"gec took {toc-tic:.1f}s")
    return gec_result_list


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

    def infer(test_path, **kwparser):
        device = dict_for_infer['device']
        test_file_list = path_loader(test_path)
        test_dataset = prepare_dataset(test_file_list, None, processor,
                                       data_args)

        result_list = predict(test_dataset)
        os.system(f'rm ./model.arpa')
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
    # os.system('df -h')
    # exit(0)
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
    # train = DatasetWrapper(Dataset.from_dict({}))
    # val = DatasetWrapper(Dataset.from_dict({}))
    # bind_dataset(train, val)
    # nsml.load(checkpoint='1002', session='nia1030/final_stt_1/291')
    # print(len(train.dataset))
    # print(len(val.dataset))

    # exit(0)

    decoder = CTCBeamDecoder(
        list(processor.tokenizer.get_vocab().keys()),
        # model_path='./model.arpa',
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=1,
        cutoff_prob=1.0,
        beam_width=1,
        num_processes=8,
        blank_id=0,
        log_probs_input=True  # No softmax layer in Wav2Vec2ForCTC
    )

    # if data_args.mode == 'test':
    #     path = Path('./model.arpa')
    #     bind_file(str(path))
    #     print("Loading kenlm...")
    #     nsml.load(checkpoint='6', session='nia1030/final_stt_1/260')
    #     print("Loading complete!")

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
    # print(model)

    gec_tokenizer = BartTokenizerFast.from_pretrained("hyunwoongko/kobart")
    gec_model = BartForConditionalGeneration.from_pretrained(
        "hyunwoongko/kobart")

    bind_model(model, training_args)
    if model_args.pause:
        nsml.paused(scope=locals())

    if data_args.mode == 'train':
        if model_args.data_type == 1:
            print("Loading pretrained model with external data")
            nsml.load(checkpoint='0', session='nia1030/final_stt_1/356')
        elif model_args.data_type == 2:
            print("Loading pretrained model with external data")
            nsml.load(checkpoint='0', session='nia1030/final_stt_1/356')
        # nsml.save(0)
        # exit()
        gec_train_data = {'noise': [], 'orig': []}
        gec_val_data = {'noise': [], 'orig': []}

        print("Dataset preparation begin!")
        train_dataset = Dataset.from_dict({})
        val_dataset = Dataset.from_dict({})

        if data_args.use_external_data:
            train_dataset, val_dataset = get_external_data(
                processor, data_args)

        else:
            file_list, label = path_loader(DATASET_PATH)
            if model_args.data_type == 2:
                label['no_header'] = label['file_name'].apply(
                    lambda row: int(row[3:]) < 118681)
            print("Loading competition data...")
            train_dataset, val_dataset = prepare_dataset(file_list,
                                                         label,
                                                         processor,
                                                         args=data_args)
        print("Finished dataset preparation")

        # import time
        # tic = time.time()
        # pred = predict(val_dataset, is_submit=False)
        # toc = time.time()
        # print(f"Inference took {toc-tic:.1f}s")
        # print(pred)
        # exit(0)

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

            noise = [re.sub('<unk>|<s>|<\/s>', '', x) for x in pred_str]
            noise = [join_jamos(x)
                     + processor.tokenizer.eos_token for x in noise]
            orig = [re.sub('<unk>|<s>|<\/s>', '', x) for x in label_str]
            orig = [join_jamos(x)
                    + processor.tokenizer.eos_token for x in orig]
            gec_val_data['noise'].extend(noise)
            gec_val_data['orig'].extend(orig)
            # --------------- For future model training ------------------ #
            # logits = pred_logits
            # logits = np.array_split(logits, max(
            #     len(logits) // data_args.cpu_batch_size, 1))
            # for logit in logits:
            #     beam_results, beam_score, timestep, out_lens = decoder.decode(
            #         torch.from_numpy(logit))
            #     # select best predection
            #     pred_ids = [beam_results[i][0][:out_lens[i][0]]
            #                 for i in range(out_lens.shape[0])]
            #     decoded_strings = processor.batch_decode(pred_ids)
            #     for noise_str, orig_str in zip(decoded_strings, label_str):
            #         pred = join_jamos(noise_str)
            #         pred = re.sub('<unk>|<s>|<\/s>', '', pred)
            #         orig = join_jamos(orig_str)
            #         orig = re.sub('<unk>|<s>|<\/s>', '', orig)
            #         gec_val_data['noise'].append(
            #             pred + processor.tokenizer.eos_token)
            #         gec_val_data['orig'].append(
            #             orig + processor.tokenizer.eos_token)
            # --------------- For future model training ------------------ #

            return {"wer": wer, "cer": cer}

        if model_args.freeze_feature_extractor:
            model.freeze_feature_extractor()

        # optimizer = transformers.Adafactor(model.parameters(), scale_parameter=True, relative_step=True, lr = None, warmup_init = True)
        # lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=training_args.learning_rate, amsgrad=True)
        lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            transformers.AdamW(model.parameters(),
                               lr=training_args.learning_rate),
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.num_train_epochs *
            (len(train_dataset) // training_args.per_device_train_batch_size //
             training_args.gradient_accumulation_steps // training_args.world_size),
            num_cycles=training_args.num_train_epochs
        )

        trainer = Wav2VecCTCTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.feature_extractor,
            callbacks=[NSMLCallback],
            optimizers=(optimizer, lr_scheduler),
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
