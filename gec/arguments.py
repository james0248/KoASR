from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    data_type: int = field(
        default=None,
        metadata={
            "help": "Select dataset"
        },
    )
    dropout: Optional[float] = field(
        default=0.07, metadata={
            "help": "The dropout ratio for the attention probabilities."
        },
    )
    attention_dropout: Optional[float] = field(
        default=0.094, metadata={
            "help": "The dropout ratio for the attention probabilities."
        },
    )
    activation_dropout: Optional[float] = field(
        default=0.055, metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    classifier_dropout: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    encoder_layerdrop: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Dropout for encoder layer"
        },
    )
    decoder_layerdrop: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Dropout for decoder layer"
        },
    )
    pause: Optional[int] = field(
        default=0,
        metadata={"help": "Whether to submit or not"},
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    mode: Optional[str] = field(
        default="train",
        metadata={
            "help": "Set mode for training or testing. Defaults to 'train'"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    writer_batch_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Disk and memory"},
    )
    use_processed_data: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use preprocessed data"
        },
    )


@dataclass
class Seq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    pass
