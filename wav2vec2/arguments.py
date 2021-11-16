from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    data_type: int = field(
        default=None,
        metadata={
            "help":
            "Select dataset"
        },
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={
            "help": "The dropout ratio for the attention probabilities."
        },
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."
        },
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True,
        metadata={
            "help":
            "Whether to freeze the feature extractor layers of the model."
        })
    gradient_checkpointing_: Optional[bool] = field(
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
    layerdrop: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The LayerDrop probability."
        }
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
    load_external_data: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to load external data from Google drive",
        },
    )
    use_external_data: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use external data for training",
        },
    )
    target_text_column: Optional[str] = field(
        default="target_text",
        metadata={
            "help":
            "Column in the dataset that contains label (target text). Defaults to 'text'"
        },
    )
    gdrive_code: Optional[str] = field(
        default="",
        metadata={
            "help":
            "code for gdrive access"
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
    writer_batch_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Disk and memory"},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    pass
