import os
from dataclasses import dataclass, field
from pathlib import Path
from transformers import TrainingArguments
from transformers.trainer_utils import (
    IntervalStrategy,
    SaveStrategy,
)

@dataclass
class GeneralArguments:
    """
    Whether to preprocess data, train or evaluate
    """
    preprocess: bool = field(
        default=False,
        metadata={
            "help": "Use this option to process the data."
        },
    )
    train: bool = field(
        default=False,
        metadata={
            "help": "Use this option to train the model."
        },
    )
    evaluate: bool = field(
        default=False,
        metadata={
            "help": "Use this option to evaluate the model."
        },
    )
    predict: bool = field(
        default=False,
        metadata={
            "help": "Use this option to use the model for inference."
        },
    )
    def __post_init__(self):
        var_list = [self.preprocess, self.train, self.evaluate, self.predict]
        if not any(v==True for v in var_list):
            raise ValueError(f"You should use at least one of these options: '{['--preprocess', '--train', '--evaluate', '--predict']}'")
        elif var_list.count(True) != 1:
            raise ValueError(f"You should use only one of these options: '{['--preprocess', '--train', '--evaluate', '--evaluate']}'")




@dataclass
class DataProcessingArguments:
    """
    Arguments related to dataset preprocessing.
    """
    zip_file_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the zip file."
        },
    )
    img_dir: str | None = field(
        default=None,
        metadata={"help": "Where to extract the images."}
    )
    ds_dir: str | None = field(
        default=None,
        metadata={
            "help": "Directory where to save the dataset."
        },
    )
    extract_data: bool = field(
        default=False,
        metadata={
            "help": "Instructs the program to extract data."
        },
    )
    process_data: bool = field(
        default=False,
        metadata={
            "help": "Instructs the program to proces the extracted data."
        },
    )


    def __post_init__(self):
        if self.extract_data:
            if not self.zip_file_path:
                raise ValueError(
                    "You must specify the zip file containing images."
                )
            if not self.img_dir:
                raise ValueError(
                    "You must specify a valid directory where to save the images."
                )
        if self.process_data:
            if not self.img_dir:
                raise ValueError(
                    "You must specify a valid directory containing the images."
                )
            if not self.ds_dir:
                raise ValueError(
                    "You must specify a directory where datasets are to be saved."
                )


@dataclass
class ModelArguments:
    """
    Model parameters.
    """
    encoder_name: str = field(
        default="resnet34",
        metadata={
            "help": "Encoder architecture."
        },
    )
    encoder_depth: int= field(
        default=5,
        metadata={
            "help": "Depth of the encoder."
        },
    )
    encoder_weights: str = field(
        default="imagenet",
        metadata={
            "help": "Pretrained weights."
        },
    )
    decoder_use_norm: str = field(
        default="batchnorm",
    )
    decoder_channels: tuple = field(
        default=(256, 128, 64, 32, 16),
    )
    decoder_attention_type: str | None = field(
        default=None,
    )
    decoder_interpolation: str = field(
        default='nearest',
    )
    in_channels: int = field(
        default=3,
    )
    classes: int = field(
        default=2,
    )
    activation: str | None = field(
        default=None,
    )
    aux_params: tuple | None = field(
        default=None,
    )
    dataset_path: str | None = field(
        default=None,
    )

    def __post_init__(self):
        pass


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Training arguments.
    """
    output_dir: str | None = field(
        default="./output_image_segmentation/",
        metadata={
            "help": "Output directory."
        },
    )
    learning_rate: float = field(
        default=6e-5,
        metadata={
            "help": "Learning rate."
        },
    )
    max_steps: int = field(
        default=50000,
        metadata={
            "help": "Overrides `num_train_epochs`. If set to a positive number, the total number of training steps to perform."
        },
    )
    warmup_steps: float = field(
        default=15000,
        metadata={
            "help": "Number of steps for a linear warmup from 0 to `learning_rate`. Can be an integer (exact steps) or a float in [0, 1) (ratio of total steps)."
        },
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={
            "help": "Training batch size."
        },
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={
            "help": "Eval batch size."
        },
    )
    eval_strategy: IntervalStrategy | str = field(
        default="steps",
        metadata={"help": "When to run evaluation. Options: 'no', 'steps', 'epoch'."},
    )
    eval_steps: float | None = field(
        default=2500,
        metadata={
            "help": (
                "Number of update steps between evaluations if `eval_strategy='steps'`. Defaults to `logging_steps` if not set."
                " Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    save_strategy: SaveStrategy | str = field(
        default="steps",
        metadata={
            "help": "The checkpoint save strategy to adopt during training. Options: 'no', 'epoch', 'steps', 'best'."
        },
    )
    save_steps: float = field(
        default=2500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Whether or not to automatically remove the columns unused by the model forward method."},
    )

    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load the best checkpoint at the end of training. Requires `eval_strategy` to be set."},
    )
    metric_for_best_model: str | None = field(
        default="mean_iou",
        metadata={
            "help": "Metric to use for comparing models when `load_best_model_at_end=True`. Defaults to 'loss'."
        },
    )

    greater_is_better: bool | None = field(
        default=True,
        metadata={"help": "Whether higher metric values are better. Defaults based on `metric_for_best_model`."},
    )

    save_total_limit: int | None = field(
        default=3,
        metadata={
            "help": "Maximum number of checkpoints to keep. Deletes older checkpoints in `output_dir`. The best checkpoint is always retained when `load_best_model_at_end=True`."
        },
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": "Compile the model using `torch.compile()` for faster training."}
    )
    torch_compile_mode: str | None = field(
        default=None,
        metadata={
            "help": "Compilation mode for `torch.compile()`. If set, automatically enables `torch_compile`.",
        },
    )
    torch_compile_backend: str | None = field(
        default=None,
        metadata={
            "help": "Backend for `torch.compile()`. If set, automatically enables `torch_compile`.",
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "Enable bfloat16 (BF16) mixed precision training. Generally preferred over FP16 due to better numerical stability."
        },
    )

    use_cache: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use cache for the model For training, this is usually not needed apart from some PEFT methods that uses `past_key_values`."
        },
    )

    do_train: bool = field(
        default=True,
        metadata={
            "help": "Whether to run training. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    do_eval: bool = field(
        default=True,
        metadata={
            "help": "Whether to run evaluation. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    do_predict: bool = field(
        default=True,
        metadata={
            "help": "Whether to run predictions on the test set. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    disable_tqdm: bool | None = field(
        default=False,
        metadata={"help": "Disable tqdm progress bars. Defaults to True if log_level is warning or lower."},
    )

    # --- Experiment Tracking ---
    report_to: None | str | list[str] = field(
        default="wandb",
        metadata={
            "help": "The list of integrations to report the results and logs to. Use 'all' for all installed integrations, 'none' for no integrations."
        },
    )
    run_name: str | None = field(
        default='unet_plusplus',
        metadata={
            "help": (
                "An optional descriptor for the run. Notably used for trackio, wandb, mlflow comet and swanlab "
                "logging."
            )
        },
    )
    project: str = field(
        default="image_semantic_segmentation",
        metadata={"help": "The name of the project to use for logging. Currently, only used by Trackio."},
    )
    # --- Dataloader ---
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    dataloader_persistent_workers: bool = field(
        default=True,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. "
            "Can potentially speed up training, but will increase RAM usage."
        },
    )
    dataloader_prefetch_factor: int | None = field(
        default=4,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
            )
        },
    )


    def __post_init__(self):
        super().__post_init__()



@dataclass
class EvalArguments:
    """
    Arguments for evaluation.
    """
    model_path: str  = field(
        default=Path("./output_image_segmentation/final"),
        metadata={
            "help": "Path to model weights."
        },
    )

    img_path: str  = field(
        default=Path("./data/raw_test_set"),
        metadata={
            "help": "Path to raw test data."
        },
    )

    ds_path: str  = field(
        default=Path("./data/processed"),
        metadata={
            "help": "Path to raw test data."
        },
    )

    split: str = field(
        default="test",
        metadata={
            "help": "Dataset fraction to use."
        },
    )

    save_path: str  = field(
        default=Path("./output_predictions"),
        metadata={
            "help": "Where to save results."
        },
    )

    batch_size: int = field(
        default=32,
        metadata={
            "help": "Evaluation batch size."
        },
    )

    num_proc: int = field(
        default=2,
        metadata={
            "help": "Dataloader's number of processes."
        },
    )




    def __post_init__(self):
        pass
