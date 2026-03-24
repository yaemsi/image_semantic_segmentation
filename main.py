import os
import secrets
import sys
import wandb

from datasets import load_from_disk
from loguru import logger
from transformers import (
    HfArgumentParser,
    Trainer
)

from image_semantic_segmentation import (
    UNetPlusPlusConfig,
    UNetPlusPlusHF,
    GeneralArguments,
    DataProcessingArguments,
    ModelArguments,
    CustomTrainingArguments,
    EvalArguments,
    simple_loss_func,
    padding_fn,
    compute_metrics,
    seg_data_collator,
    extract_data,
    build_dataset,
    evaluate,
    predict
)



def main():
    logger.info("*** Main program ***")
    parser = HfArgumentParser(
        (GeneralArguments, DataProcessingArguments, ModelArguments, CustomTrainingArguments, EvalArguments)
        )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file
        gen_args, data_args, model_args, train_args= parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        # If we pass only one argument to the script and it's the path to a yaml file
        gen_args, data_args, model_args, train_args= parser.parse_yaml_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        gen_args, data_args, model_args, train_args, eval_args = parser.parse_args_into_dataclasses()

    if gen_args.preprocess:
        if data_args.extract_data:
            logger.info(f">>> Extracting the data...")
            extract_data(
                data_args.zip_file_path,
                data_args.img_dir
                )
        if data_args.process_data:
            logger.info(f">>> Converting the images to dataset...")
            build_dataset(
                data_args.img_dir,
                data_args.ds_dir
                )

    elif gen_args.train:
        logger.info(f">>> Training the model...")
        logger.info(f"# Loading the dataset...")

        ds_path = model_args.dataset_path
        ds = load_from_disk(ds_path)

        # Apply to your dataset
        #ds["train"].set_transform(train_preprocess_fn)
        #ds["validation"].set_transform(eval_preprocess_fn)
        #ds["test"].set_transform(eval_preprocess_fn)
        ds["train"].set_transform(padding_fn)
        ds["validation"].set_transform(padding_fn)
        ds["test"].set_transform(padding_fn)

        logger.info(f"# Instantiating the U-net...")
        config = UNetPlusPlusConfig(**model_args.__dict__)
        model = UNetPlusPlusHF(config)
        if train_args.torch_compile:
            logger.info(f"# Compiling the model...")
            model.compile()

        logger.info(f"# Setting up wandb...")
        wandb.init(project = train_args.project, name = f"{train_args.run_name}_{secrets.token_hex(4)}")

        logger.info(f"# Initializing the trainer...")
        trainer = Trainer(
            model = model,
            args = train_args,
            train_dataset = ds["train"],
            eval_dataset = ds["validation"],
            compute_metrics = compute_metrics,
            data_collator = seg_data_collator,
            compute_loss_func=simple_loss_func
        )
        logger.info(f"# Launching the training...")
        trainer.train()

        logger.info(f"# Saving the model...")
        model.save_pretrained(os.path.join(train_args.output_dir, 'final'), from_pt=True)

    elif gen_args.evaluate:
        evaluate(
            model_path = eval_args.model_path,
            dataset_path = eval_args.ds_path,
            save_path = eval_args.save_path,
            split = eval_args.split,
            batch_size = eval_args.batch_size,
            num_proc = eval_args.num_proc,
        )
    elif gen_args.predict:
        predict(
            img_path=eval_args.img_path,
            save_path=eval_args.save_path,
            model_path = eval_args.model_path,
            batch_size = eval_args.batch_size,
            num_proc = eval_args.num_proc
        )

    logger.info("******* Done *******")




if __name__ == "__main__":
    main()
