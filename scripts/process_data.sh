#!/bin/bash

# Parameters
export ZIP_FILE_PATH="./data/data.zip"
export DS_PATH="./data/raw/test"
export OUTPUT_DIR="./data"


python  main.py --preprocess \
    --zip_file_path $ZIP_FILE_PATH \
    --img_dir $DS_PATH \
    --ds_dir $OUTPUT_DIR \
    --extract_data true \
    --process_data true

