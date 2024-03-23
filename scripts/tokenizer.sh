#!/bin/bash

python tools/tokenizer.py \
    --text_input_path tmp/raw_data.txt \
    --bin_output_path tmp/output.bin
