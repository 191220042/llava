#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-sefp.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-sefp.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_output_sefp.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_result_sefp.json
