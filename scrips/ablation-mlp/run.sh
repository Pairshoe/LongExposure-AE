#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/ablation-mlp
save_json="$script_dir/record.json"


# 检查并创建或清空 JSON 文件
if [ ! -f "$save_json" ]; then
    echo "[]" > "$save_json"
else
    > "$save_json"
    echo "[]" > "$save_json"
fi



# 设置 PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$src_dir"
else
    if [[ ":$PYTHONPATH:" != *":$src_dir:"* ]]; then
        export PYTHONPATH="$PYTHONPATH:$src_dir"
    fi
fi

python "$script_dir/profile_sparse_pattern.py" \
    --data ./dataset/valid_opt.jsonl \

python "$script_dir/benchmark_sparse_mlp.py" \
    --save_json "$save_json" \
    > "$script_dir/output/sparse_mlp.log"

python "$script_dir/plot_benchmark_sparsity.py" \
    --output "$script_dir/exp-ablation-mlp-sparsity.png"

python "$script_dir/plot_benchmark_time.py" \
    --save_json "$save_json" \
    --output "$script_dir/exp-ablation-mlp-time.png"