#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


num_nodes=4
num_proc_per_node=8
node_rank=0
master_addr=localhost
master_port=12346

train_config=conf/train.yaml
data_config_path=.tmp/manifest_NAME  # remember to make a symlink to ualm/.tmp
stats_dir=${data_config_path}/stats
exp_dir=exp/ualm_4nodes

inference_config=conf/inference.yaml
inference_step=50000
inference_nj=1

. utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

CUDA_LAUNCH_BLOCKING=1 deepspeed \
  --num_nodes ${num_nodes} \
  --num_gpus ${num_proc_per_node} \
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
    ../../../scripts/train.py \
    --data-config-path ${data_config_path} \
    --train-config ${train_config} \
    --stats-dir ${stats_dir} \
    --output-dir ${exp_dir}
