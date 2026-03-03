#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=100

num_nodes=1
num_proc_per_node=8
node_rank=0
master_addr=localhost
master_port=12346


train_config=conf/train.yaml
data_config_path=.tmp/manifest_tta_audiocaps_test
stats_dir=${data_config_path}/stats
exp_dir=exp/ualm_4nodes

inference_config=conf/inference.yaml
inference_nj=1

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


inference_tag=$(basename "${inference_config%.*}")

inference_step=600000
inference_dir=${exp_dir}/inference/${inference_tag}_step_${inference_step}
mkdir -p ${inference_dir}

inference_ckpt=${exp_dir}/checkpoints/step_${inference_step}/global_step${inference_step}/mp_rank_00_model_states.pt

echo "Start model inference. Log at ${inference_dir}/logs/inference.*.log"
${cuda_cmd} JOB=1:${inference_nj} ${inference_dir}/logs/inference.JOB.log \
CUDA_LAUNCH_BLOCKING=1 ../../../scripts/inference.py \
    --rank JOB --world-size ${inference_nj} \
    --train-config ${train_config} \
    --inference-config ${inference_config} \
    --model-checkpoint ${inference_ckpt} \
    --output-dir ${inference_dir} \
    --data-config-path ${data_config_path} \
    --num-worker 1
