MAIN_ROOT=$PWD/../../..

export PATH=$PWD/utils/:$PATH
export LC_ALL=C
export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
# export PYTHONPATH=${MAIN_ROOT}:$PYTHONPATH
case ":${PYTHONPATH:-}:" in
  *":$MAIN_ROOT:"*) ;;  # already in PYTHONPATH, do nothing
  *) export PYTHONPATH="$MAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# NOTE(kamo): Source at the last to overwrite the setting
. local/path.sh

# NOTE(Jinchuan): avoid pytorch memory segmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ESPNET_DATASET_REGISTRY
