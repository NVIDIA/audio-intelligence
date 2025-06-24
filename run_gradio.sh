EXP_ROOT=/path/to/your/exp/root && \
MODEL_NAME=etta && \
CKPT_NAME=model_unwrap_step_1000000.ckpt && \
python run_gradio.py \
    --ckpt-path $EXP_ROOT/$MODEL_NAME/$CKPT_NAME \
    --share