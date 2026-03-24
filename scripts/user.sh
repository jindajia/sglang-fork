# === All layers int4+H, only layer 0 gets QR ===
HF_HOME="/data/shared/huggingface" \
PYTHON_BIN="/home/charlie/miniconda3/envs/sglangfork/bin/python" \
GPU=1 PORT=37131 MODE=hadamard_qr \
RUN_ID=int4_h_qr_layer0_only \
MEM_FRACTION_STATIC=0.4 MAX_RUNNING_REQUESTS=8 MAX_QUEUED_REQUESTS=16 \
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507" \
Q_ROTATION_PATH="/data/shared/charlie/sglangfork/per_layer_rotations/q_rotation_layer0.pt" \
bash scripts/launch_hadamard_qr_server_tp1.sh

HF_HOME="/data/shared/huggingface" \
PYTHON_BIN="/home/charlie/miniconda3/envs/sglangfork/bin/python" \
MODE_TAG=hadamard_qr PORT=37131 \
RUN_ID=int4_h_qr_layer0_only \
PRESET_NAME=gpqa_think NUM_WORKERS=2 REPEAT=3 \
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507" \
bash scripts/run_hadamard_qr_eval_tp1.sh
