# Stage0 prepare environment and install dependencies
bash setup_env.sh
source /data/$USER/miniconda/bin/activate sglang_env
conda activate sglang_env

export HF_HOME=/data/shared/huggingface

# Stage 1 dump kv cache

# for example we want to dump kv cache for Qwen3-4B-Thinking-2507
# Stage1.1 we need first start the server 
export DUMP_KVCACHE=true
export DUMP_KVCACHE_TOKENS=20000 # number of tokens to dump
export DUMP_KVCACHE_DIR="/data/jisenli2/kv-cache/Qwen3-4B-Thinking-2507/mmlu_pro-20000-tokens/"
mkdir -p $DUMP_KVCACHE_DIR
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Thinking-2507 \
    --max-running-requests 32 \
    --max-queued-requests 32 \
    --page-size 128 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.8 \
    --pp-max-micro-batch-size 32 \
    --kv-cache-dtype auto \
    --prefill-attention-backend fa3 \
    --decode-attention-backend triton \
    --sampling-backend flashinfer \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --host 0.0.0.0 \
    --port 30001 \
    --disable-cuda-graph

export DUMP_KVCACHE_DIR="/data/jisenli2/kv-cache/GLM-4.7-FP8/mmlu_pro-20000-tokens/"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-FP8 \
    --max-running-requests 32 \
    --max-queued-requests 32 \
    --page-size 128 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.8 \
    --pp-max-micro-batch-size 32 \
    --kv-cache-dtype auto \
    --prefill-attention-backend fa3 \
    --decode-attention-backend triton \
    --sampling-backend flashinfer \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --host 0.0.0.0 \
    --port 30001 \
    --disable-cuda-graph
# Stage1.2 then we need to send some sample requests to server to dump kv cache
# for exmaple here I use tore-eval to send sample requests
lm_eval --model local-completions --tasks mmlu_pro \
    --limit 16 \
    --model_args model=Qwen/Qwen3-4B-Thinking-2507,base_url=http://localhost:30001/v1/completions,max_model_len=20000,num_concurrent=32,max_retries=1,tokenized_requests=False

lm_eval --model local-chat-completions --tasks mmlu_pro \
      --limit 5 \
      --apply_chat_template \
      --model_args "model=zai-org/GLM-4.7-FP8,base_url=http://localhost:30001/v1/chat/completions,max_model_len=202752,num_concurrent=32,max_retries=1,tokenized_requests=False,temperature=1.0"

# Stage2 after finish with dump kv cache, we need to use dumped kv cache to calculate kmeans centroids

# please check flash-kmeans.ipynb


# Stage3 run with kmeans serving
# need to set KV Centroids path, and number of clusters
# unset the DUMP_KVCACHE flag to avoid accidentally dumping kv cache again
unset DUMP_KVCACHE
export HF_HOME=/data/shared/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_KV_CENTROIDS_PATH=/data/jisenli2/kv-cache/Qwen3-4B-Thinking-2507/mmlu_pro-20000-tokens/c_64/
export N_CLUSTERS=64
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Thinking-2507 \
    --max-running-requests 32 \
    --max-queued-requests 32 \
    --page-size 128 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.8 \
    --pp-max-micro-batch-size 32 \
    --kv-cache-dtype int4 \
    --prefill-attention-backend fa3 \
    --decode-attention-backend triton \
    --sampling-backend flashinfer \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --host 0.0.0.0 --port 30001 \
    2>&1 | tee /data/jisenli2/kv_rotation/sglang_stage3_TP1.log

curl http://localhost:30001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-4B-Thinking-2507", "messages": [{"role": "user", "content": "who are you"}], "max_tokens": 100}' \
    2>&1 | tee -a /data/jisenli2/kv_rotation/sglang_stage3_TP4_output.log