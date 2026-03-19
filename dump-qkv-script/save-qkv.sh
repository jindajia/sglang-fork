# step 1: start sglang engine
export DUMP_KVCACHE=true
export DUMP_KVCACHE_TOKENS=10000
export DUMP_KVCACHE_DIR="/data/jinda/kv-cache/Qwen3-4B-thinking-2507/qkv_test_10000-tokens"
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
    --prefill-attention-backend triton \
    --decode-attention-backend triton \
    --sampling-backend flashinfer \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --host 0.0.0.0 \
    --port 30001 \
    --disable-cuda-graph \
    --skip-server-warmup

# step 2: send sample requests to sglang engine
python -m tore_eval.eval \
    ./run_simple_evals.yaml

# step 3: use dump-qkv.ipynb to concat chunks