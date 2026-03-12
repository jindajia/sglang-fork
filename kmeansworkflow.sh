# Stage 1 dump kv cache

# for example we want to dump kv cache for Qwen3-4B-Thinking-2507
# Stage1.1 we need first start the server 
export DUMP_KVCACHE=true
export DUMP_KVCACHE_TOKENS=20000 # number of tokens to dump
export DUMP_KVCACHE_DIR="path-we-want-to-save-kv-cache"
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
# Stage1.2 then we need to send some sample requests to server to dump kv cache
# for exmaple here I use tore-eval to send sample requests
lm_eval --model local-completions --tasks mmlu_pro \
    --model_args model=Qwen/Qwen3-4B-Thinking-2507,base_url=http://localhost:30001/v1/completions,max_model_len=20000,num_concurrent=32,max_retries=1,tokenized_requests=False

# Stage2 after finish with dump kv cache, we need to use dumped kv cache to calculate kmeans centroids

# please check flash-kmeans.ipynb


# Stage3 run with kmeans serving
# need to set KV Centroids path, and number of clusters
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_KV_CENTROIDS_PATH=/data/jinda/kv-cache/Qwen3-4B-thinking-2507/c_64/
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
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --host 0.0.0.0 --port 30001