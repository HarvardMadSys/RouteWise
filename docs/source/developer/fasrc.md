# **FASRC Deployment**

1. login to fasrc login ndoe
2. alloc gpu resources `alloc -c 32 -m 1024 -g 2 -u h200 -t 168 -p seas_gpu`

    then you will get a gpu node name, you need to ssh to the gpu node

3. ssh to the gpu node, serving the model by vllm

    ```
    # docker deploy on holygpu node
    podman run --rm \
        --device nvidia.com/gpu=all \
        -p 8000:8000 \
        --ipc=host \
        -v /n/netscratch/juncheng_lab/muxint/llm_models:/models:Z \
        docker.io/vllm/vllm-openai:latest \
        --model /models/meta-llama_Llama-4-Scout-17B-16E \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.95

    # qwen3 coder
    podman run --rm \
        --device nvidia.com/gpu=all \
        -p 8000:8000 \
        --ipc=host \
        -v /n/netscratch/juncheng_lab/muxint/llm_models:/models:Z \
        docker.io/vllm/vllm-openai:latest \
        --model /models/Qwen_Qwen3-Coder-480B-A35B-Instruct-FP8 \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.95
    ```

    - Note that you should change the port name and gpu model name. You need to change the tensor-parallel-size based on the number of allocated node .
4. create the ssh tunnel

    ```
    ssh -N -f -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=60 -R 0.0.0.0:8000:localhost:8000 murphy@freeinference.org
    # you need to change the port name

    # run this monitor script could continuously maintain the ssh
    nohup ./tunnel_monitor.sh > /dev/null 2>&1 &
    ```

5. check if the reverse proxy work

    ```
    curl http://freeinference.org:8001/v1/models
    ```


## **docker**

```
# save docker images
podman save docker.io/vllm/vllm-openai:latest -o /n/netscratch/juncheng_lab/muxint/vllm-openai-latest.tar

# load docker images
podman load -i /n/netscratch/juncheng_lab/muxint/vllm-openai-latest.tar

# check existed images
podman images
```
