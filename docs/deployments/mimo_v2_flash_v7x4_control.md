# MiMo V2 Flash 单机 v7x-4 部署（control test）

## 适用场景

- TPU v7x，单机 4 chips（每 chip 2 cores → tp=8）
- 验证 sgl_jax MoE / attention 实现整体正确性（Flash 已知数值正确，可作 Pro 对照基线）
- 复用集群 `tpuv7x-64-node`、节点池 `v7x-4-pool`、PVC `inference-model-storage-poc-tpu-hns-pvc`

## launch 命令

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 8 --dp-size 2 --ep-size 8 \
  --moe-backend fused \
  --host 0.0.0.0 --port 30271 \
  --page-size 256 \
  --context-length 262144 \
  --disable-radix-cache \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 \
  --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup \
  --log-level info \
  --max-running-requests 128 \
  --dp-schedule-policy round_robin
```

注意：v7x 一个 chip 两个 core，`tp-size=8` 用满 4 chips（8 cores）。`dp-size`、`ep-size` 与 tp 独立。

## Pod manifest（`/tmp/v7x-mimo-flash.yaml`）

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: v7x-mimo-flash
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: tpu7x
    cloud.google.com/gke-tpu-topology: 2x2x1
    cloud.google.com/gke-nodepool: v7x-4-pool
  serviceAccountName: gcs-account
  containers:
  - name: v7x-mimo-flash
    image: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
    command: ["bash","-lc"]
    args:
    - |
      set -e
      cd /tmp
      git clone https://github.com/primatrix/sglang-jax.git
      cd sglang-jax
      git fetch origin feat/mimo-v2-pro
      git checkout feat/mimo-v2-pro
      pip install -e .
      JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
        --model-path /models/MiMo-V2-Flash \
        --trust-remote-code \
        --tp-size 8 --dp-size 2 --ep-size 8 \
        --moe-backend fused \
        --host 0.0.0.0 --port 30271 \
        --page-size 256 --context-length 262144 \
        --disable-radix-cache --chunked-prefill-size 4096 \
        --dtype bfloat16 --mem-fraction-static 0.95 \
        --swa-full-tokens-ratio 0.2 --skip-server-warmup \
        --log-level info --max-running-requests 128 \
        --dp-schedule-policy round_robin
    ports:
    - containerPort: 30271
      name: http
    resources:
      requests:
        google.com/tpu: 4
      limits:
        google.com/tpu: 4
    volumeMounts:
    - name: model-storage
      mountPath: /models
    - name: dev-shm
      mountPath: /dev/shm
  volumes:
  - name: dev-shm
    emptyDir:
      medium: Memory
  - name: gke-gcsfuse-cache
    emptyDir:
      medium: Memory
  - name: model-storage
    persistentVolumeClaim:
      claimName: inference-model-storage-poc-tpu-hns-pvc
```

## 部署 + 健康检查

```bash
kubectl apply -f /tmp/v7x-mimo-flash.yaml
kubectl wait --for=condition=Ready pod/v7x-mimo-flash --timeout=900s
kubectl logs -f v7x-mimo-flash
```

健康检查：

```bash
kubectl exec v7x-mimo-flash -- curl -s http://localhost:30271/health
```

## Coherence test（确定性 generate）

```bash
for prompt in "The capital of France is" "2+2=" "Write a haiku about autumn:"; do
  kubectl exec v7x-mimo-flash -- curl -s http://localhost:30271/generate \
    -d "{\"text\":\"$prompt\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":32}}"
  echo
done
```

期望：英文连贯输出，无单 token 重复。

## Teardown

```bash
kubectl delete pod v7x-mimo-flash
```
