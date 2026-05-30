# SGLang Parallel Configuration Guide

This guide summarizes common SGLang serving configurations for TP, PP, DP,
EP, DeepEP, DP attention, and MoE-DP. It is intended as a practical checklist
for benchmark cases in this repository.

References:

- SGLang server arguments: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
- SGLang DP/DPA/SMG guide: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/dp_dpa_smg_guide.md
- SGLang expert parallelism guide: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md
- SGLang PD disaggregation guide: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md

## 1. Quick Map

| Mode | Main flags | Best fit | Main advantage | Main cost |
| --- | --- | --- | --- | --- |
| Single GPU | none | Small models, correctness smoke tests | Simplest baseline | Limited memory/throughput |
| TP | `--tp N` / `--tp-size N` | Dense models, general multi-GPU serving | Shards weights/compute across GPUs | Collective communication, KV cache may be duplicated |
| PP | `--pp-size N` | Very large dense models or limited per-GPU memory | Shards layers, lowers per-GPU weight memory | Pipeline bubbles, needs enough batch/micro-batch |
| Native DP | `--dp-size N` with `sglang.launch_server` | Debug/legacy only | Simple in-process replicas | Not recommended for production; weak routing/HA |
| SMG DP | `python -m sglang_router.launch_server --dp-size N` | Production DP, shared-prefix workloads | Cache-aware routing, health checks, metrics | Extra router component |
| DPA | `--dp-size N --enable-dp-attention` | MLA/MoE models such as DeepSeek, Kimi, MiniMax; also Qwen | Reduces KV duplication, improves high-batch throughput | Requires valid `tp_size % dp_size == 0`; less useful for low-batch latency |
| EP | `--ep N` / `--ep-size N` | MoE models | Shards expert weights and expert compute | All-to-all routing cost |
| DeepEP | `--moe-a2a-backend deepep` | Large-scale MoE EP | Optimized token dispatch/combine | Backend constraints; needs compatible hardware/build |
| DPA + EP | `--tp N --dp-size D --ep E --enable-dp-attention --moe-a2a-backend deepep` | DeepSeek-class MoE at scale | Separates attention DP from expert sharding | Most complex topology |
| MoE-DP | `--moe-dp-size N` | Advanced MoE topology tuning | Decouples MoE data parallelism from other parallel dimensions | Newer/advanced path; verify version support |
| Attention CP | `--attn-cp-size N` | Very long context attention | Splits context dimension for attention | More collectives and tuning complexity |
| PD disaggregation | `--disaggregation-mode prefill/decode` + router | Workloads where prefill and decode interfere | Separate compute-heavy prefill from memory-bound decode | KV transfer and multi-service orchestration |

## 2. Parameter Meanings

### TP: Tensor Parallelism

Flags:

```bash
--tp 8
# aliases: --tp-size, --tensor-parallel-size
```

TP splits tensor operations and model weights across GPUs. It is the default
multi-GPU choice for dense models and for first bring-up of large models.

Use TP when:

- The model does not fit on one GPU.
- You need a safe, broadly supported multi-GPU baseline.
- The model is dense, or MoE-specific EP tuning is not ready yet.

Avoid overusing TP when:

- KV cache capacity is the bottleneck, especially for MLA models.
- Interconnect bandwidth is weak.
- You need many independent replicas for throughput; SMG DP may be better.

### PP: Pipeline Parallelism

Flags:

```bash
--pp-size 2
# alias: --pipeline-parallel-size
--pp-max-micro-batch-size 8
--pp-async-batch-depth 1
```

PP splits model layers across pipeline stages. It helps when model weights are
too large even after TP, or when a single node cannot hold all layers.

Use PP when:

- Weight memory is the primary bottleneck.
- TP alone cannot fit the model.
- Batch/concurrency is high enough to hide pipeline bubbles.

Trade-offs:

- Small batch/low concurrency can suffer from bubbles.
- Debugging is harder than pure TP.
- Tune micro-batch size and async depth rather than assuming PP is faster.

### Native DP

Flags:

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --dp-size 4
```

Native DP creates multiple replicas inside one SGLang instance. The official
guide currently recommends SMG for production DP because native DP lacks
production-grade routing, observability, cache-aware routing, and fault
tolerance.

Use native DP only for:

- Local experiments.
- Legacy RL or internal workflows that already depend on it.
- Quick behavior checks where production routing is irrelevant.

### SMG-Based DP

Flags:

```bash
python -m sglang_router.launch_server \
  --model-path /path/to/model \
  --dp-size 4 \
  --router-policy cache_aware \
  --host 0.0.0.0 \
  --port 30000
```

SMG is the recommended production data-parallel router. It can launch or route
to multiple workers and offers cache-aware routing, health checking, metrics,
and better operational control.

Use SMG DP when:

- The model fits per replica or per TP group.
- Throughput scales with more replicas.
- Workload has shared prefixes and cache locality matters.
- You need production routing, retries, health checks, and metrics.

### DPA: Data Parallel Attention

Flags:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --dp-size 8 \
  --enable-dp-attention
```

DPA applies data parallelism to attention while preserving tensor/expert
parallelism for other parts. SGLang documents DPA as especially useful for MLA
models because KV cache duplication under TP can limit batch size and decode
throughput.

Rules and constraints:

- Set both `--dp-size` and `--enable-dp-attention`.
- `dp_size` must be greater than 1.
- The documented constraint is `tp_size % dp_size == 0`.
- If `dp_size == 1`, DPA is effectively disabled.

Use DPA when:

- Serving MLA models such as DeepSeek, MiniMax, Kimi-K2.
- Long context or high concurrency makes KV cache memory the bottleneck.
- You target throughput rather than minimum single-request latency.

Be cautious when:

- Batch is tiny and latency is the only objective.
- The model is a standard GQA dense model where TP/SMG DP is simpler.

### EP: Expert Parallelism

Flags:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --ep 8
```

EP distributes MoE expert weights across GPUs. It is useful when expert weights
are too large or when expert compute should be distributed across GPUs.

Use EP when:

- The model is MoE.
- Expert weights dominate memory.
- Token routing can benefit from dedicated expert placement.

Common EP backend flags:

```bash
--moe-a2a-backend none|deepep|mooncake|mori|nixl|ascend_fuseep
--moe-runner-backend auto|deep_gemm|triton|cutlass|flashinfer_trtllm|flashinfer_cutlass|flashinfer_mxfp4
```

### DeepEP

Flags:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --ep 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto
```

DeepEP is an all-to-all backend for efficient MoE token dispatch and combine.
The official EP guide recommends `--deepep-mode auto` for automatic switching:
`normal` favors prefill throughput, and `low_latency` favors decode latency.

Important backend constraint:

- DeepEP/Mooncake/NIXL-EP/MORI/Ascend fused EP currently require
  `ep_size == tp_size`.
- If `ep_size < tp_size`, use `--moe-a2a-backend none` unless your local build
  explicitly supports another hybrid path.

Use DeepEP when:

- Large-scale MoE EP is enabled.
- Interconnect supports efficient all-to-all.
- You need higher throughput than fallback all-reduce/all-gather routing.

Add overlap only after baseline is stable:

```bash
--enable-two-batch-overlap
--enable-single-batch-overlap
```

### DPA + EP for MoE

Typical DeepSeek-style template:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --dp-size 8 \
  --ep 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm
```

This combines:

- DP attention: attention/KV cache is partitioned by DP workers.
- EP: experts are sharded across GPUs.
- DeepEP: MoE token routing uses optimized all-to-all.

Use this when:

- The model is a large MoE with MLA attention.
- You are serving high concurrency/high throughput workloads.
- KV cache memory and expert memory are both bottlenecks.

### MoE-DP

Flags:

```bash
--moe-dp-size 1
# alias: --moe-data-parallel-size
```

MoE-DP controls the data parallelism size for MoE layers. It is an advanced
dimension exposed separately from general `--dp-size`. In modern SGLang
topologies, it can be used with attention context parallelism or DPA to tune
attention and MoE dimensions independently.

Use MoE-DP when:

- You are already tuning MoE topology beyond simple TP/EP.
- Attention and MoE have different bottlenecks.
- You need to decouple `attention_cp_size`, DPA, and MoE layout.

Guidance:

- Start with `--moe-dp-size 1` unless you have a specific reason.
- Verify exact support in your installed SGLang with:

```bash
python -m sglang.launch_server --help | grep -E 'moe-dp|attention-context|dp-attention'
```

### Attention Context Parallelism

Flags:

```bash
--attn-cp-size 2
# alias: --attention-context-parallel-size
```

Attention CP splits the context dimension for attention. It is useful for very
long context workloads where attention memory/compute dominates, but it adds
communication and should be benchmarked carefully.

Use it when:

- Long context dominates prefill.
- Other memory reductions are insufficient.
- Your SGLang version and model architecture support it well.

### PD Disaggregation

Flags:

```bash
# Prefill worker
python -m sglang.launch_server \
  --model-path /path/to/model \
  --disaggregation-mode prefill \
  --port 30000

# Decode worker
python -m sglang.launch_server \
  --model-path /path/to/model \
  --disaggregation-mode decode \
  --port 30001

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --port 8000
```

Use PD when:

- Prefill-heavy requests interrupt decode and hurt TPOT/tail latency.
- You want independent prefill/decode scaling.
- You need different knobs per stage, for example larger prefill batches and
  stricter decode `--max-running-requests`.

Costs:

- KV transfer path must be stable.
- More services and logs.
- More failure modes than IFB/non-PD.

## 3. Decision Flow

1. Start with a fit check.
   - If the model fits on one GPU: single-GPU baseline.
   - If not: TP.
   - If TP still cannot fit: add PP or quantization.

2. Identify model type.
   - Dense GQA: TP or SMG DP.
   - MLA dense: DPA can help at high throughput.
   - MoE: evaluate EP; for large MoE, evaluate DeepEP.
   - MoE + MLA: DPA + EP is the main high-throughput path.

3. Identify workload.
   - Low QPS/latency: avoid unnecessary DP/PP; keep topology simple.
   - High throughput: SMG DP, DPA, EP overlap knobs.
   - Shared prefix: SMG cache-aware routing and radix cache.
   - Long context: tune `--chunked-prefill-size`, `--page-size`,
     `--attn-cp-size`, and KV cache dtype.
   - Mixed long prefill + decode: consider PD disaggregation.

4. Choose production routing.
   - For DP production serving, prefer SMG over native DP.
   - Enable metrics and worker health checks.

5. Benchmark before combining more dimensions.
   - Baseline: TP only.
   - Add one dimension at a time: DPA, EP, DeepEP, overlap, PP, CP.
   - Track TTFT, TPOT, P90/P99, RPS, output throughput, total throughput,
     cache hit, GPU memory, and OOM/errors.

## 4. Common Templates

### Dense Model, Single Node TP

```bash
python -m sglang.launch_server \
  --model-path /model/dense \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

### Dense Model, Production DP with SMG

```bash
python -m sglang_router.launch_server \
  --model-path /model/dense \
  --dp-size 4 \
  --tp 2 \
  --router-policy cache_aware \
  --host 0.0.0.0 \
  --port 30000
```

### Large MoE, EP + DeepEP

```bash
python -m sglang.launch_server \
  --model-path /model/moe \
  --tp 8 \
  --ep 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto \
  --host 0.0.0.0 \
  --port 30000
```

### DeepSeek/Qwen MoE High Throughput: DPA + EP

```bash
python -m sglang.launch_server \
  --model-path /model/moe-mla \
  --tp 8 \
  --dp-size 8 \
  --ep 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto \
  --host 0.0.0.0 \
  --port 30000
```

### Long Context Debug Template

```bash
python -m sglang.launch_server \
  --model-path /model/long-context \
  --tp 8 \
  --chunked-prefill-size 8192 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

### Multi-Node TP

```bash
# node 0
python -m sglang.launch_server \
  --model-path /model/dense \
  --tp 16 \
  --dist-init-addr node0:50000 \
  --nnodes 2 \
  --node-rank 0

# node 1
python -m sglang.launch_server \
  --model-path /model/dense \
  --tp 16 \
  --dist-init-addr node0:50000 \
  --nnodes 2 \
  --node-rank 1
```

## 5. How to Add These Modes to This Benchmark Project

In `configs/cases.yaml` or `configs/ifb_matrix.yaml`, map the fields as:

```yaml
tp_size: 8          # --tp-size / --tp
pp_size: 1          # --pp-size
dp_size: 8          # --dp-size
ep_size: 8          # --ep-size / --ep
atten_cp_size: 1    # --attn-cp-size
quantization: null
extra_args:
  - --enable-dp-attention
  - --enable-dp-lm-head
  - --moe-a2a-backend
  - deepep
  - --moe-runner-backend
  - deep_gemm
  - --deepep-mode
  - auto
```

For advanced MoE-DP:

```yaml
extra_args:
  - --moe-dp-size
  - "1"
```

For SMG DP, use the router entrypoint rather than `sglang.launch_server`. The
current IFB runner is aimed at one SGLang server process; for production SMG
benchmarks, add a separate deploy wrapper that starts workers and router, then
points the benchmark client at the router port.

## 6. Validation Checklist

Before full benchmark:

- `curl http://host:port/health`
- `curl http://host:port/v1/models`
- Confirm GPUs are selected as expected.
- Confirm effective parallel settings in `server.log`.
- Confirm request path: `/v1/completions` or `/v1/chat/completions`.
- For radix cache, inspect `#cached-token` and `#new-token` in `Prefill batch`.
- For DPA, confirm `dp_size > 1` and no warning says DPA is disabled.
- For DeepEP, confirm backend initialization succeeds and `ep_size == tp_size`
  unless using a documented hybrid fallback.
- For PD, confirm prefill/decode/router all return healthy status and KV
  transfer has no repeated timeout or OOM messages.

Benchmark order:

1. Single short smoke: bs=1, few prompts.
2. Cache-hit smoke if shared-prefix workload.
3. One representative SLA run.
4. Full concurrency scan.
5. Repeat the best points to check stability.

## 7. Practical Recommendations

- Dense model, single-node: start with TP.
- Dense model, many replicas: use SMG DP with cache-aware routing.
- MLA model: evaluate DPA if throughput and KV memory are bottlenecks.
- MoE model: start with TP-only baseline, then EP, then DeepEP.
- Large MoE MLA model: DPA + EP + DeepEP is the main high-throughput target.
- Very long context: tune chunked prefill, KV dtype, page size, and possibly
  attention CP.
- Mixed prefill/decode workload with bad TPOT tail: evaluate PD disaggregation.
- Production DP: prefer SMG over native DP.
