# SGLang 并行配置指南

本文整理 SGLang 服务部署中常见的 TP、PP、DP、SMG DP、DPA、EP、DeepEP、MoE-DP、Attention CP 和 PD 分离配置方式，说明它们的适用场景、优势、代价，以及在本 benchmark 工程中的配置方式。

参考资料：

- SGLang server arguments: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
- SGLang DP/DPA/SMG guide: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/dp_dpa_smg_guide.md
- SGLang expert parallelism guide: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md
- SGLang PD disaggregation guide: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md

## 1. 配置速查

| 模式 | 主要参数 | 适用场景 | 主要优势 | 主要代价 |
| --- | --- | --- | --- | --- |
| 单卡 | 无 | 小模型、功能 smoke、正确性验证 | 最简单，最容易排障 | 显存和吞吐受单卡限制 |
| TP | `--tp N` / `--tp-size N` | Dense 模型、通用多卡部署 | 切分权重和矩阵计算，通用性强 | 通信开销增加，KV cache 可能重复 |
| PP | `--pp-size N` | 超大 Dense 模型、单组 TP 仍放不下 | 按层切分，降低单卡权重显存 | pipeline bubble，需要足够 batch 才能摊薄 |
| Native DP | `--dp-size N` + `sglang.launch_server` | 本地实验、遗留链路 | 一个进程内起多个 replica，验证方便 | 官方不推荐生产使用，路由和可观测性弱 |
| SMG DP | `python -m sglang_router.launch_server --dp-size N` | 生产 DP、共享前缀负载 | cache-aware 路由、健康检查、metrics | 多一个 router 组件 |
| DPA | `--dp-size N --enable-dp-attention` | MLA / MoE 模型，高并发长上下文 | 减少 KV 重复，提升高 batch 吞吐 | 要满足 `tp_size % dp_size == 0`，低 batch 不一定收益 |
| EP | `--ep N` / `--ep-size N` | MoE 模型 | 切分 expert 权重和 expert 计算 | token all-to-all 路由成本 |
| DeepEP | `--moe-a2a-backend deepep` | 大规模 MoE EP | 优化 token dispatch/combine | 对硬件、构建和拓扑有要求 |
| DPA + EP | `--tp N --dp-size D --ep E --enable-dp-attention --moe-a2a-backend deepep` | DeepSeek 类大 MoE 模型 | attention DP 和 expert sharding 同时生效 | 拓扑最复杂，需要逐项验证 |
| MoE-DP | `--moe-dp-size N` | 高级 MoE 拓扑调优 | 单独调 MoE 层的数据并行维度 | 新路径，需确认版本支持 |
| Attention CP | `--attn-cp-size N` | 超长上下文 attention | 切分 attention context 维度 | 通信更多，调优复杂 |
| PD 分离 | `--disaggregation-mode prefill/decode` + router | prefill 和 decode 相互干扰的场景 | prefill/decode 独立扩缩容 | KV transfer 和多服务编排复杂 |

## 2. 各配置方式说明

### TP: Tensor Parallelism

常用参数：

```bash
--tp 8
# 别名：--tp-size, --tensor-parallel-size
```

TP 会把模型权重和 tensor 计算切分到多张 GPU 上，是 Dense 模型和大模型首次多卡部署时最常用的基线方案。

适合使用 TP 的情况：

- 模型单卡放不下。
- 需要一个稳定、通用、容易验证的多卡基线。
- 模型是 Dense 模型，或者 MoE 的 EP / DeepEP 还没有调通。

不宜盲目增大 TP 的情况：

- KV cache 容量是瓶颈，尤其是 MLA 模型。
- 卡间互联带宽较弱。
- 目标是多副本吞吐扩展，此时 SMG DP 可能更合适。

### PP: Pipeline Parallelism

常用参数：

```bash
--pp-size 2
# 别名：--pipeline-parallel-size
--pp-max-micro-batch-size 8
--pp-async-batch-depth 1
```

PP 按层切分模型，把不同层放到不同 pipeline stage 上。它适合权重显存压力特别大的模型，尤其是 TP 后仍然放不下时。

适合使用 PP 的情况：

- 权重显存是主要瓶颈。
- 单纯 TP 仍无法加载模型。
- batch / concurrency 足够高，可以摊薄 pipeline bubble。

代价：

- 小 batch、低并发时可能因为 pipeline bubble 导致延迟变差。
- 排障难度高于纯 TP。
- 需要调 `--pp-max-micro-batch-size` 和 `--pp-async-batch-depth`，不能默认认为 PP 一定更快。

### Native DP

常用参数：

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --dp-size 4
```

Native DP 会在一个 SGLang 实例中创建多个数据并行 replica。官方文档更推荐生产环境使用 SMG DP，因为 Native DP 在生产路由、可观测性、cache-aware 路由和故障隔离方面较弱。

Native DP 适合：

- 本地实验。
- 已经依赖 Native DP 的遗留 RL 或内部链路。
- 快速验证行为，不关注生产路由能力。

### SMG DP

常用参数：

```bash
python -m sglang_router.launch_server \
  --model-path /path/to/model \
  --dp-size 4 \
  --router-policy cache_aware \
  --host 0.0.0.0 \
  --port 30000
```

SMG 是官方推荐的生产 DP 路由方式。它可以启动或路由到多个 worker，并提供 cache-aware routing、健康检查、metrics 和更好的运维控制。

适合使用 SMG DP 的情况：

- 模型可以在每个 replica 或每个 TP group 内放下。
- 增加 replica 能线性或近似线性提升吞吐。
- 负载有 shared prefix，cache locality 很重要。
- 需要生产级路由、重试、健康检查和 metrics。

### DPA: Data Parallel Attention

常用参数：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --dp-size 8 \
  --enable-dp-attention
```

DPA 会对 attention 部分使用数据并行，同时保留其他部分的 tensor/expert 并行。SGLang 文档中强调，DPA 对 MLA 模型尤其有用，因为纯 TP 下 KV cache 可能被重复保存，导致 batch size 和 decode throughput 受限。

规则和约束：

- 需要同时设置 `--dp-size` 和 `--enable-dp-attention`。
- `dp_size` 必须大于 1。
- 文档约束为 `tp_size % dp_size == 0`。
- 如果 `dp_size == 1`，DPA 实际上不会带来效果。

适合使用 DPA 的情况：

- 服务 DeepSeek、MiniMax、Kimi-K2 等 MLA 模型。
- 长上下文或高并发导致 KV cache 显存成为瓶颈。
- 目标更偏吞吐，而不是极低单请求延迟。

需要谨慎的情况：

- batch 很小，只关注单请求 latency。
- 模型是标准 GQA Dense 模型，此时 TP 或 SMG DP 更简单。

### EP: Expert Parallelism

常用参数：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --ep 8
```

EP 用于 MoE 模型，会把 expert 权重分布到不同 GPU 上。它适合 expert 权重较大、expert 计算需要跨卡分摊的场景。

适合使用 EP 的情况：

- 模型是 MoE。
- expert 权重占显存较大。
- token routing 可以受益于独立的 expert placement。

常见 EP 后端参数：

```bash
--moe-a2a-backend none|deepep|mooncake|mori|nixl|ascend_fuseep
--moe-runner-backend auto|deep_gemm|triton|cutlass|flashinfer_trtllm|flashinfer_cutlass|flashinfer_mxfp4
```

### DeepEP

常用参数：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --ep 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto
```

DeepEP 是面向 MoE token dispatch/combine 的 all-to-all 后端。官方 EP 文档建议使用 `--deepep-mode auto` 自动切换模式：`normal` 更偏 prefill 吞吐，`low_latency` 更偏 decode 延迟。

重要约束：

- DeepEP、Mooncake、NIXL-EP、MORI、Ascend fused EP 当前通常要求 `ep_size == tp_size`。
- 如果 `ep_size < tp_size`，除非本地构建明确支持其他混合路径，否则建议使用 `--moe-a2a-backend none`。

适合使用 DeepEP 的情况：

- 已经启用大规模 MoE EP。
- 硬件互联适合高效 all-to-all。
- fallback all-reduce / all-gather 路径吞吐不足。

重叠优化建议在 baseline 稳定后再加：

```bash
--enable-two-batch-overlap
--enable-single-batch-overlap
```

### DPA + EP

DeepSeek 类 MoE 模型的典型模板：

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

这个组合同时启用：

- DP attention：attention 和 KV cache 按 DP worker 分摊。
- EP：expert 权重和 expert 计算跨 GPU 切分。
- DeepEP：MoE token routing 使用优化的 all-to-all。

适合使用 DPA + EP 的情况：

- 模型是带 MLA attention 的大 MoE。
- 目标是高并发、高吞吐。
- KV cache 显存和 expert 权重显存都是瓶颈。

### MoE-DP

常用参数：

```bash
--moe-dp-size 1
# 别名：--moe-data-parallel-size
```

MoE-DP 控制 MoE 层的数据并行大小，是独立于通用 `--dp-size` 暴露出来的高级维度。在现代 SGLang 拓扑中，它可以和 Attention CP、DPA 等一起使用，用于单独调 attention 和 MoE 的并行布局。

适合使用 MoE-DP 的情况：

- 已经在调 TP / EP 之外的 MoE 高级拓扑。
- attention 和 MoE 的瓶颈不同。
- 需要解耦 `attention_cp_size`、DPA 和 MoE layout。

建议：

- 没有明确原因时先保持 `--moe-dp-size 1`。
- 先确认当前安装的 SGLang 是否支持相关参数：

```bash
python -m sglang.launch_server --help | grep -E 'moe-dp|attention-context|dp-attention'
```

### Attention Context Parallelism

常用参数：

```bash
--attn-cp-size 2
# 别名：--attention-context-parallel-size
```

Attention CP 会切分 attention 的 context 维度。它适合超长上下文场景，尤其是 attention 显存和计算成为主要瓶颈时。但它也会引入额外通信，因此必须通过 benchmark 验证。

适合使用 Attention CP 的情况：

- 长上下文 prefill 占主要耗时。
- 其他显存优化仍不足。
- 当前 SGLang 版本和模型架构对该参数支持良好。

### PD 分离

常用参数：

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

适合使用 PD 分离的情况：

- prefill-heavy 请求会打断 decode，导致 TPOT 或尾延迟变差。
- 希望 prefill 和 decode 独立扩缩容。
- 希望不同阶段使用不同参数，例如 prefill 批量更大、decode 对 `--max-running-requests` 更严格。

代价：

- KV transfer 链路必须稳定。
- 服务数量和日志更多。
- 比 IFB / 非 PD 模式有更多故障点。

## 3. 配置决策流程

1. 先确认模型能否放下。
   - 单卡能放下：先做单卡 baseline。
   - 单卡放不下：先用 TP。
   - TP 后仍放不下：考虑 PP 或量化。

2. 再判断模型类型。
   - Dense GQA：优先 TP 或 SMG DP。
   - MLA Dense：高吞吐场景评估 DPA。
   - MoE：评估 EP，大 MoE 进一步评估 DeepEP。
   - MoE + MLA：高吞吐目标通常走 DPA + EP。

3. 再判断业务负载。
   - 低 QPS / 低延迟：避免过度并行，拓扑越简单越好。
   - 高吞吐：考虑 SMG DP、DPA、EP、overlap。
   - 共享前缀明显：使用 radix cache，生产 DP 可考虑 SMG cache-aware 路由。
   - 长上下文：调 `--chunked-prefill-size`、`--page-size`、`--attn-cp-size`、KV cache dtype。
   - 长 prefill 和 decode 混部互相影响：考虑 PD 分离。

4. 生产 DP 优先选择 SMG。
   - 比 Native DP 更适合路由、健康检查、metrics 和故障隔离。

5. 每次只增加一个并行维度。
   - 基线：TP only。
   - 逐步加入：DPA、EP、DeepEP、overlap、PP、CP。
   - 每步记录 TTFT、TPOT、P90/P99、RPS、输出吞吐、总吞吐、cache hit、GPU 显存和 OOM/error。

## 4. 常见命令模板

### Dense 模型，单节点 TP

```bash
python -m sglang.launch_server \
  --model-path /model/dense \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000
```

### Dense 模型，SMG 生产 DP

```bash
python -m sglang_router.launch_server \
  --model-path /model/dense \
  --dp-size 4 \
  --tp 2 \
  --router-policy cache_aware \
  --host 0.0.0.0 \
  --port 30000
```

### 大 MoE，EP + DeepEP

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

### DeepSeek / Qwen MoE 高吞吐，DPA + EP

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

### 长上下文调试模板

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

### 多节点 TP

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

## 5. 在本工程中的配置方式

在 `configs/cases.yaml` 或 `configs/ifb_matrix.yaml` 中，可以按下面方式映射：

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

高级 MoE-DP：

```yaml
extra_args:
  - --moe-dp-size
  - "1"
```

SMG DP 需要使用 router 入口，而不是直接 `sglang.launch_server`。当前 IFB runner 主要面向单个 SGLang server 进程；如果要做生产 SMG benchmark，建议增加单独 deploy wrapper，先启动 workers 和 router，再让压测客户端访问 router 端口。

## 6. 全量测试前检查清单

服务检查：

- `curl http://host:port/health`
- `curl http://host:port/v1/models`
- 确认 GPU 可见性和卡数符合预期。
- 确认 `server.log` 中实际生效的并行参数。
- 确认请求路径是 `/v1/completions` 或 `/v1/chat/completions`。

功能检查：

- radix cache：看 `Prefill batch` 中的 `#cached-token` 和 `#new-token`。
- DPA：确认 `dp_size > 1`，且日志没有提示 DPA 被禁用。
- DeepEP：确认后端初始化成功；除非本地构建明确支持混合路径，否则保证 `ep_size == tp_size`。
- PD：确认 prefill、decode、router 都健康，KV transfer 没有反复 timeout 或 OOM。

推荐测试顺序：

1. bs=1、少量 prompt 的 smoke。
2. 如果是 shared-prefix 负载，先跑 cache-hit smoke。
3. 跑一个代表性 SLA 点。
4. 放开完整并发扫描。
5. 对最优点重复测试，确认稳定性。

## 7. 实用建议

- Dense 单节点：从 TP 开始。
- Dense 多副本：优先 SMG DP，并使用 cache-aware routing。
- MLA 模型：如果吞吐和 KV 显存是瓶颈，评估 DPA。
- MoE 模型：先 TP-only baseline，再 EP，再 DeepEP。
- 大 MoE + MLA：DPA + EP + DeepEP 通常是高吞吐目标配置。
- 超长上下文：重点调 chunked prefill、KV dtype、page size，必要时评估 Attention CP。
- prefill/decode 混部导致 TPOT 尾延迟差：评估 PD 分离。
- 生产 DP：优先 SMG，不建议长期使用 Native DP。
