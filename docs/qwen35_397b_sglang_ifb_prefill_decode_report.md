# Qwen3.5-397B-FP8 SGLang IFB Prefill/Decode 测试报告

## 测试范围

- 测试时间：2026-05-31
- 节点：`guobj@10.16.1.36`
- 容器：`gbj_sgl0522`
- 目录：`/mnt11/task/sgl`
- 模型：`/model/qwen3.5/Qwen3.5-397B-A17B-Channel-FP8`
- 模式：SGLang IFB，非 PD 分离，关闭 radix-cache
- GPU：通过 `hy-smi` 确认 0-7 共 8 张卡空闲后使用，正式配置为 `tp_size=8`
- SLA：prefill `mean TTFT < 3s`，decode `mean TPOT < 75ms`

正式结果文件：

- CSV：`reports/sglang_ifb_tp8_formal/sweep_result.csv`
- HTML：`reports/sglang_ifb_tp8_formal/sweep_result.html`
- 本次配置：`configs/sglang_ifb_tp8_formal.yaml`

## 启动配置

核心服务参数：

```bash
python3 -m sglang.launch_server \
  --model-path /model/qwen3.5/Qwen3.5-397B-A17B-Channel-FP8 \
  --served-model-name qwen3-397-fp8 \
  --host 10.16.1.36 \
  --port 30005 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.82 \
  --attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --chunked-prefill-size -1 \
  --skip-server-warmup \
  --cuda-graph-max-bs 32 \
  --decode-log-interval 20 \
  --disable-radix-cache
```

说明：

- `hy-smi` 显示 8 卡可用，因此正式测试不再限制为 4 卡。
- 服务启动后显存约 92%-99%，测试结束后 runner 已清理服务，8 卡显存恢复 0%。
- 服务端日志中 prefill 阶段 `#cached-token: 0`，符合关闭 radix-cache 的预期。

## Prefill 结果

### 输入 4096，输出 1

| target request-rate | actual RPS | mean TTFT ms | P90 TTFT ms | P99 TTFT ms | total token throughput | SLA |
|---:|---:|---:|---:|---:|---:|:---:|
| 3.5 | 3.427 | 2149.88 | 3316.01 | 3733.51 | 14040.20 | 通过 |
| 3.75 | 3.644 | 2295.36 | 3483.31 | 4034.05 | 14930.94 | 通过 |
| 4.0 | 3.763 | 2985.54 | 4279.75 | 4754.25 | 15415.88 | 通过 |
| 4.25 | 3.813 | 3931.91 | 5292.64 | 6212.63 | 15623.39 | 失败 |
| 4.5 | 3.810 | 4882.46 | 6786.78 | 8182.36 | 15609.90 | 失败 |
| 5.0 | 3.813 | 6472.78 | 9594.75 | 11428.34 | 15620.69 | 失败 |

结论：

- SLA 内最优点是 target request-rate `4.0`，实际 `Request throughput = 3.763 req/s`。
- target request-rate 继续升高后，实际 RPS 基本停在 `3.81 req/s` 左右，但 mean TTFT 快速恶化，说明已经到 prefill 排队拐点。
- P90/P99 在通过点也高于 3s，因此如果 SLA 后续要求尾延迟，需要把目标 RPS 降到 `3.5-3.75` 区间。

### 输入 65536，输出 1

| target request-rate | actual RPS | mean TTFT ms | P90 TTFT ms | P99 TTFT ms | total token throughput | SLA |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.05 | 0.123 | 8501.57 | 12225.04 | 12319.51 | 8051.31 | 失败 |
| 0.1 | 0.174 | 10699.46 | 17912.71 | 19308.58 | 11428.10 | 失败 |
| 0.2 | 0.217 | 12490.27 | 21424.42 | 23468.59 | 14196.97 | 失败 |

结论：

- TP8 IFB 下，65536 输入长度即使在最低测试负载下 mean TTFT 也超过 3s。
- 该长度需要继续尝试 PP/CP 或更激进的长上下文优化；本轮 TP8 不满足 prefill SLA。

## Decode 结果

输入 1，输出 1024，`request-rate=inf`，主要看 mean TPOT、P90/P99、生成吞吐。

| max concurrency | actual RPS | mean TPOT ms | P99 TPOT ms | output token throughput | total token throughput | SLA decode active req capacity |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.906 | 46.78 | 66.00 | 927.30 | 928.21 | 69.55 |
| 192 | 0.921 | 46.82 | 65.98 | 942.85 | 943.77 | 70.71 |
| 256 | 0.911 | 47.33 | 66.72 | 932.89 | 933.80 | 69.97 |
| 384 | 0.920 | 46.79 | 65.87 | 942.05 | 942.97 | 70.65 |
| 512 | 0.918 | 47.00 | 66.37 | 939.93 | 940.85 | 70.50 |

说明：

- `request-rate=inf` 会一次性压入请求，客户端 mean TTFT 主要反映排队，不作为 decode 优化目标。
- Decode SLA 内最优吞吐点是 `mc=192`，output throughput `942.85 token/s`，mean TPOT `46.82ms`，P99 TPOT `65.98ms`。
- 按 `1000 / 75 = 13.33 token/s/req` 换算，单个 TP8 decode 实例在 75ms TPOT SLA 下约可承载 `942.85 / 13.33 = 70.71` 个活跃 decode 请求。

## PD 配比推导

以 `4096 -> 1024` 为目标请求形态估算：

- Prefill 最优实际 RPS：`3.763 req/s`
- 每个请求 decode 输出长度：`1024 tokens`
- Decode 最优生成吞吐：`942.85 token/s`
- 若希望 prefill 端按 `3.763 req/s` 连续输入，decode token 需求约为 `3.763 * 1024 = 3853 token/s`
- 需要的 TP8 decode 实例数约为 `3853 / 942.85 = 4.09`

因此：

- 若 P 和 D 都使用同等 TP8 实例，按 4096 输入、1024 输出估算，推荐配比约为 `1P:4D`。
- 如果只有两个 8 卡节点做 `1P1D`，decode 会成为瓶颈，端到端稳定 RPS 预计接近单 D 的完成能力，即约 `0.92 req/s`。
- 如果以用户给出的 `1000/75` 公式看活跃 decode 容量，单 D 约可承载 `70.71` 个活跃 decode 请求；对应 `3.763 req/s * 1024 * 0.075 = 289` 个活跃 decode 请求需求，也同样得到约 `289 / 70.71 = 4.09` 个 D 实例。

对 `65536 -> 1024`：

- TP8 prefill 已不满足 `mean TTFT < 3s`，当前不能直接推导可用 PD 配比。
- 需要优先找到 65536 输入长度下满足 SLA 的 prefill 切分方式，再做 P:D 配比。

## 其他切分方式的 smoke 结果

已用 8 卡做过多种切分方式 smoke，当前观察到的主要失败原因如下：

| 切分方式 | 现象 | 初步原因 |
|---|---|---|
| `cp8_mf09_cg16` | 启动失败 | FA3/page attention 断言 `num_heads<=num_kv_heads*48` |
| `tp8cp4_mf09_cg16` | 启动失败 | 同上，GQA/MQA 头数与 CP/FA3 组合不满足内核约束 |
| `tp8cp2_mf09_cg16` | 可启动但 4096 prefill 不达标，65536 出现失败 | CP2 在该配置下吞吐低于 TP8，长输入有稳定性/显存风险 |
| `ep8_aiter` | 启动失败 | `aiter_moe_fp8_w8a8` 找不到合适 backend |
| `tp8cp2ep4_aiter_mf09_cg16` | 启动失败 | 显存不足，日志建议调整 `--mem-fraction-static` |
| `deepep_ep8_aiter` | 启动失败 | scheduler 进程异常退出，疑似 DeepEP/当前后端组合不稳定 |

下一步建议：

- CP 类配置改用非 FA3 attention backend 或调整 page/head 参数后重试。
- EP/DeepEP 类配置先确认当前 SGLang 版本和 AITER kernel 对 Qwen3.5-397B FP8 MoE shape 的支持。
- PP 配置需要单独补跑；该模型的层切分和 FP8/MoE 权重加载方式可能与 TP-only 有差异，不能直接套用 TP8 参数。

## 当前结论

1. 在 8 卡 TP8 IFB、关闭 radix-cache 条件下，`4096 -> 1` prefill 的 SLA 内最优点为实际 `3.763 req/s`，总吞吐 `15415.88 token/s`。
2. `65536 -> 1` 在 TP8 IFB 下不满足 `mean TTFT < 3s`，需要换 PP/CP 或其他长上下文切分继续优化。
3. Decode `1 -> 1024` 的最优点约为 `mc=192`，mean TPOT `46.82ms`，P99 TPOT `65.98ms`，output throughput `942.85 token/s`。
4. 对 `4096 -> 1024`，若按同等 TP8 实例做 PD 分离，估算需要约 `1P:4D` 才能让 decode 不成为瓶颈；两节点 `1P1D` 时应将入口 RPS 控制在约 `0.9 req/s` 级别。
