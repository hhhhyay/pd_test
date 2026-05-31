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

## 多切分方式 smoke 结果

补跑配置：`configs/sglang_ifb_layout_repair_smoke.yaml`  
结果文件：`reports/sglang_ifb_layout_repair_smoke/sweep_result.csv`

### 可完成请求的切分

| 切分方式 | prefill target rr | actual RPS | mean TTFT ms | P99 TTFT ms | total token throughput | prefill SLA | decode mean TPOT ms | decode output throughput |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|
| `tp8` | 4.0 | 3.763 | 2985.54 | 4754.25 | 15415.88 | 通过 | 46.82 | 942.85 |
| `cp2_triton_tp8` | 2.0 | 2.160 | 1491.97 | 2907.52 | 8849.08 | 通过 | 57.08 | 477.74 |
| `cp2_triton_tp8` | 3.0 | 2.577 | 3853.11 | 6771.76 | 10556.30 | 失败 | 57.08 | 477.74 |
| `cp4_triton_tp8` | 2.0 | 2.029 | 3068.63 | 5094.37 | 8311.98 | 失败 | 52.12 | 540.61 |
| `cp8_triton_tp8` | 2.0 | 1.489 | 8758.14 | 15850.45 | 6099.86 | 失败 | 50.41 | 553.55 |

结论：

- `tp8` 仍是当前可用切分里 prefill 吞吐最高的方案，SLA 内实际 RPS `3.763 req/s`。
- 将 CP 从 FA3 改成 triton 后，`cp2/cp4/cp8` 都可以启动并完成请求，说明之前的 CP 启动失败主要来自 FA3 page attention 的 GQA head 约束。
- 注意：base TP8 配置保留 `SGLANG_KV_LAYOUT_DCU_FA=1` 和 `--attention-backend fa3`；CP smoke 配置会通过 `env_overrides` 将 `SGLANG_KV_LAYOUT_DCU_FA=0`，并替换为 `--attention-backend triton`。
- 若 CP/DP/EP smoke 启动时报 `AssertionError: capture_bs=[0]`，这是 cuda graph capture batch size 解析为 0 导致的启动期断言；`layout_repair_smoke` 配置已改为禁用 cuda graph/piecewise graph，先验证切分稳定性，正式性能测试再单独打开 graph 调优。
- CP 增大后 prefill 性能下降明显；`cp2` 在低负载下能通过 mean TTFT SLA，但吞吐只有 TP8 的约 `57%`，不是最优 prefill 配置。
- CP 的 decode TPOT 可以满足 75ms，但输出吞吐也低于 TP8。

### 启动或运行失败的切分

| 切分方式 | 结果 | 关键原因 |
|---|---|---|
| `pp8_no_spec` | 启动失败 | PP 切分加载 Qwen3.5 FP8 MoE 权重时报 `KeyError: model.layers.22.mlp.experts.w13_weight` |
| `tp4pp2_no_spec` | 启动失败 | 同 PP 权重键问题，去掉 EAGLE 后仍复现 |
| `dp2_tp4_attention` | 启动失败 | FA3 page attention 断言 `num_heads<=num_kv_heads*48` |
| `cp2dp2_tp4_triton` | 运行期失败 | 请求进入后 `dp_attention.py` all_gather 报 `output tensor size must be equal to world_size times input tensor size` |
| `ep8_auto` | 启动失败 | 当前 ROCm 路径提示没有可用 MoE 实现，要求启用 AITER MoE 并关闭 `SGLANG_USE_FP8_W8A8_MOE` |
| `ep8_triton_moe` | 启动失败 | 同上，指定 `--moe-runner-backend triton` 仍无可用 MoE 实现 |
| `cp2ep4_triton_tp8` | 启动失败 | 同 EP MoE 后端问题 |
| `ep8_aiter` | 启动失败 | AITER 路线在该 MoE shape 下报 `[aiter_moe_fp8_w8a8] no suitable backend found` |
| `deepep_ep8_triton` | 启动失败 | DeepEP 启动阶段 scheduler 异常退出 |

说明：

- 模型 config 显示 `num_attention_heads=32`、`num_key_value_heads=2`，GQA 比例很高；FA3/page attention 对 `qheads*mtp/kvheads` 有约束，因此 CP/DP attention 组合容易触发断言。
- PP 失败与 speculative 无关；`pp8_no_spec`、`tp4pp2_no_spec` 均复现同一权重键错误，更像当前 SGLang 版本对该 FP8 MoE checkpoint 的 PP 权重加载映射不兼容。
- EP/DeepEP 失败集中在 ROCm MoE kernel 支持上；默认/ triton 路线提示无可用实现，AITER 路线又无法覆盖该专家矩阵 shape。

## 当前结论

1. 在 8 卡 TP8 IFB、关闭 radix-cache 条件下，`4096 -> 1` prefill 的 SLA 内最优点为实际 `3.763 req/s`，总吞吐 `15415.88 token/s`。
2. `65536 -> 1` 在 TP8 IFB 下不满足 `mean TTFT < 3s`；已尝试 PP/CP/EP/DP/CPDP/CPEP/DeepEP，其中只有 CP 能完成请求，但 4096 prefill 吞吐已低于 TP8，尚未找到能解决 65536 TTFT 的更优切分。
3. Decode `1 -> 1024` 的最优点约为 `mc=192`，mean TPOT `46.82ms`，P99 TPOT `65.98ms`，output throughput `942.85 token/s`。
4. 对 `4096 -> 1024`，若按同等 TP8 实例做 PD 分离，估算需要约 `1P:4D` 才能让 decode 不成为瓶颈；两节点 `1P1D` 时应将入口 RPS 控制在约 `0.9 req/s` 级别。
5. 当前环境下推荐的 IFB baseline 是 `tp8`；CP2 可作为可启动对照，但不作为最优方案。PP、EP、DeepEP 需要先修复框架/内核兼容性，再进入性能比较。
