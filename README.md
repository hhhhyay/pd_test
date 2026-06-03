# vLLM / SGLang 推理性能测试工程

本工程用于在单机多 GPU 或远端多 GPU 节点上，自动部署并测试 vLLM / SGLang 的普通模式、PD 分离模式，以及 SGLang IFB 非 PD 模式下的推理性能。

当前工程包含两条测试链路：

- 通用 vLLM / SGLang normal / PD 测试：使用 `configs/cases.yaml`、`deploy/*`、`benchmark/run_bench.py`。
- SGLang IFB / 非 PD 矩阵测试：使用 `configs/ifb_matrix.yaml`、`benchmark/run_ifb_matrix.py`，已适配 Qwen3.5-397B-A17B-Channel-FP8 在 NMZ 环境中的 radix-cache 命中率测试。

## 目录结构

```text
configs/cases.yaml                  # 通用 normal / PD case 配置
configs/ifb_matrix.yaml             # SGLang IFB 矩阵配置
configs/sglang_ifb_tp8_formal.yaml  # Qwen3.5 397B IFB prefill/decode TP8 正式测试配置
configs/sglang_ifb_layout_repair_smoke.yaml # TP/PP/CP/DP/EP/DeepEP 切分 smoke 配置
deploy/vllm_pd.sh                   # vLLM normal / PD 部署入口
deploy/sglang_pd.sh                 # SGLang normal / PD 部署入口
deploy/cleanup.sh                   # 清理 case 进程
benchmark/run_bench.py              # 通用部署、健康检查、压测入口
benchmark/run_ifb_matrix.py         # SGLang IFB 远端矩阵压测入口
benchmark/run_sglang_ifb_sweep.py   # SGLang IFB prefill/decode 吞吐与切分扫描入口
benchmark/parse_logs.py             # 解析框架日志并生成报告
docs/ifb_sglang_matrix.md           # IFB 矩阵测试说明
docs/sglang_parallel_config_guide.md # SGLang 并行配置指南
docs/qwen35_397b_sglang_ifb_prefill_decode_report.md # Qwen3.5 397B IFB prefill/decode 测试报告
reports/                            # 本地报告输出
logs/                               # 本地运行日志
```

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

运行环境需要提前安装可用的 vLLM 或 SGLang，并保证模型路径、GPU 驱动、NCCL/RCCL/DTK 等运行时可用。

## 通用 normal / PD 测试

编辑 `configs/cases.yaml`，启用需要测试的 case：

```yaml
- name: sglang_pd_1p1d_qwen
  enabled: true
  framework: sglang
  mode: pd
  pd_layout: 1P1D
  model_path: /data/models/Qwen3-32B
  tp_size: 1
  pp_size: 1
  dp_size: 1
  ep_size: 1
  atten_cp_size: 1
  quantization: null
  max_num_batched_tokens: 16384
  max_num_seqs: 512
  input_len: 2048
  output_len: 256
  concurrency: 64
  num_prompts: 1000
```

字段说明：

- `framework`: `vllm` 或 `sglang`
- `mode`: `normal` 或 `pd`
- `pd_layout`: `1P1D`、`1P2D`、`2P2D`、`2P4D`
- `tp_size` / `pp_size` / `dp_size` / `ep_size` / `atten_cp_size`: 并行配置
- `quantization`: 量化方式，不使用时填 `null`
- `max_num_batched_tokens` / `max_num_seqs`: 服务端批处理限制
- `input_len` / `output_len` / `concurrency` / `num_prompts`: 压测负载参数

运行全量测试：

```bash
python benchmark/run_bench.py
python benchmark/parse_logs.py
```

运行单个 case：

```bash
python benchmark/run_bench.py --case vllm_normal_1gpu --include-disabled
python benchmark/parse_logs.py
```

如果服务已经手动启动，可以跳过部署：

```bash
python benchmark/run_bench.py --case vllm_normal_1gpu --include-disabled --skip-deploy
```

## PD 启动命令适配

不同 vLLM / SGLang 版本、厂商分支和 PD 实现的启动参数差异很大，因此部署脚本支持通过环境变量传入命令模板：

```bash
export VLLM_PREFILL_CMD_TEMPLATE="python -m your.vllm.prefill_server --your-pd-args"
export VLLM_DECODE_CMD_TEMPLATE="python -m your.vllm.decode_server --your-pd-args"
export VLLM_ROUTER_CMD_TEMPLATE="python -m your.vllm.router --your-router-args"

export SGLANG_PREFILL_CMD_TEMPLATE="python -m your.sglang.prefill_server --your-pd-args"
export SGLANG_DECODE_CMD_TEMPLATE="python -m your.sglang.decode_server --your-pd-args"
export SGLANG_ROUTER_CMD_TEMPLATE="python -m your.sglang.router --your-router-args"
```

脚本会自动补充模型路径、端口、TP、batch token、最大并发序列数等通用参数。真实 PD 测试时，请替换为当前环境中可用的 prefill / decode / router 启动命令。

## SGLang IFB 矩阵测试

IFB / 非 PD 远端矩阵测试入口：

```text
configs/ifb_matrix.yaml
benchmark/run_ifb_matrix.py
```

当前模板覆盖：

- 模型：`/model/qwen3.5/Qwen3.5-397B-A17B-Channel-FP8`
- 模式：SGLang IFB / 非 PD
- 接口：OpenAI-compatible `/v1/completions`
- 输入输出组合：`(2048, 1024)`、`(65536, 1024)`、`(32768, 128)`
- radix-cache 目标命中率：`50%`、`90%`、`99%`
- 并发扫描：`1-32` 步长 1，`32-64` 步长 2，`64+` 步长 4
- SLA：`mean TTFT < 10s`，`mean TPOT < 75ms`

常用命令：

```bash
# 查看将要运行多少个点
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --dry-run

# 只跑第一个点，验证链路
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --smoke --reuse-server

# 跑通 50% / 90% / 99% 三个命中率的小流量流程
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --smoke --smoke-all-hit-rates --reuse-server --no-resume

# 调试服务端 cache 命中展示，关闭 smoke warmup 并增加 prompt 数
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --smoke --smoke-all-hit-rates --smoke-prompts 4 --smoke-warmup-requests 0 --reuse-server --no-resume

# 通过网关节点进入目标节点
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --ssh-target guobj@10.16.1.9 --nested-ssh-target nmz22 --docker-container gbj_sgl0522 --client-host 10.16.1.22 --smoke --smoke-all-hit-rates --no-resume

# 全量矩阵
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml
```

详细说明见：

- `docs/ifb_sglang_matrix.md`
- `docs/sglang_parallel_config_guide.md`

## SGLang 并行配置指南

SGLang 的 TP / PP / DP / EP / DeepEP / DP attention / MoE-DP / Attention CP / PD 分离配置方式，已整理在：

```text
docs/sglang_parallel_config_guide.md
```

该文档包含：

- 各并行方式的参数、适用场景、优势和代价
- Dense / MoE / MLA / 长上下文 / PD 场景下的推荐配置流程
- 常见启动命令模板
- 如何映射到本工程的 `tp_size`、`pp_size`、`dp_size`、`ep_size`、`atten_cp_size` 和 `extra_args`
- 全量测试前的检查清单

## GLM5-INT8 IFB / PD 测试

GLM5-INT8 测试按 prefill 和 decode 两个口径拆分：

```bash
# prefill：关闭 radix-cache，output_len=1，max_concurrency=256，通过 request-rate 找 mean TTFT < 30s 的实际 Request throughput
python benchmark/run_sglang_ifb_sweep.py --config configs/glm5_int8_ifb_prefill_sweep.yaml --layouts tp8,tp8cp8,tp8ep8,deepep_tp8ep8 --no-resume

# decode：开启 radix-cache，generated-shared-prefix 构造 99% 命中率，request-rate=inf，通过 max_concurrency 扫描 mean TPOT < 50ms 的容量
python benchmark/run_sglang_ifb_sweep.py --config configs/glm5_int8_ifb_decode_cache99_sweep.yaml --layouts tp8,tp8cp8,tp8ep8,deepep_tp8ep8 --no-resume
```

完整 layout 覆盖 `tp8`、`pp8`、`tp4pp2`、`tp8cp8`、`cp4_tp8`、`dp2_tp4_moe`、`tp8ep8`、`deepep_tp8ep8`。连接到 `10.16.1.20` 后，在 `/mnt11/task/sgl/` 下运行；若目标环境为裸机执行，配置里的 `docker_container: null` 会让 runner 直接通过 SSH 执行命令。

PD 分离启动脚本：

```bash
# P 节点
bash deploy/sglang_glm5_pd.sh prefill <master_ip> 0 tp8cp8

# D 节点
bash deploy/sglang_glm5_pd.sh decode <master_ip> 0 deepep

# Router
DECODE_URL=http://<decode_ip>:30000 bash deploy/sglang_glm5_pd.sh router <prefill_ip>

# 清理
bash deploy/sglang_glm5_pd.sh cleanup <master_ip>
```

结果目录：

```text
reports/glm5_int8_ifb_prefill_sweep
reports/glm5_int8_ifb_decode_cache99_sweep
/mnt11/task/sgl/pd_test_runs/glm5_int8_ifb_prefill_sweep
/mnt11/task/sgl/pd_test_runs/glm5_int8_ifb_decode_cache99_sweep
```

## 指标说明

压测程序记录的核心指标：

- `TTFT`: 请求发出到首 token 返回的时间
- `TPOT`: 首 token 后平均每个输出 token 的生成时间
- `ITL`: 相邻 streamed chunk 的间隔
- `E2E latency`: 请求端到端耗时
- `request throughput`: 请求吞吐，单位 req/s
- `output token throughput`: 输出 token 吞吐
- `total token throughput`: 输入加输出 token 总吞吐
- `success_rate`: 成功率

SGLang IFB 测试还会记录：

- `gsp_shared_len`: 生成 shared prefix 的目标长度
- `gsp_question_len`: 非共享问题部分目标长度
- `server_cached_tokens`: 从服务端日志解析出的 cached token 数
- `server_new_tokens`: 从服务端日志解析出的 new token 数
- `server_observed_cache_ratio`: 根据服务端 `#cached-token / (#cached-token + #new-token)` 估算的实际命中比例

注意：SGLang 服务端日志中的 `#new-token + #cached-token` 不一定严格等于配置里的 `input_len`。原因包括 tokenizer 实际长度偏差、`--page-size` 分页对齐、warmup/request 顺序以及 radix-cache 的实际命中状态。

## 报告输出

通用测试输出：

```text
reports/bench_metrics.csv
reports/result.csv
reports/result.html
```

IFB 矩阵测试输出：

```text
reports/ifb_sglang_qwen3_397_fp8_nmz/result.csv
reports/ifb_sglang_qwen3_397_fp8_nmz/result.html
```

远端原始日志和 benchmark JSONL 默认输出到：

```text
/mnt11/task/sgl/pd_test_runs/ifb_sglang_qwen3_397_fp8_nmz
```

## 日志解析

`benchmark/parse_logs.py` 会尽量从框架日志中解析：

- prefill 阶段耗时
- decode 阶段耗时
- KV transfer 耗时
- GPU 显存
- OOM
- 错误请求

不同框架和版本日志格式差异较大，解析逻辑采用正则兜底策略。如果本地日志有固定字段，可以在 `benchmark/parse_logs.py` 的 `PATTERNS` 中补充更精确的表达式。

## 清理服务

停止单个 case：

```bash
CASE_NAME=vllm_pd_1p1d bash deploy/cleanup.sh
```

停止所有由本工程记录 PID 的服务：

```bash
bash deploy/cleanup.sh
```

远端 IFB 测试如果使用 `run_ifb_matrix.py` 且没有传 `--reuse-server`，脚本会在结束时自动清理 SGLang server。
