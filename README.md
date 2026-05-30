# LLM Inference Benchmark for vLLM/SGLang PD Disaggregation

这个工程用于在单机多 GPU 环境中自动部署并测试 vLLM/SGLang 的普通模式和 PD 分离模式性能。它默认通过 OpenAI-compatible `/v1/chat/completions` 流式接口压测，并输出请求级指标与日志解析报告。

## 目录结构

```text
configs/cases.yaml        # case 配置
deploy/vllm_pd.sh         # vLLM normal/PD 部署入口
deploy/sglang_pd.sh       # SGLang normal/PD 部署入口
deploy/cleanup.sh         # 停止 case 进程
benchmark/run_bench.py    # 自动部署、健康检查、压测
benchmark/parse_logs.py   # 解析框架日志并生成报告
reports/                  # 压测结果输出目录
logs/{case_name}/         # 每个 case 的部署日志、服务日志和 PID
```

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

同时需要提前安装可用的 vLLM 或 SGLang，并确保 `bash`、`python`、CUDA 驱动和模型文件都在运行环境可访问。

## 添加新模型

编辑 `configs/cases.yaml`，把 `model_path` 改成模型本地路径或框架可识别的模型名：

```yaml
model_path: /data/models/Qwen3-32B
```

如果需要量化，设置：

```yaml
quantization: awq
```

不使用量化时保持 `null`。

## 添加新 case

复制一个已有 case，修改 `name`、`framework`、`mode` 和资源参数：

```yaml
- name: vllm_pd_2p4d_qwen32b
  enabled: true
  framework: vllm
  mode: pd
  pd_layout: 2P4D
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

`pd_layout` 支持 `1P1D`、`1P2D`、`2P2D`、`2P4D`，含义是 prefill server 数量和 decode server 数量。普通模式设置：

```yaml
mode: normal
pd_layout: null
```

## PD 启动命令适配

vLLM 和 SGLang 的 PD 分离启动参数在不同版本、分支和厂商构建中差异较大，所以部署脚本提供了命令模板环境变量：

```bash
export VLLM_PREFILL_CMD_TEMPLATE="python -m your.vllm.prefill_server --your-pd-args"
export VLLM_DECODE_CMD_TEMPLATE="python -m your.vllm.decode_server --your-pd-args"
export VLLM_ROUTER_CMD_TEMPLATE="python -m your.vllm.router --your-router-args"

export SGLANG_PREFILL_CMD_TEMPLATE="python -m your.sglang.prefill_server --your-pd-args"
export SGLANG_DECODE_CMD_TEMPLATE="python -m your.sglang.decode_server --your-pd-args"
export SGLANG_ROUTER_CMD_TEMPLATE="python -m your.sglang.router --your-router-args"
```

脚本会自动追加通用参数，包括模型路径、端口、TP、最大 batch token 和最大并发序列数。默认模板使用普通 OpenAI API server 占位，便于先验证压测链路；真实 PD 测试时请替换成你所用版本的 PD 命令。

PD router/proxy 默认设置 `CUDA_VISIBLE_DEVICES=`，即不额外占用 GPU；如果你的 router 也需要 GPU，可以设置 `ROUTER_CUDA_VISIBLE_DEVICES=0` 或其他设备列表。

## 运行全量测试

先把要跑的 case 设置为 `enabled: true`，然后执行：

```bash
python benchmark/run_bench.py
python benchmark/parse_logs.py
```

## 运行 SGLang IFB 矩阵测试

非 PD/IFB 的远端矩阵测试入口见 `configs/ifb_matrix.yaml` 和 `benchmark/run_ifb_matrix.py`。当前模板覆盖 Qwen3.5-397B-A17B-Channel-FP8 在 NMZ 环境下的 radix-cache 命中率矩阵。

```bash
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --dry-run
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --smoke --reuse-server
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --smoke --smoke-all-hit-rates --reuse-server --no-resume
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --smoke --smoke-all-hit-rates --smoke-prompts 4 --smoke-warmup-requests 0 --reuse-server --no-resume
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml --ssh-target guobj@10.16.1.9 --nested-ssh-target nmz22 --docker-container <container-name> --smoke --smoke-all-hit-rates --reuse-server --no-resume
python benchmark/run_ifb_matrix.py --config configs/ifb_matrix.yaml
```

详细说明见 `docs/ifb_sglang_matrix.md`。

压测程序会对每个 case 至少 warmup 30 条请求，再正式测试。主要输出：

```text
reports/bench_metrics.csv # 压测原始汇总
reports/result.csv        # 合并日志解析后的结果
reports/result.html       # HTML 表格报告
```

## 运行单个 case

即使 case 是 `enabled: false`，也可以显式指定运行：

```bash
python benchmark/run_bench.py --case vllm_normal_1gpu --include-disabled
python benchmark/parse_logs.py
```

如果服务已经由你手动启动，可以跳过部署：

```bash
python benchmark/run_bench.py --case vllm_normal_1gpu --include-disabled --skip-deploy
```

此时脚本会直接访问 `host:base_port`，默认是 `127.0.0.1:18000`。

## 指标说明

`benchmark/run_bench.py` 记录：

- `TTFT`: 请求发出到首个 streamed token 的时间
- `TPOT`: 首 token 后平均每个输出 token 时间
- `ITL`: 相邻 streamed chunk 的平均间隔
- `E2E latency`: 请求端到端耗时
- `request throughput`: 成功请求吞吐
- `output token throughput`: 输出 token 估算吞吐
- `total token throughput`: 输入加输出 token 估算吞吐
- `success_rate`: 成功率

`benchmark/parse_logs.py` 会尽量从框架日志中解析 prefill、decode、KV transfer、GPU 显存、OOM 和错误请求。由于不同版本日志格式差异很大，解析采用正则兜底策略；如果你的日志里有固定字段，可以在 `PATTERNS` 中补充更精确的表达式。

## 清理服务

停止单个 case：

```bash
CASE_NAME=vllm_pd_1p1d bash deploy/cleanup.sh
```

停止所有由本工程记录 PID 的服务：

```bash
bash deploy/cleanup.sh
```
