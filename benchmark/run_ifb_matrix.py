#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import logging
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to load config {path}: {exc}") from exc


def q(value: Any) -> str:
    return shlex.quote(str(value))


def join_shell_args(args: list[Any]) -> str:
    rendered = []
    for arg in args:
        if arg == "__AUTO_HOST_IP__":
            rendered.append("$(hostname -I | awk '{print $1}')")
        else:
            rendered.append(q(arg))
    return " ".join(rendered)


def run_local(cmd: list[str], timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    logging.debug("Running local command: %s", " ".join(shlex.quote(x) for x in cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    if result.stdout:
        logging.debug("stdout: %s", result.stdout[-4000:])
    if result.stderr:
        logging.debug("stderr: %s", result.stderr[-4000:])
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr[-4000:]}")
    return result


def ssh(remote: str, command: str, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_local(["ssh", "-o", "BatchMode=yes", remote, command], timeout=timeout, check=check)


def docker_exec(remote: str, container: str, command: str, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return ssh(remote, f"docker exec {q(container)} bash -lc {q(command)}", timeout=timeout, check=check)


def concurrency_values(scan: dict[str, Any]) -> list[int]:
    start = int(scan.get("start", 1))
    end = int(scan["end"])
    rules = scan.get("rules", [])
    values: list[int] = []
    current = start
    while current <= end:
        values.append(current)
        step = None
        for rule in rules:
            if current < int(rule["until"]):
                step = int(rule["step"])
                break
        if step is None:
            step = int(rules[-1]["step"]) if rules else 1
        current += step
    return values


def gsp_lengths(input_len: int, hit_rate: float) -> tuple[int, int]:
    if hit_rate >= 1.0:
        shared = max(input_len - 1, 1)
        question = 1
    else:
        shared = max(int(round(input_len * hit_rate)), 1)
        question = max(input_len - shared, 1)
    return shared, question


def ensure_remote_dirs(cfg: dict[str, Any]) -> None:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    remote_dir = cfg["reports"]["remote_dir"]
    docker_exec(remote, container, f"mkdir -p {q(remote_dir)}/logs {q(remote_dir)}/bench")


def cleanup_server(cfg: dict[str, Any]) -> None:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    remote_dir = cfg["reports"]["remote_dir"]
    cmd = f"""
set +e
if [ -f {q(remote_dir)}/server.pid ]; then
  pid="$(cat {q(remote_dir)}/server.pid)"
  kill "$pid" 2>/dev/null || true
  sleep 3
  kill -9 "$pid" 2>/dev/null || true
  rm -f {q(remote_dir)}/server.pid
fi
pkill -f 'sglang.*{cfg["server"]["model_path"]}' 2>/dev/null || true
"""
    docker_exec(remote, container, cmd, check=False)


def server_command(cfg: dict[str, Any]) -> str:
    remote_cfg = cfg["remote"]
    server = cfg["server"]
    host = remote_cfg.get("host", "0.0.0.0")
    if host == "auto_ip":
        host = "__AUTO_HOST_IP__"
    args = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        server["model_path"],
        "--served-model-name",
        server.get("served_model_name", server.get("model_name", "model")),
        "--host",
        host,
        "--port",
        str(remote_cfg["port"]),
        "--tp-size",
        str(server.get("tp_size", 1)),
        "--pp-size",
        str(server.get("pp_size", 1)),
        "--mem-fraction-static",
        str(server.get("mem_fraction_static", 0.90)),
    ]
    optional_parallel = [
        ("atten_cp_size", "--attention-context-parallel-size"),
        ("ep_size", "--moe-data-parallel-size"),
        ("dp_size", "--data-parallel-size"),
    ]
    for key, flag in optional_parallel:
        value = server.get(key)
        if value not in (None, "", "null"):
            args.extend([flag, str(value)])
    max_running_requests = server.get("max_running_requests")
    if max_running_requests not in (None, "", "null"):
        args.extend(["--max-running-requests", str(max_running_requests)])
    max_total_tokens = server.get("max_total_tokens")
    if max_total_tokens not in (None, "", "null"):
        args.extend(["--max-total-tokens", str(max_total_tokens)])
    quant = server.get("quantization")
    if quant:
        args.extend(["--quantization", str(quant)])
    args.extend(str(x) for x in server.get("extra_args", []))
    if server.get("enable_radix_cache", True):
        args.extend(str(x) for x in server.get("radix_cache_extra_args", []))
    else:
        args.append("--disable-radix-cache")
    return join_shell_args(args)


def server_env_prefix(cfg: dict[str, Any]) -> str:
    env = cfg["server"].get("env", {})
    if not env:
        return ""
    return " ".join(f"export {key}={q(value)};" for key, value in env.items())


def start_server(cfg: dict[str, Any]) -> None:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    workdir = cfg["remote"]["workdir"]
    remote_dir = cfg["reports"]["remote_dir"]
    cmd = f"""
set -euo pipefail
cd {q(workdir)}
{server_env_prefix(cfg)}
nohup {server_command(cfg)} > {q(remote_dir)}/logs/server.log 2>&1 &
echo "$!" > {q(remote_dir)}/server.pid
"""
    docker_exec(remote, container, cmd)
    logging.info("Started SGLang server, logs at %s/logs/server.log", remote_dir)


def wait_ready(cfg: dict[str, Any]) -> None:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    host = cfg["remote"].get("client_host", "127.0.0.1")
    port = int(cfg["remote"]["port"])
    timeout_s = int(cfg["benchmark"].get("ready_timeout_s", 1800))
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        result = docker_exec(
            remote,
            container,
            f"curl -fsS http://{host}:{port}/health || curl -fsS http://{host}:{port}/v1/models",
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            logging.info("SGLang server is ready")
            return
        last = (result.stderr or result.stdout)[-1000:]
        time.sleep(10)
    raise TimeoutError(f"SGLang server did not become ready in {timeout_s}s. Last output: {last}")


def bench_command(cfg: dict[str, Any], input_len: int, output_len: int, hit_rate: float, concurrency: int, output_file: str) -> str:
    remote_cfg = cfg["remote"]
    bench = cfg["benchmark"]
    server = cfg["server"]
    shared_len, question_len = gsp_lengths(input_len, hit_rate)
    prompts = int(bench.get("num_prompts", 256))
    group_size = max(prompts, concurrency, 1)
    backend = bench.get("backend", "sglang-oai-chat")
    served_model_name = server.get("served_model_name", server.get("model_name", server["model_path"]))
    model_arg = server["model_path"] if backend == "sglang" else served_model_name
    args = [
        "python",
        "-m",
        "sglang.bench_serving",
        "--backend",
        backend,
        "--host",
        remote_cfg.get("client_host", "127.0.0.1"),
        "--port",
        str(remote_cfg["port"]),
        "--model",
        model_arg,
        "--served-model-name",
        served_model_name,
        "--tokenizer",
        server.get("tokenizer_path", server["model_path"]),
        "--dataset-name",
        bench.get("dataset_name", "generated-shared-prefix"),
        "--num-prompts",
        str(prompts),
        "--request-rate",
        str(bench.get("request_rate", "inf")),
        "--max-concurrency",
        str(concurrency),
        "--warmup-requests",
        str(int(bench.get("warmup_requests", 30))),
        "--gsp-num-groups",
        "1",
        "--gsp-prompts-per-group",
        str(group_size),
        "--gsp-system-prompt-len",
        str(shared_len),
        "--gsp-question-len",
        str(question_len),
        "--gsp-output-len",
        str(output_len),
        "--gsp-range-ratio",
        str(bench.get("random_range_ratio", 1.0)),
        "--output-file",
        output_file,
        "--output-details",
        "--disable-tqdm",
    ]
    if bench.get("gsp_fast_prepare", False):
        args.append("--gsp-fast-prepare")
    if bench.get("tokenize_prompt", False):
        args.append("--tokenize-prompt")
    return " ".join(q(x) for x in args)


def parse_jsonish(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    return {}


TEXT_METRICS = {
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s*([0-9.]+)",
    "p90_ttft_ms": r"P90 TTFT \(ms\):\s*([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s*([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s*([0-9.]+)",
    "p90_tpot_ms": r"P90 TPOT \(ms\):\s*([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s*([0-9.]+)",
    "mean_e2e_latency_ms": r"Mean E2E Latency \(ms\):\s*([0-9.]+)",
    "p90_e2e_latency_ms": r"P90 E2E Latency \(ms\):\s*([0-9.]+)",
    "p99_e2e_latency_ms": r"P99 E2E Latency \(ms\):\s*([0-9.]+)",
    "request_throughput": r"Request throughput \(req/s\):\s*([0-9.]+)",
    "output_throughput": r"Output token throughput \(tok/s\):\s*([0-9.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s*([0-9.]+)",
    "successful_requests": r"Successful requests:\s*([0-9.]+)",
}


def parse_text_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, pattern in TEXT_METRICS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def normalize_metric(data: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return ""


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def merge_detail_metrics(data: dict[str, Any]) -> dict[str, Any]:
    ttfts = data.get("ttfts")
    if isinstance(ttfts, list):
        p90 = percentile(ttfts, 0.90)
        if p90 is not None and "p90_ttft_ms" not in data:
            data["p90_ttft_ms"] = p90 * 1000 if p90 < 100 else p90
    return data


def capture_server_cache_log(cfg: dict[str, Any], tag: str) -> str:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    remote_dir = cfg["reports"]["remote_dir"]
    excerpt_file = f"{remote_dir}/logs/{tag}.server_cache.log"
    cmd = (
        "grep -Ei 'Prefill batch|cached-token|new-token|radix|prefix|cache hit|hit rate' "
        f"{q(remote_dir)}/logs/server.log | tail -400 > {q(excerpt_file)} || true"
    )
    docker_exec(remote, container, cmd, timeout=60, check=False)
    return excerpt_file


def summarize_server_cache_excerpt(cfg: dict[str, Any], excerpt_file: str) -> dict[str, Any]:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    py = (
        "import json,os,re;"
        f"p={excerpt_file!r};"
        "text=open(p, encoding='utf-8', errors='ignore').read() if os.path.exists(p) else '';"
        "pairs=[(int(a), int(b)) for a,b in re.findall(r'#new-token:\\s*(\\d+).*?#cached-token:\\s*(\\d+)', text)];"
        "new=sum(a for a,b in pairs); cached=sum(b for a,b in pairs);"
        "ratio=(cached/(new+cached) if (new+cached) else None);"
        "print(json.dumps({'server_prefill_batches':len(pairs),'server_new_tokens':new,'server_cached_tokens':cached,'server_observed_cache_ratio':ratio}))"
    )
    result = docker_exec(remote, container, f"python -c {q(py)}", timeout=60, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {}


def read_remote_json_summary(cfg: dict[str, Any], output_file: str) -> dict[str, Any]:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    keys = [
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "output_throughput",
        "total_throughput",
        "mean_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "ttfts",
    ]
    py = (
        "import json;"
        f"p={output_file!r};"
        "obj=json.loads(open(p, encoding='utf-8').readline());"
        f"keys={keys!r};"
        "print(json.dumps({k: obj.get(k) for k in keys if k in obj}, ensure_ascii=False))"
    )
    result = docker_exec(remote, container, f"python -c {q(py)}", timeout=60, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        return merge_detail_metrics(json.loads(result.stdout.strip().splitlines()[-1]))
    except json.JSONDecodeError:
        return {}


def run_one_bench(cfg: dict[str, Any], input_len: int, output_len: int, hit_rate: float, concurrency: int) -> dict[str, Any]:
    remote = cfg["remote"]["ssh_target"]
    container = cfg["remote"]["docker_container"]
    remote_dir = cfg["reports"]["remote_dir"]
    tag = f"in{input_len}_out{output_len}_hit{int(hit_rate * 100)}_bs{concurrency}"
    output_file = f"{remote_dir}/bench/{tag}.jsonl"
    log_file = f"{remote_dir}/logs/{tag}.log"
    shared_len, question_len = gsp_lengths(input_len, hit_rate)
    prompts = int(cfg["benchmark"].get("num_prompts", 256))
    group_size = max(prompts, concurrency, 1)
    clear_cache = ""
    if cfg["benchmark"].get("clear_gsp_cache", False):
        clear_cache = (
            "rm -f "
            f"/root/.cache/sglang/benchmark/gen_shared_prefix_*_1_{group_size}_{shared_len}_{question_len}_{output_len}_*.pkl\n"
        )
    cmd = f"""
set -euo pipefail
{clear_cache}
{bench_command(cfg, input_len, output_len, hit_rate, concurrency, output_file)} > {q(log_file)} 2>&1
tail -200 {q(log_file)}
"""
    logging.info("Benchmark %s", tag)
    result = docker_exec(remote, container, cmd, timeout=None, check=False)
    text = result.stdout + "\n" + result.stderr
    data = parse_jsonish(text)
    data.update({k: v for k, v in parse_text_metrics(text).items() if k not in data})
    detail_data = read_remote_json_summary(cfg, output_file)
    data.update({k: v for k, v in detail_data.items() if v not in (None, "")})
    server_cache_log = capture_server_cache_log(cfg, tag)
    cache_data = summarize_server_cache_excerpt(cfg, server_cache_log)
    data.update({k: v for k, v in cache_data.items() if v not in (None, "")})
    row = {
        "tag": tag,
        "input_len": input_len,
        "output_len": output_len,
        "radix_cache_hit_rate_target": hit_rate,
        "gsp_shared_len": shared_len,
        "gsp_question_len": question_len,
        "concurrency": concurrency,
        "returncode": result.returncode,
        "remote_output_file": output_file,
        "remote_log_file": log_file,
        "remote_server_cache_log": server_cache_log,
        "server_observed_cache_ratio": normalize_metric(data, ["server_observed_cache_ratio"]),
        "server_cached_tokens": normalize_metric(data, ["server_cached_tokens"]),
        "server_new_tokens": normalize_metric(data, ["server_new_tokens"]),
        "mean_ttft_ms": normalize_metric(data, ["mean_ttft_ms", "mean_ttft", "avg_ttft_ms"]),
        "p90_ttft_ms": normalize_metric(data, ["p90_ttft_ms", "p90_ttft"]),
        "p99_ttft_ms": normalize_metric(data, ["p99_ttft_ms", "p99_ttft"]),
        "mean_tpot_ms": normalize_metric(data, ["mean_tpot_ms", "mean_tpot", "avg_tpot_ms"]),
        "p90_tpot_ms": normalize_metric(data, ["p90_tpot_ms", "p90_tpot"]),
        "p99_tpot_ms": normalize_metric(data, ["p99_tpot_ms", "p99_tpot"]),
        "request_throughput": normalize_metric(data, ["request_throughput", "request_throughput_rps"]),
        "output_throughput": normalize_metric(data, ["output_throughput", "output_token_throughput"]),
        "total_token_throughput": normalize_metric(data, ["total_token_throughput", "total_throughput"]),
        "raw_summary": json.dumps(data, ensure_ascii=False),
    }
    row["sla_pass"] = sla_pass(cfg, row)
    if result.returncode != 0:
        row["error"] = (result.stderr or result.stdout)[-2000:]
    return row


def as_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def sla_pass(cfg: dict[str, Any], row: dict[str, Any]) -> bool:
    sla = cfg["benchmark"].get("sla", {})
    mean_ttft_ms = as_float(row.get("mean_ttft_ms"))
    mean_tpot_ms = as_float(row.get("mean_tpot_ms"))
    if math.isnan(mean_ttft_ms) or math.isnan(mean_tpot_ms):
        return False
    return mean_ttft_ms < float(sla.get("mean_ttft_s_lt", 10)) * 1000 and mean_tpot_ms < float(sla.get("mean_tpot_ms_lt", 75))


def write_reports(local_dir: Path, rows: list[dict[str, Any]]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    csv_path = local_dir / "result.csv"
    html_path = local_dir / "result.html"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cells = []
    header = "".join(f"<th>{name}</th>" for name in fieldnames if name != "raw_summary")
    for row in rows:
        cells.append("<tr>" + "".join(f"<td>{row.get(name, '')}</td>" for name in fieldnames if name != "raw_summary") + "</tr>")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>IFB Matrix Results</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;font-size:12px}"
        "th,td{border:1px solid #d7dce5;padding:6px 8px;text-align:right}th{background:#eef2f8}"
        "td:first-child,th:first-child{text-align:left}</style>"
        "<h1>IFB SGLang Matrix Results</h1><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(cells)
        + "</tbody></table>",
        encoding="utf-8",
    )
    logging.info("Wrote %s and %s", csv_path, html_path)


def read_existing_rows(local_dir: Path) -> list[dict[str, Any]]:
    csv_path = local_dir / "result.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_matrix(
    cfg: dict[str, Any],
    dry_run: bool,
    smoke: bool,
    smoke_all_hit_rates: bool,
    reuse_server: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    if smoke:
        cfg = copy.deepcopy(cfg)
        cfg["benchmark"]["num_prompts"] = 2
        cfg["benchmark"]["warmup_requests"] = 1
    ensure_remote_dirs(cfg)
    local_dir = ROOT_DIR / cfg["reports"]["local_dir"]
    rows: list[dict[str, Any]] = read_existing_rows(local_dir) if resume else []
    completed = {row.get("tag") for row in rows if str(row.get("returncode", "")) == "0"}
    pairs = cfg["benchmark"]["input_output_pairs"]
    hit_rates = [float(x) for x in cfg["benchmark"]["radix_cache_hit_rates"]]
    concurrencies = concurrency_values(cfg["benchmark"]["concurrency_scan"])
    if smoke:
        pairs = pairs[:1]
        if not smoke_all_hit_rates:
            hit_rates = hit_rates[:1]
        concurrencies = concurrencies[:1]
    if dry_run:
        logging.info("Would run %d benchmark points", len(pairs) * len(hit_rates) * len(concurrencies))
        return rows
    if not reuse_server:
        cleanup_server(cfg)
        start_server(cfg)
    wait_ready(cfg)
    try:
        for pair in pairs:
            for hit_rate in hit_rates:
                for concurrency in concurrencies:
                    tag = f"in{int(pair['input_len'])}_out{int(pair['output_len'])}_hit{int(hit_rate * 100)}_bs{int(concurrency)}"
                    if tag in completed:
                        logging.info("Skipping completed %s", tag)
                        continue
                    row = run_one_bench(cfg, int(pair["input_len"]), int(pair["output_len"]), hit_rate, int(concurrency))
                    rows.append(row)
                    write_reports(local_dir, rows)
                    if cfg["benchmark"].get("stop_after_first_sla_failure", False) and not row["sla_pass"]:
                        logging.info("Stopping scan after SLA failure at %s", row["tag"])
                        break
                time.sleep(int(cfg["benchmark"].get("cooldown_s", 5)))
    finally:
        if not reuse_server:
            cleanup_server(cfg)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IFB SGLang radix-cache benchmark matrix on a remote Docker container.")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "ifb_matrix.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only the first point to validate the flow.")
    parser.add_argument("--smoke-all-hit-rates", action="store_true", help="With --smoke, run all configured radix-cache hit rates for the first input/output pair and first concurrency.")
    parser.add_argument("--reuse-server", action="store_true", help="Reuse an already running SGLang server and do not stop it.")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing local result.csv rows.")
    parser.add_argument("--ssh-target", help="Override remote.ssh_target from the config, for example nmz22.")
    parser.add_argument("--client-host", help="Override remote.client_host from the config.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    try:
        cfg = load_config(Path(args.config))
        if args.ssh_target:
            cfg["remote"]["ssh_target"] = args.ssh_target
        if args.client_host:
            cfg["remote"]["client_host"] = args.client_host
        rows = run_matrix(cfg, args.dry_run, args.smoke, args.smoke_all_hit_rates, args.reuse_server, not args.no_resume)
        if rows:
            write_reports(ROOT_DIR / cfg["reports"]["local_dir"], rows)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130
    except Exception as exc:
        logging.exception("IFB matrix failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
