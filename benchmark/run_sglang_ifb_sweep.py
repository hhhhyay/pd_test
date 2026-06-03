#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import logging
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_ifb_matrix import (
    ROOT_DIR,
    bench_command,
    cleanup_server,
    docker_exec,
    ensure_remote_dirs,
    load_config,
    parse_jsonish,
    parse_text_metrics,
    q,
    read_remote_json_summary,
    remote_target,
    setup_logging,
    start_server,
    wait_ready,
)


def slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "")


def selected(items: list[dict[str, Any]], names: set[str] | None) -> list[dict[str, Any]]:
    if not names:
        return items
    return [item for item in items if item["name"] in names]


def merge_server(base: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    server = copy.deepcopy(base)
    for key, value in layout.get("server_overrides", {}).items():
        server[key] = value
    if layout.get("env_overrides"):
        server.setdefault("env", {})
        server["env"].update(layout["env_overrides"])
    if layout.get("extra_args_remove_flags"):
        remove_flags = set(layout["extra_args_remove_flags"])
        flags_with_values = {
            "--speculative-algorithm",
            "--speculative-num-steps",
            "--speculative-eagle-topk",
            "--speculative-num-draft-tokens",
            "--num-continuous-decode-steps",
            "--cuda-graph-max-bs",
            "--cuda-graph-bs",
            "--attention-backend",
            "--prefill-attention-backend",
            "--decode-attention-backend",
            "--linear-attn-prefill-backend",
            "--page-size",
            "--chunked-prefill-size",
            "--kv-cache-dtype",
            "--moe-a2a-backend",
            "--moe-runner-backend",
            "--deepep-mode",
        }
        cleaned = []
        skip_next = False
        for item in server.get("extra_args", []):
            if skip_next:
                skip_next = False
                continue
            if item in remove_flags:
                skip_next = item in flags_with_values
                continue
            cleaned.append(item)
        server["extra_args"] = cleaned
    if layout.get("extra_args_append"):
        server["extra_args"] = list(server.get("extra_args", [])) + list(layout["extra_args_append"])
    return server


def request_rates(workload: dict[str, Any], override: list[float] | None) -> list[float]:
    if override:
        return override
    return [float(x) for x in workload.get("request_rates", [math.inf])]


def rate_label(rate: float) -> str:
    return "inf" if math.isinf(rate) else str(rate)


def max_concurrency_values(workload: dict[str, Any], override: list[int] | None) -> list[int]:
    if override:
        return override
    values = workload.get("max_concurrency_values")
    if values:
        return [int(x) for x in values]
    return [int(workload.get("max_concurrency", 512))]


def summarize_server_excerpt(cfg: dict[str, Any], excerpt_file: str) -> dict[str, Any]:
    remote = remote_target(cfg)
    container = cfg["remote"]["docker_container"]
    py = (
        "import json,os,re,statistics;"
        f"p={excerpt_file!r};"
        "text=open(p, encoding='utf-8', errors='ignore').read() if os.path.exists(p) else '';"
        "pref=[(int(a),int(b),float(c)) for a,b,c in re.findall(r'#new-token:\\s*(\\d+).*?#cached-token:\\s*(\\d+).*?input throughput \\(token/s\\):\\s*([0-9.]+)', text)];"
        "dec=[float(x) for x in re.findall(r'gen throughput \\(token/s\\):\\s*([0-9.]+)', text)];"
        "rq=[float(x) for x in re.findall(r'run[-_ ]?que(?:ue)?\\s*[:=]\\s*([0-9.]+)', text, flags=re.I)];"
        "new=sum(x[0] for x in pref); cached=sum(x[1] for x in pref);"
        "pin=[x[2] for x in pref if x[2] > 0];"
        "out={'server_prefill_batches':len(pref),'server_new_tokens':new,'server_cached_tokens':cached,"
        "'server_observed_cache_ratio':(cached/(new+cached) if (new+cached) else None),"
        "'server_prefill_input_tps_avg':(statistics.mean(pin) if pin else None),"
        "'server_prefill_input_tps_max':(max(pin) if pin else None),"
        "'server_decode_batches':len(dec),'server_decode_gen_tps_avg':(statistics.mean(dec) if dec else None),"
        "'server_decode_gen_tps_max':(max(dec) if dec else None),"
        "'server_decode_run_queue_avg':(statistics.mean(rq) if rq else None),"
        "'server_decode_run_queue_max':(max(rq) if rq else None)};"
        "print(json.dumps(out))"
    )
    result = docker_exec(remote, container, f"python -c {q(py)}", timeout=60, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {}


def wait_ready_or_crash(cfg: dict[str, Any]) -> None:
    remote = remote_target(cfg)
    container = cfg["remote"]["docker_container"]
    host = cfg["remote"].get("client_host", "127.0.0.1")
    port = cfg["remote"]["port"]
    remote_dir = cfg["reports"]["remote_dir"]
    server_log = cfg["reports"].get("server_log_path", f"{remote_dir}/logs/server.log")
    timeout_s = int(cfg["benchmark"].get("ready_timeout_s", 1800))
    deadline = time.time() + timeout_s
    crash_pattern = "Scheduler hit an exception|Traceback|AssertionError|KeyError|RuntimeError|OutOfMemory|OOM|Address already in use"
    while time.time() < deadline:
        probe = docker_exec(
            remote,
            container,
            f"curl -fsS http://{host}:{port}/health || curl -fsS http://{host}:{port}/v1/models",
            timeout=30,
            check=False,
        )
        if probe.returncode == 0:
            logging.info("SGLang server is ready")
            return
        crash = docker_exec(
            remote,
            container,
            f"grep -E '{crash_pattern}' {q(server_log)} | tail -40",
            timeout=30,
            check=False,
        )
        if crash.returncode == 0 and crash.stdout.strip():
            raise RuntimeError(crash.stdout.strip()[-4000:])
        time.sleep(10)
    raise TimeoutError(f"SGLang server did not become ready within {timeout_s}s")


def server_log_line_count(cfg: dict[str, Any]) -> int:
    remote = remote_target(cfg)
    container = cfg["remote"]["docker_container"]
    remote_dir = cfg["reports"]["remote_dir"]
    server_log = cfg["reports"].get("server_log_path", f"{remote_dir}/logs/server.log")
    result = docker_exec(remote, container, f"wc -l < {q(server_log)}", timeout=60, check=False)
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


def capture_server_excerpt(cfg: dict[str, Any], tag: str, start_line: int) -> str:
    remote = remote_target(cfg)
    container = cfg["remote"]["docker_container"]
    remote_dir = cfg["reports"]["remote_dir"]
    server_log = cfg["reports"].get("server_log_path", f"{remote_dir}/logs/server.log")
    excerpt_file = f"{remote_dir}/logs/{tag}.server.log"
    from_line = max(start_line + 1, 1)
    cmd = (
        f"tail -n +{from_line} {q(server_log)} | "
        "grep -Ei 'Prefill batch|Decode batch|cached-token|new-token|gen throughput|input throughput|run-que|run_queue|run queue|OOM|error|exception' "
        f"> {q(excerpt_file)} || true"
    )
    docker_exec(remote, container, cmd, timeout=60, check=False)
    return excerpt_file


def phase_sla_pass(phase: str, row: dict[str, Any], sla: dict[str, Any]) -> bool:
    if phase == "decode":
        tpot = to_float(row.get("mean_tpot_ms"))
        return not math.isnan(tpot) and tpot < float(sla.get("mean_tpot_ms_lt", 75))
    ttft = to_float(row.get("mean_ttft_ms"))
    return not math.isnan(ttft) and ttft < float(sla.get("mean_ttft_s_lt", 3)) * 1000


def to_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def run_point(cfg: dict[str, Any], layout: dict[str, Any], workload: dict[str, Any], rate: float) -> dict[str, Any]:
    remote = remote_target(cfg)
    container = cfg["remote"]["docker_container"]
    remote_dir = cfg["reports"]["remote_dir"]
    phase = workload["phase"]
    input_len = int(workload["input_len"])
    output_len = int(workload["output_len"])
    max_concurrency = int(workload.get("max_concurrency", 512))
    cfg = copy.deepcopy(cfg)
    cfg["benchmark"]["request_rate"] = rate
    cfg["benchmark"]["num_prompts"] = int(workload.get("num_prompts", cfg["benchmark"].get("num_prompts", 512)))
    cfg["benchmark"]["warmup_requests"] = int(workload.get("warmup_requests", cfg["benchmark"].get("warmup_requests", 30)))
    cfg["benchmark"]["dataset_name"] = workload.get("dataset_name", cfg["benchmark"].get("dataset_name", "random"))
    cfg["benchmark"]["random_range_ratio"] = float(workload.get("random_range_ratio", cfg["benchmark"].get("random_range_ratio", 1.0)))
    hit_rate = float(workload.get("hit_rate", cfg["benchmark"].get("hit_rate", 0.0)))
    tag = f"{layout['name']}_{workload['name']}_rr{slug(rate_label(rate))}_mc{max_concurrency}"
    output_file = f"{remote_dir}/bench/{tag}.jsonl"
    log_file = f"{remote_dir}/logs/{tag}.log"
    server_start_line = server_log_line_count(cfg)
    cmd = f"""
set -euo pipefail
rm -f {q(output_file)}
{bench_command(cfg, input_len, output_len, hit_rate, max_concurrency, output_file)} > {q(log_file)} 2>&1
tail -200 {q(log_file)}
"""
    logging.info("Benchmark %s", tag)
    bench_timeout_s = cfg["benchmark"].get("bench_timeout_s")
    if bench_timeout_s not in (None, "", "null"):
        bench_timeout_s = int(bench_timeout_s)
    try:
        result = docker_exec(remote, container, cmd, timeout=bench_timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        timeout_msg = f"Benchmark timed out after {bench_timeout_s}s: {exc}"
        logging.error("%s", timeout_msg)
        result = subprocess.CompletedProcess(exc.cmd, 124, stdout="", stderr=timeout_msg)
    text = result.stdout + "\n" + result.stderr
    data = parse_jsonish(text)
    data.update({k: v for k, v in parse_text_metrics(text).items() if k not in data})
    data.update({k: v for k, v in read_remote_json_summary(cfg, output_file).items() if v not in (None, "")})
    server_excerpt = capture_server_excerpt(cfg, tag, server_start_line)
    data.update({k: v for k, v in summarize_server_excerpt(cfg, server_excerpt).items() if v not in (None, "")})
    output_tps = to_float(data.get("output_throughput"))
    decode_sla_ms = float(cfg["benchmark"].get("sla", {}).get("mean_tpot_ms_lt", 75))
    row = {
        "layout": layout["name"],
        "phase": phase,
        "workload": workload["name"],
        "input_len": input_len,
        "output_len": output_len,
        "request_rate_target": rate_label(rate),
        "max_concurrency": max_concurrency,
        "num_prompts": cfg["benchmark"]["num_prompts"],
        "warmup_requests": cfg["benchmark"]["warmup_requests"],
        "returncode": result.returncode,
        "completed": data.get("completed", ""),
        "success_rate": (to_float(data.get("completed")) / cfg["benchmark"]["num_prompts"] if data.get("completed") not in ("", None) else ""),
        "mean_ttft_ms": data.get("mean_ttft_ms", ""),
        "p90_ttft_ms": data.get("p90_ttft_ms", ""),
        "p99_ttft_ms": data.get("p99_ttft_ms", ""),
        "mean_tpot_ms": data.get("mean_tpot_ms", ""),
        "p99_tpot_ms": data.get("p99_tpot_ms", ""),
        "mean_itl_ms": data.get("mean_itl_ms", ""),
        "p95_itl_ms": data.get("p95_itl_ms", ""),
        "p99_itl_ms": data.get("p99_itl_ms", ""),
        "request_throughput": data.get("request_throughput", ""),
        "input_throughput": data.get("input_throughput", ""),
        "output_throughput": data.get("output_throughput", ""),
        "total_token_throughput": data.get("total_token_throughput", data.get("total_throughput", "")),
        "decode_req_capacity_at_75ms": (output_tps / (1000 / 75) if not math.isnan(output_tps) else ""),
        "server_prefill_input_tps_avg": data.get("server_prefill_input_tps_avg", ""),
        "server_prefill_input_tps_max": data.get("server_prefill_input_tps_max", ""),
        "server_decode_gen_tps_avg": data.get("server_decode_gen_tps_avg", ""),
        "server_decode_gen_tps_max": data.get("server_decode_gen_tps_max", ""),
        "server_decode_run_queue_avg": data.get("server_decode_run_queue_avg", ""),
        "server_decode_run_queue_max": data.get("server_decode_run_queue_max", ""),
        "decode_req_capacity_at_sla_ms": (output_tps / (1000 / decode_sla_ms) if not math.isnan(output_tps) else ""),
        "radix_cache_hit_rate_target": hit_rate,
        "server_observed_cache_ratio": data.get("server_observed_cache_ratio", ""),
        "server_cached_tokens": data.get("server_cached_tokens", ""),
        "server_new_tokens": data.get("server_new_tokens", ""),
        "server_log_excerpt": server_excerpt,
        "remote_output_file": output_file,
        "remote_log_file": log_file,
        "sla_pass": phase_sla_pass(phase, data, cfg["benchmark"].get("sla", {})),
        "raw_summary": json.dumps(data, ensure_ascii=False),
    }
    if result.returncode != 0:
        row["error"] = (result.stderr or result.stdout)[-2000:]
    return row


def write_report(local_dir: Path, rows: list[dict[str, Any]]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    csv_path = local_dir / "sweep_result.csv"
    html_path = local_dir / "sweep_result.html"
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    header = "".join(f"<th>{x}</th>" for x in fieldnames if x != "raw_summary")
    body = "".join("<tr>" + "".join(f"<td>{r.get(x, '')}</td>" for x in fieldnames if x != "raw_summary") + "</tr>" for r in rows)
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>SGLang IFB Sweep</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;font-size:12px}"
        "td,th{border:1px solid #ddd;padding:5px 7px;text-align:right}td:first-child,th:first-child{text-align:left}"
        "th{background:#f1f4f8}</style><h1>SGLang IFB Sweep</h1><table><thead><tr>"
        + header + "</tr></thead><tbody>" + body + "</tbody></table>",
        encoding="utf-8",
    )
    logging.info("Wrote %s and %s", csv_path, html_path)


def read_existing(local_dir: Path) -> list[dict[str, Any]]:
    path = local_dir / "sweep_result.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(args: argparse.Namespace) -> int:
    cfg0 = load_config(Path(args.config))
    layout_names = set(args.layouts.split(",")) if args.layouts else None
    workload_names = set(args.workloads.split(",")) if args.workloads else None
    rate_override = [float(x) for x in args.request_rates.split(",")] if args.request_rates else None
    concurrency_override = [int(x) for x in args.max_concurrency_values.split(",")] if args.max_concurrency_values else None
    layouts = selected(cfg0["layouts"], layout_names)
    workloads = selected(cfg0["workloads"], workload_names)
    local_dir = ROOT_DIR / cfg0["reports"]["local_dir"]
    rows = [] if args.no_resume else read_existing(local_dir)
    done = {
        (r.get("layout"), r.get("workload"), str(r.get("request_rate_target")), str(r.get("max_concurrency")))
        for r in rows
        if str(r.get("returncode")) == "0"
    }
    if args.dry_run:
        total = 0
        for layout in layouts:
            for workload in workloads:
                for max_concurrency in max_concurrency_values(workload, concurrency_override):
                    for rate in request_rates(workload, rate_override):
                        total += 1
                        logging.info(
                            "Would run layout=%s workload=%s rr=%s mc=%s",
                            layout["name"],
                            workload["name"],
                            rate_label(rate),
                            max_concurrency,
                        )
        logging.info("Would run %d benchmark points across %d layouts and %d workloads", total, len(layouts), len(workloads))
        return 0
    for layout in layouts:
        cfg = copy.deepcopy(cfg0)
        cfg["server"] = merge_server(cfg0["server_base"], layout)
        ensure_remote_dirs(cfg)
        try:
            if not args.reuse_server:
                cleanup_server(cfg)
                start_server(cfg)
            wait_ready_or_crash(cfg)
        except Exception as exc:
            logging.exception("Failed to start layout %s: %s", layout["name"], exc)
            rows.append(
                {
                    "layout": layout["name"],
                    "phase": "",
                    "workload": "",
                    "input_len": "",
                    "output_len": "",
                    "request_rate_target": "",
                    "max_concurrency": "",
                    "num_prompts": "",
                    "warmup_requests": "",
                    "returncode": 1,
                    "completed": "",
                    "success_rate": "",
                    "mean_ttft_ms": "",
                    "p90_ttft_ms": "",
                    "p99_ttft_ms": "",
                    "mean_tpot_ms": "",
                    "p99_tpot_ms": "",
                    "mean_itl_ms": "",
                    "p95_itl_ms": "",
                    "p99_itl_ms": "",
                    "request_throughput": "",
                    "input_throughput": "",
                    "output_throughput": "",
                    "total_token_throughput": "",
                    "decode_req_capacity_at_75ms": "",
                    "server_prefill_input_tps_avg": "",
                    "server_prefill_input_tps_max": "",
                    "server_decode_gen_tps_avg": "",
                    "server_decode_gen_tps_max": "",
                    "server_decode_run_queue_avg": "",
                    "server_decode_run_queue_max": "",
                    "decode_req_capacity_at_sla_ms": "",
                    "radix_cache_hit_rate_target": "",
                    "server_observed_cache_ratio": "",
                    "server_cached_tokens": "",
                    "server_new_tokens": "",
                    "server_log_excerpt": "",
                    "remote_output_file": "",
                    "remote_log_file": "",
                    "sla_pass": False,
                    "raw_summary": "{}",
                    "error": str(exc),
                }
            )
            write_report(local_dir, rows)
            cleanup_server(cfg)
            continue
        try:
            for workload in workloads:
                stop_workload = False
                for max_concurrency in max_concurrency_values(workload, concurrency_override):
                    workload_cfg = copy.deepcopy(workload)
                    workload_cfg["max_concurrency"] = max_concurrency
                    for rate in request_rates(workload_cfg, rate_override):
                        key = (layout["name"], workload_cfg["name"], rate_label(rate), str(max_concurrency))
                        if key in done:
                            logging.info("Skipping completed %s/%s/rps%s/mc%s", *key)
                            continue
                        row = run_point(cfg, layout, workload_cfg, rate)
                        rows.append(row)
                        write_report(local_dir, rows)
                        if args.stop_on_sla_fail and not row["sla_pass"]:
                            logging.info(
                                "Stopping workload after SLA miss: %s %s rps=%s mc=%s",
                                layout["name"],
                                workload_cfg["name"],
                                rate,
                                max_concurrency,
                            )
                            stop_workload = True
                            break
                        time.sleep(int(cfg["benchmark"].get("cooldown_s", 3)))
                    if stop_workload:
                        break
        finally:
            if not args.reuse_server:
                cleanup_server(cfg)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SGLang IFB layouts with request-rate driven prefill/decode workloads.")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "sglang_ifb_prefill_decode_sweep.yaml"))
    parser.add_argument("--layouts", help="Comma-separated layout names to run.")
    parser.add_argument("--workloads", help="Comma-separated workload names to run.")
    parser.add_argument("--request-rates", help="Comma-separated request rates overriding workload config.")
    parser.add_argument("--max-concurrency-values", help="Comma-separated max-concurrency values overriding workload config.")
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-sla-fail", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    try:
        return run(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130
    except Exception as exc:
        logging.exception("Sweep failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
