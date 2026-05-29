#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import logging
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
REPORT_DIR = ROOT_DIR / "reports"
BENCH_FIELDS = [
    "case_name",
    "framework",
    "mode",
    "pd_layout",
    "model_path",
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
    "atten_cp_size",
    "quantization",
    "input_len",
    "output_len",
    "concurrency",
    "num_prompts",
    "success_rate",
    "request_throughput_rps",
    "output_token_throughput_tps",
    "total_token_throughput_tps",
    "ttft_avg_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_avg_ms",
    "itl_avg_ms",
    "e2e_avg_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "failed_requests",
    "benchmark_elapsed_s",
    "error",
]


@dataclass
class RequestMetric:
    ok: bool
    error: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float | None
    e2e_s: float
    itl_s: float | None

    @property
    def tpot_s(self) -> float | None:
        if self.output_tokens <= 0:
            return None
        if self.ttft_s is None:
            return self.e2e_s / self.output_tokens
        decode_tokens = max(self.output_tokens - 1, 1)
        return max(self.e2e_s - self.ttft_s, 0.0) / decode_tokens


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to read cases file {path}: {exc}") from exc
    defaults = doc.get("defaults", {}) or {}
    cases = doc.get("cases", []) or []
    if not isinstance(cases, list):
        raise ValueError("configs/cases.yaml must contain a list at key 'cases'")
    return defaults, cases


def merged_case(defaults: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(case)
    required = [
        "name",
        "framework",
        "mode",
        "model_path",
        "tp_size",
        "input_len",
        "output_len",
        "concurrency",
        "num_prompts",
    ]
    missing = [k for k in required if merged.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Case {case.get('name', '<unnamed>')} missing fields: {missing}")
    return merged


def case_env(case: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    mapping = {
        "CASE_NAME": "name",
        "FRAMEWORK": "framework",
        "MODE": "mode",
        "PD_LAYOUT": "pd_layout",
        "MODEL_PATH": "model_path",
        "HOST": "host",
        "BASE_PORT": "base_port",
        "TP_SIZE": "tp_size",
        "PP_SIZE": "pp_size",
        "DP_SIZE": "dp_size",
        "EP_SIZE": "ep_size",
        "ATTEN_CP_SIZE": "atten_cp_size",
        "QUANTIZATION": "quantization",
        "MAX_NUM_BATCHED_TOKENS": "max_num_batched_tokens",
        "MAX_NUM_SEQS": "max_num_seqs",
    }
    for env_key, case_key in mapping.items():
        value = case.get(case_key)
        env[env_key] = "" if value is None else str(value)
    return env


def deploy_case(case: dict[str, Any]) -> None:
    framework = str(case["framework"]).lower()
    script = ROOT_DIR / "deploy" / f"{framework}_pd.sh"
    if not script.exists():
        raise FileNotFoundError(f"No deploy script for framework={framework}: {script}")
    logging.info("Deploying case %s with %s", case["name"], script)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=str(ROOT_DIR),
        env=case_env(case),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        logging.info(result.stdout.strip())
    if result.stderr:
        logging.warning(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Deploy failed for {case['name']} with exit code {result.returncode}")


def cleanup_case(case: dict[str, Any]) -> None:
    script = ROOT_DIR / "deploy" / "cleanup.sh"
    try:
        subprocess.run(
            ["bash", str(script)],
            cwd=str(ROOT_DIR),
            env=case_env(case),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        logging.warning("Cleanup failed for %s: %s", case.get("name"), exc)


async def wait_health(session: aiohttp.ClientSession, base_url: str, timeout_s: int, interval_s: int) -> None:
    deadline = time.perf_counter() + timeout_s
    paths = ["/health", "/v1/models"]
    last_error = ""
    while time.perf_counter() < deadline:
        for path in paths:
            try:
                async with session.get(f"{base_url}{path}", timeout=10) as resp:
                    text = await resp.text()
                    if resp.status < 500:
                        logging.info("Health check passed at %s: HTTP %s", path, resp.status)
                        return
                    last_error = f"{path}: HTTP {resp.status} {text[:200]}"
            except Exception as exc:
                last_error = f"{path}: {exc}"
        await asyncio.sleep(interval_s)
    raise TimeoutError(f"Service health check timed out after {timeout_s}s. Last error: {last_error}")


def make_prompt(input_len: int, seed: int, prompt_char: str) -> str:
    random.seed(seed)
    prefix = "You are benchmarking an LLM inference server. Repeat the pattern and answer concisely.\n"
    body_len = max(input_len - len(prefix), 1)
    return prefix + (prompt_char * body_len)


def parse_stream_chunk(line: bytes) -> str:
    text = line.decode("utf-8", errors="ignore").strip()
    if not text.startswith("data:"):
        return ""
    data = text[5:].strip()
    if data == "[DONE]":
        return ""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or choices[0].get("text") or {}
    if isinstance(delta, str):
        return delta
    return delta.get("content") or ""


async def one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    case: dict[str, Any],
    prompt: str,
    prompt_tokens_estimate: int,
) -> RequestMetric:
    started = time.perf_counter()
    first_token_time: float | None = None
    token_times: list[float] = []
    output_text = ""
    payload = {
        "model": case["model_path"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(case["output_len"]),
        "temperature": 0,
        "stream": True,
    }
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=float(case.get("request_timeout_s", 300))),
        ) as resp:
            if resp.status >= 400:
                error = await resp.text()
                return RequestMetric(False, f"HTTP {resp.status}: {error[:300]}", prompt_tokens_estimate, 0, None, time.perf_counter() - started, None)
            async for line in resp.content:
                content = parse_stream_chunk(line)
                if not content:
                    continue
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                output_text += content
    except Exception as exc:
        return RequestMetric(False, str(exc), prompt_tokens_estimate, 0, None, time.perf_counter() - started, None)

    ended = time.perf_counter()
    output_tokens_estimate = max(len(token_times), len(output_text.split()), 1 if output_text else 0)
    intervals = [b - a for a, b in zip(token_times, token_times[1:])]
    itl = statistics.mean(intervals) if intervals else None
    ttft = first_token_time - started if first_token_time is not None else None
    return RequestMetric(True, "", prompt_tokens_estimate, output_tokens_estimate, ttft, ended - started, itl)


async def run_load(session: aiohttp.ClientSession, base_url: str, case: dict[str, Any], total: int, label: str) -> list[RequestMetric]:
    concurrency = int(case["concurrency"])
    prompt = make_prompt(int(case["input_len"]), int(case.get("seed", 42)), str(case.get("prompt_char", "A")))
    prompt_tokens_estimate = max(int(case["input_len"]), 1)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for idx in range(total):
        queue.put_nowait(idx)

    results: list[RequestMetric] = []

    async def worker(worker_id: int) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            metric = await one_request(session, base_url, case, prompt, prompt_tokens_estimate)
            if not metric.ok:
                logging.debug("%s worker %s request failed: %s", label, worker_id, metric.error)
            results.append(metric)
            queue.task_done()

    started = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    elapsed = time.perf_counter() - started
    logging.info("%s completed: %s requests in %.2fs", label, len(results), elapsed)
    return results


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def summarize(case: dict[str, Any], metrics: list[RequestMetric], elapsed_s: float) -> dict[str, Any]:
    ok = [m for m in metrics if m.ok]
    ttft = [m.ttft_s for m in ok if m.ttft_s is not None]
    tpot = [m.tpot_s for m in ok if m.tpot_s is not None]
    itl = [m.itl_s for m in ok if m.itl_s is not None]
    e2e = [m.e2e_s for m in ok]
    output_tokens = sum(m.output_tokens for m in ok)
    input_tokens = sum(m.prompt_tokens for m in ok)
    total = len(metrics)
    success_rate = len(ok) / total if total else 0.0
    return {
        "case_name": case["name"],
        "framework": case["framework"],
        "mode": case["mode"],
        "pd_layout": case.get("pd_layout") or "",
        "model_path": case["model_path"],
        "tp_size": case.get("tp_size", ""),
        "pp_size": case.get("pp_size", ""),
        "dp_size": case.get("dp_size", ""),
        "ep_size": case.get("ep_size", ""),
        "atten_cp_size": case.get("atten_cp_size", ""),
        "quantization": case.get("quantization") or "",
        "input_len": case["input_len"],
        "output_len": case["output_len"],
        "concurrency": case["concurrency"],
        "num_prompts": case["num_prompts"],
        "success_rate": success_rate,
        "request_throughput_rps": len(ok) / elapsed_s if elapsed_s > 0 else 0.0,
        "output_token_throughput_tps": output_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "total_token_throughput_tps": (input_tokens + output_tokens) / elapsed_s if elapsed_s > 0 else 0.0,
        "ttft_avg_ms": statistics.mean(ttft) * 1000 if ttft else "",
        "ttft_p50_ms": percentile(ttft, 0.50) * 1000 if ttft else "",
        "ttft_p95_ms": percentile(ttft, 0.95) * 1000 if ttft else "",
        "tpot_avg_ms": statistics.mean(tpot) * 1000 if tpot else "",
        "itl_avg_ms": statistics.mean(itl) * 1000 if itl else "",
        "e2e_avg_ms": statistics.mean(e2e) * 1000 if e2e else "",
        "e2e_p50_ms": percentile(e2e, 0.50) * 1000 if e2e else "",
        "e2e_p95_ms": percentile(e2e, 0.95) * 1000 if e2e else "",
        "failed_requests": total - len(ok),
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    normalized = {field: row.get(field, "") for field in BENCH_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BENCH_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


async def run_case(case: dict[str, Any], skip_deploy: bool) -> dict[str, Any]:
    host = case.get("host", "127.0.0.1")
    base_port = int(case.get("base_port", 18000))
    base_url = f"http://{host}:{base_port}"
    if not skip_deploy:
        deploy_case(case)
    try:
        async with aiohttp.ClientSession() as session:
            await wait_health(
                session,
                base_url,
                int(case.get("health_timeout_s", 600)),
                int(case.get("startup_poll_interval_s", 5)),
            )
            warmup = max(int(case.get("warmup_prompts", 30)), 30)
            await run_load(session, base_url, case, warmup, "warmup")
            started = time.perf_counter()
            metrics = await run_load(session, base_url, case, int(case["num_prompts"]), "benchmark")
            elapsed_s = time.perf_counter() - started
            row = summarize(case, metrics, elapsed_s)
            row["benchmark_elapsed_s"] = elapsed_s
            return row
    finally:
        if not skip_deploy and bool(case.get("stop_after_case", True)):
            cleanup_case(case)


def select_cases(cases: list[dict[str, Any]], names: set[str] | None, include_disabled: bool) -> list[dict[str, Any]]:
    selected = []
    for case in cases:
        if names and case.get("name") not in names:
            continue
        if not include_disabled and not case.get("enabled", True):
            continue
        selected.append(case)
    return selected


async def main_async(args: argparse.Namespace) -> int:
    defaults, raw_cases = load_cases(Path(args.config))
    names = set(args.case or []) or None
    cases = [merged_case(defaults, c) for c in raw_cases]
    cases = select_cases(cases, names, args.include_disabled)
    if not cases:
        logging.warning("No cases selected. Enable cases in configs/cases.yaml or pass --include-disabled --case NAME.")
        return 0
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_csv = REPORT_DIR / "bench_metrics.csv"
    for case in cases:
        logging.info("Running case: %s", case["name"])
        try:
            row = await run_case(case, args.skip_deploy)
            append_csv(result_csv, row)
            logging.info("Case %s finished: success_rate=%.3f rps=%.3f", case["name"], row["success_rate"], row["request_throughput_rps"])
        except Exception as exc:
            logging.exception("Case %s failed: %s", case.get("name"), exc)
            append_csv(
                result_csv,
                {
                    "case_name": case.get("name", ""),
                    "framework": case.get("framework", ""),
                    "mode": case.get("mode", ""),
                    "pd_layout": case.get("pd_layout") or "",
                    "error": str(exc),
                },
            )
            if args.fail_fast:
                return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy and benchmark vLLM/SGLang normal and PD cases.")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "cases.yaml"))
    parser.add_argument("--case", action="append", help="Run only this case name. Can be passed multiple times.")
    parser.add_argument("--include-disabled", action="store_true", help="Allow running disabled cases selected by --case or all cases.")
    parser.add_argument("--skip-deploy", action="store_true", help="Benchmark an already running endpoint.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
