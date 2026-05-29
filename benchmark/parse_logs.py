#!/usr/bin/env python3
import argparse
import csv
import html
import logging
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
REPORT_DIR = ROOT_DIR / "reports"

PATTERNS = {
    "prefill_ms": re.compile(r"prefill[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)\s*(ms|s)", re.IGNORECASE),
    "decode_ms": re.compile(r"decode[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)\s*(ms|s)", re.IGNORECASE),
    "kv_transfer_ms": re.compile(r"(?:kv|cache).{0,30}(?:transfer|send|recv|receive)[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)\s*(ms|s)", re.IGNORECASE),
    "gpu_mem": re.compile(r"(?:gpu|cuda|memory|mem)[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB|MiB|MB)", re.IGNORECASE),
    "oom": re.compile(r"out of memory|cuda oom|oom", re.IGNORECASE),
    "error": re.compile(r"\b(error|exception|traceback|failed|http 4[0-9]{2}|http 5[0-9]{2})\b", re.IGNORECASE),
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def to_ms(value: str, unit: str) -> float:
    num = float(value)
    return num * 1000 if unit.lower() == "s" else num


def to_mib(value: str, unit: str) -> float:
    num = float(value)
    unit = unit.lower()
    return num * 1024 if unit in ("gib", "gb") else num


def stats(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    return f"{sum(values) / len(values):.3f}", f"{max(values):.3f}"


def parse_case_logs(case_dir: Path) -> dict[str, Any]:
    prefill: list[float] = []
    decode: list[float] = []
    kv: list[float] = []
    gpu_mem: list[float] = []
    oom_count = 0
    error_count = 0
    parsed_files = 0
    for log_file in case_dir.glob("*.log"):
        parsed_files += 1
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logging.warning("Failed to read %s: %s", log_file, exc)
            continue
        for match in PATTERNS["prefill_ms"].finditer(text):
            prefill.append(to_ms(match.group(1), match.group(2)))
        for match in PATTERNS["decode_ms"].finditer(text):
            decode.append(to_ms(match.group(1), match.group(2)))
        for match in PATTERNS["kv_transfer_ms"].finditer(text):
            kv.append(to_ms(match.group(1), match.group(2)))
        for match in PATTERNS["gpu_mem"].finditer(text):
            gpu_mem.append(to_mib(match.group(1), match.group(2)))
        oom_count += len(PATTERNS["oom"].findall(text))
        error_count += len(PATTERNS["error"].findall(text))
    prefill_avg, prefill_max = stats(prefill)
    decode_avg, decode_max = stats(decode)
    kv_avg, kv_max = stats(kv)
    gpu_avg, gpu_max = stats(gpu_mem)
    return {
        "case_name": case_dir.name,
        "log_files": parsed_files,
        "prefill_avg_ms": prefill_avg,
        "prefill_max_ms": prefill_max,
        "decode_avg_ms": decode_avg,
        "decode_max_ms": decode_max,
        "kv_transfer_avg_ms": kv_avg,
        "kv_transfer_max_ms": kv_max,
        "gpu_mem_avg_mib": gpu_avg,
        "gpu_mem_max_mib": gpu_max,
        "oom_count": oom_count,
        "error_count": error_count,
    }


def read_bench_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row.get("case_name", ""): row for row in csv.DictReader(f)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("<html><body><p>No results.</p></body></html>", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    header = "".join(f"<th>{html.escape(key)}</th>" for key in fieldnames)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in fieldnames)
        body_rows.append(f"<tr>{cells}</tr>")
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLM Inference Benchmark Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dce5; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f8; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fafbfe; }}
  </style>
</head>
<body>
  <h1>LLM Inference Benchmark Results</h1>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def build_report() -> list[dict[str, Any]]:
    bench = read_bench_metrics(REPORT_DIR / "bench_metrics.csv")
    rows = []
    if LOG_DIR.exists():
        for case_dir in sorted(p for p in LOG_DIR.iterdir() if p.is_dir()):
            row = {}
            row.update(bench.get(case_dir.name, {}))
            row.update(parse_case_logs(case_dir))
            rows.append(row)
    for case_name, bench_row in bench.items():
        if case_name and not any(row.get("case_name") == case_name for row in rows):
            rows.append(bench_row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse framework logs and benchmark metrics into reports.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        rows = build_report()
        write_csv(REPORT_DIR / "result.csv", rows)
        write_html(REPORT_DIR / "result.html", rows)
        logging.info("Wrote %s and %s", REPORT_DIR / "result.csv", REPORT_DIR / "result.html")
        return 0
    except Exception as exc:
        logging.exception("Failed to parse logs: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
