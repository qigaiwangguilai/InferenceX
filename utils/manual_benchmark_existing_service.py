#!/usr/bin/env python3
"""Benchmark an already-running OpenAI-compatible service from config keys.

This intentionally does not start containers, Slurm jobs, srt-slurm, or GitHub
Actions. It reuses the existing matrix generator, benchmark client, and result
processor so local endpoint runs stay comparable to InferenceX configs.

Examples:

  # Basic: run one GB200 DSV4 Dynamo-vLLM point against an existing service.
  # By default this calls http://HOST:PORT/v1/chat/completions.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-gb200-dynamo-vllm \
    --base-url http://HOST:PORT \
    --seq-lens 8k1k \
    --conc 256

  # Same run with bearer auth. Equivalent to exporting OPENAI_API_KEY first.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-gb200-dynamo-vllm \
    --base-url http://HOST:PORT \
    --api-key "$OPENAI_API_KEY" \
    --seq-lens 8k1k \
    --conc 256

  # Single-node MTP config against an existing vLLM endpoint. The result will
  # carry spec_decoding=mtp from the config key; the service itself must already
  # have been started with the matching MTP serving flags.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-b200-vllm-mtp \
    --base-url http://HOST:PORT \
    --seq-lens 1k1k \
    --conc 16

  # Use the legacy completions endpoint instead of chat completions.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-gb200-dynamo-vllm \
    --base-url http://HOST:PORT \
    --backend openai \
    --endpoint /v1/completions \
    --seq-lens 8k1k \
    --conc 256

  # Use a local tokenizer/model path while sending a service-visible model name.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-gb200-dynamo-vllm \
    --base-url http://HOST:PORT \
    --seq-lens 8k1k \
    --conc 256 512 1024 \
    --model-override /models/deepseek-v4-pro \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro

  # Run multiple concurrencies and keep raw/aggregated JSON under a custom dir.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-gb200-dynamo-vllm \
    --base-url http://HOST:PORT \
    --seq-lens 8k1k \
    --conc 256 512 1024 \
    --result-dir ./manual_results/dsv4_gb200

  # Skip health checks during local iteration and only print commands.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-gb200-dynamo-vllm \
    --base-url http://HOST:PORT \
    --seq-lens 8k1k \
    --conc 256 \
    --skip-health-check \
    --dry-run

  # Override GPU denominators used by process_result.py aggregation. Use this
  # when the actual P/D deployment differs from nvidia-master.yaml labels.
  python3 utils/manual_benchmark_existing_service.py \
    --config-files configs/nvidia-master.yaml \
    --config-key dsv4-fp4-gb200-dynamo-vllm \
    --base-url http://HOST:PORT \
    --seq-lens 8k1k \
    --conc 256 \
    --prefill-gpus 8 \
    --decode-gpus 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "utils/matrix_logic/generate_sweep_configs.py"
BENCHMARK_CLIENT = REPO_ROOT / "utils/bench_serving/benchmark_serving.py"
PROCESS_RESULT = REPO_ROOT / "utils/process_result.py"


def run_command(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    capture: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return None
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=capture,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise


def request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 10,
) -> tuple[int, str]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        return exc.code, body


def health_check(
    base_url: str,
    served_model_name: str,
    *,
    backend: str,
    endpoint: str,
    timeout: int,
    interval: float,
) -> None:
    deadline = time.monotonic() + timeout
    base_url = base_url.rstrip("/")
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    last_error = ""

    while time.monotonic() < deadline:
        try:
            status, body = request("GET", f"{base_url}/health", timeout=10)
            if 200 <= status < 300:
                print(f"[health] GET /health -> {status}")
                return
            last_error = f"GET /health -> {status}: {body[:300]}"

            status, body = request("GET", f"{base_url}/v1/models", timeout=10)
            if 200 <= status < 300:
                print(f"[health] GET /v1/models -> {status}")
                return
            last_error = f"GET /v1/models -> {status}: {body[:300]}"

            if backend == "openai-chat" or endpoint.endswith("/chat/completions"):
                payload = {
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                }
            else:
                payload = {
                    "model": served_model_name,
                    "prompt": "hello",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                }
            status, body = request(
                "POST",
                f"{base_url}{endpoint}",
                payload=payload,
                timeout=20,
            )
            if 200 <= status < 300:
                print(f"[health] POST {endpoint} -> {status}")
                return
            last_error = f"POST {endpoint} -> {status}: {body[:300]}"
        except Exception as exc:  # noqa: BLE001 - health output should preserve root cause.
            last_error = repr(exc)

        print(f"[health] not ready: {last_error}")
        time.sleep(interval)

    raise RuntimeError(
        f"Service did not become healthy within {timeout}s. Last error: {last_error}"
    )


def generate_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "test-config",
        "--config-files",
        *args.config_files,
        "--config-keys",
        args.config_key,
        "--no-evals",
    ]
    if args.seq_lens:
        cmd.extend(["--seq-lens", *args.seq_lens])
    if args.conc:
        cmd.extend(["--conc", *[str(c) for c in args.conc]])

    proc = run_command(cmd, capture=True)
    assert proc is not None
    return json.loads(proc.stdout)


def config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def config_int(config: dict[str, Any], key: str, default: int = 1) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ValueError(f"Config field {key!r} must be a positive integer, got {value}.")
    return value


def worker_gpu_count(worker: dict[str, Any]) -> int:
    num_workers = int(worker["num-worker"])
    if num_workers < 0:
        raise ValueError(
            f"Config field 'num-worker' must be a non-negative integer, got {num_workers}."
        )
    return (
        num_workers
        * config_int(worker, "tp")
        * config_int(worker, "pp")
        * config_int(worker, "pcp-size")
    )


def infer_chat_template_mode(
    config: dict[str, Any],
    explicit: str,
    backend: str,
) -> tuple[bool, bool]:
    if explicit == "none":
        return False, False
    if explicit == "chat":
        return True, False
    if explicit == "dsv4":
        return True, True
    if backend == "openai-chat":
        return False, False
    if config.get("model-prefix") == "dsv4":
        return True, True
    return False, False


def result_stem(config: dict[str, Any], conc: int) -> str:
    parts = [
        "manual",
        str(config["exp-name"]),
        str(config["precision"]),
        str(config["framework"]),
        f"conc{conc}",
    ]
    if "prefill" in config:
        p = config["prefill"]
        d = config["decode"]
        parts.extend([
            f"p{p['num-worker']}tp{p['tp']}pp{config_int(p, 'pp')}"
            f"dcp{config_int(p, 'dcp-size')}pcp{config_int(p, 'pcp-size')}ep{p['ep']}",
            f"d{d['num-worker']}tp{d['tp']}pp{config_int(d, 'pp')}"
            f"dcp{config_int(d, 'dcp-size')}pcp{config_int(d, 'pcp-size')}ep{d['ep']}",
        ])
    else:
        parts.append(
            f"tp{config['tp']}pp{config_int(config, 'pp')}"
            f"dcp{config_int(config, 'dcp-size')}pcp{config_int(config, 'pcp-size')}"
            f"ep{config.get('ep', 1)}"
        )
    return "_".join(parts).replace("/", "_")


def benchmark_one(
    config: dict[str, Any],
    conc: int,
    args: argparse.Namespace,
) -> Path:
    model = args.model_override or config["model"]
    served_model_name = args.served_model_name or config["model"]
    stem = result_stem(config, conc)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    num_prompts = args.num_prompts
    if num_prompts is None:
        num_prompts = max(args.min_num_prompts, conc * args.num_prompts_multiplier)

    use_chat_template, use_dsv4 = infer_chat_template_mode(
        config,
        args.chat_template,
        args.backend,
    )

    cmd = [
        sys.executable,
        str(BENCHMARK_CLIENT),
        "--model",
        str(model),
        "--served-model-name",
        str(served_model_name),
        "--backend",
        args.backend,
        "--base-url",
        args.base_url.rstrip("/"),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(config["isl"]),
        "--random-output-len",
        str(config["osl"]),
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(conc),
        "--request-rate",
        str(args.request_rate),
        "--ignore-eos",
        "--save-result",
        "--num-warmups",
        str(args.num_warmups if args.num_warmups is not None else 2 * conc),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        f"{stem}.json",
    ]
    if args.endpoint:
        cmd.extend(["--endpoint", args.endpoint])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    if args.tokenizer_mode:
        cmd.extend(["--tokenizer-mode", args.tokenizer_mode])
    if use_chat_template:
        cmd.append("--use-chat-template")
    if use_dsv4:
        cmd.append("--dsv4")
    if args.random_num_workers is not None:
        cmd.extend(["--random-num-workers", str(args.random_num_workers)])
    if args.save_detailed:
        cmd.append("--save-detailed")

    print(
        f"\n[benchmark] {config['exp-name']} conc={conc} "
        f"isl={config['isl']} osl={config['osl']} prompts={num_prompts}"
    )
    run_command(cmd, dry_run=args.dry_run)
    return result_dir / f"{stem}.json"


def process_one(
    config: dict[str, Any],
    raw_result: Path,
    args: argparse.Namespace,
) -> Path | None:
    stem = raw_result.stem
    env = os.environ.copy()
    env.pop("ROUTER_METADATA", None)
    env.pop("KV_P2P_TRANSFER", None)
    env.update({
        "RUNNER_TYPE": str(config["runner"]),
        "FRAMEWORK": str(config["framework"]),
        "PRECISION": str(config["precision"]),
        "SPEC_DECODING": str(config.get("spec-decoding", "none")),
        "RESULT_FILENAME": stem,
        "ISL": str(config["isl"]),
        "OSL": str(config["osl"]),
        "DISAGG": str(config.get("disagg", False)).lower(),
        "MODEL_PREFIX": str(config["model-prefix"]),
        "IMAGE": str(config["image"]),
    })

    router = config.get("router")
    if router is not None:
        env["ROUTER_METADATA"] = json.dumps(router)

    kv_p2p_transfer = config.get("kv-p2p-transfer")
    if kv_p2p_transfer:
        env["KV_P2P_TRANSFER"] = str(kv_p2p_transfer)

    if "prefill" in config:
        p = config["prefill"]
        d = config["decode"]
        prefill_gpus = args.prefill_gpus
        if prefill_gpus is None:
            prefill_gpus = worker_gpu_count(p)
        decode_gpus = args.decode_gpus
        if decode_gpus is None:
            decode_gpus = worker_gpu_count(d)
        env.update({
            "IS_MULTINODE": "true",
            "PREFILL_GPUS": str(prefill_gpus),
            "DECODE_GPUS": str(decode_gpus),
            "PREFILL_NUM_WORKERS": str(p["num-worker"]),
            "PREFILL_TP": str(p["tp"]),
            "PREFILL_PP_SIZE": str(config_int(p, "pp")),
            "PREFILL_DCP_SIZE": str(config_int(p, "dcp-size")),
            "PREFILL_PCP_SIZE": str(config_int(p, "pcp-size")),
            "PREFILL_EP": str(p["ep"]),
            "PREFILL_DP_ATTN": str(config_bool(p.get("dp-attn"))).lower(),
            "DECODE_NUM_WORKERS": str(d["num-worker"]),
            "DECODE_TP": str(d["tp"]),
            "DECODE_PP_SIZE": str(config_int(d, "pp")),
            "DECODE_DCP_SIZE": str(config_int(d, "dcp-size")),
            "DECODE_PCP_SIZE": str(config_int(d, "pcp-size")),
            "DECODE_EP": str(d["ep"]),
            "DECODE_DP_ATTN": str(config_bool(d.get("dp-attn"))).lower(),
        })
    else:
        if config_bool(config.get("disagg")):
            raise ValueError("Disaggregated config did not include prefill/decode fields.")
        env.update({
            "IS_MULTINODE": "false",
            "TP": str(config["tp"]),
            "PP_SIZE": str(config_int(config, "pp")),
            "DCP_SIZE": str(config_int(config, "dcp-size")),
            "PCP_SIZE": str(config_int(config, "pcp-size")),
            "EP_SIZE": str(config.get("ep", 1)),
            "DP_ATTENTION": str(config_bool(config.get("dp-attn"))).lower(),
        })

    # process_result.py reads ./RESULT_FILENAME.json and writes ./agg_RESULT_FILENAME.json.
    run_command([sys.executable, str(PROCESS_RESULT)], cwd=raw_result.parent, env=env, dry_run=args.dry_run)
    if args.dry_run:
        return None
    return raw_result.parent / f"agg_{stem}.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _total_gpu_denominator(row: dict[str, Any]) -> float:
    if row.get("is_multinode"):
        return float(row.get("num_prefill_gpu", 0)) + float(row.get("num_decode_gpu", 0))
    return (
        float(row.get("tp", 0))
        * float(row.get("pp", 1))
        * float(row.get("pcp_size", 1))
    )


def _output_gpu_denominator(row: dict[str, Any]) -> float:
    if row.get("is_multinode"):
        decode_gpus = float(row.get("num_decode_gpu", 0))
        return decode_gpus if decode_gpus > 0 else _total_gpu_denominator(row)
    return _total_gpu_denominator(row)


def infer_total_token_throughput(row: dict[str, Any]) -> float:
    if "total_token_throughput" in row:
        return float(row["total_token_throughput"])
    return float(row.get("tput_per_gpu", 0)) * _total_gpu_denominator(row)


def infer_output_throughput(row: dict[str, Any]) -> float:
    if "output_throughput" in row:
        return float(row["output_throughput"])
    return float(row.get("output_tput_per_gpu", 0)) * _output_gpu_denominator(row)


def print_summary(agg_paths: list[Path]) -> None:
    if not agg_paths:
        return
    rows = []
    for path in agg_paths:
        data = load_json(path)
        rows.append(data)

    headers = [
        "conc",
        "P/D",
        "tok/s",
        "out tok/s",
        "tok/s/GPU",
        "out/GPU",
        "mean TTFT ms",
        "p99 TTFT ms",
        "mean TPOT ms",
        "p99 TPOT ms",
    ]
    table_rows = []
    for r in sorted(rows, key=lambda x: (x.get("conc", 0), x.get("num_prefill_gpu", 0), x.get("num_decode_gpu", 0))):
        if r.get("is_multinode"):
            pd = f"{r['prefill_num_workers']}P/{r['decode_num_workers']}D"
        else:
            pd = f"TP{r['tp']}"
        table_rows.append([
            str(r.get("conc", "")),
            pd,
            f"{infer_total_token_throughput(r):.2f}",
            f"{infer_output_throughput(r):.2f}",
            f"{float(r.get('tput_per_gpu', 0)):.2f}",
            f"{float(r.get('output_tput_per_gpu', 0)):.2f}",
            f"{float(r.get('mean_ttft', 0)) * 1000:.2f}",
            f"{float(r.get('p99_ttft', 0)) * 1000:.2f}",
            f"{float(r.get('mean_tpot', 0)) * 1000:.2f}",
            f"{float(r.get('p99_tpot', 0)) * 1000:.2f}",
        ])

    widths = [
        max(len(str(row[i])) for row in [headers, *table_rows])
        for i in range(len(headers))
    ]
    numeric_cols = set(range(2, len(headers)))

    def format_row(row: list[str]) -> str:
        cells = []
        for i, cell in enumerate(row):
            if i in numeric_cols:
                cells.append(str(cell).rjust(widths[i]))
            else:
                cells.append(str(cell).ljust(widths[i]))
        return "  ".join(cells)

    print()
    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in table_rows:
        print(format_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an existing OpenAI-compatible service using an InferenceX config key."
    )
    parser.add_argument("--config-files", nargs="+", required=True)
    parser.add_argument("--config-key", required=True)
    parser.add_argument("--base-url", required=True, help="Existing service base URL, e.g. http://host:8000")
    parser.add_argument(
        "--api-key",
        help="API key for the existing service. Overrides OPENAI_API_KEY for this run.",
    )
    parser.add_argument("--seq-lens", nargs="+", help="Optional sequence length filter, e.g. 8k1k")
    parser.add_argument("--conc", nargs="+", type=int, help="Optional concurrency filter")
    parser.add_argument("--result-dir", default="manual_results")

    parser.add_argument("--model-override", help="Tokenizer/model id or local path for prompt generation")
    parser.add_argument("--served-model-name", help="Model name sent in OpenAI API payload")
    parser.add_argument("--tokenizer", help="Tokenizer id/path if different from --model")
    parser.add_argument(
        "--tokenizer-mode",
        default="auto",
        choices=["auto", "slow", "mistral", "custom", "deepseek_v4"],
    )
    parser.add_argument("--backend", default="openai-chat")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--random-range-ratio", type=float, default=0.8)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--num-prompts-multiplier", type=int, default=10)
    parser.add_argument("--min-num-prompts", type=int, default=16)
    parser.add_argument("--num-warmups", type=int)
    parser.add_argument("--random-num-workers", type=int)
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
    )
    parser.add_argument(
        "--chat-template",
        choices=["auto", "none", "chat", "dsv4"],
        default="auto",
        help=(
            "auto uses server-side chat formatting for openai-chat, and the "
            "DeepSeek-V4 template for model-prefix=dsv4 on completions-style backends."
        ),
    )

    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-timeout", type=int, default=300)
    parser.add_argument("--health-interval", type=float, default=5)

    parser.add_argument("--prefill-gpus", type=int, help="Override prefill GPU count for aggregation")
    parser.add_argument("--decode-gpus", type=int, help="Override decode GPU count for aggregation")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    matrix = generate_matrix(args)
    if not matrix:
        raise SystemExit("No benchmark configs generated.")

    served_model_name = args.served_model_name or matrix[0]["model"]
    if not args.skip_health_check:
        health_check(
            args.base_url,
            served_model_name,
            backend=args.backend,
            endpoint=args.endpoint,
            timeout=args.health_timeout,
            interval=args.health_interval,
        )

    agg_paths: list[Path] = []
    for config in matrix:
        conc_values = config.get("conc")
        if isinstance(conc_values, int):
            conc_values = [conc_values]
        if not conc_values:
            raise ValueError(f"Config has no conc values: {config}")
        for conc in conc_values:
            raw_path = benchmark_one(config, int(conc), args)
            agg_path = process_one(config, raw_path, args)
            if agg_path is not None:
                agg_paths.append(agg_path)

    print_summary(agg_paths)
    if agg_paths:
        print("\nAggregated results:")
        for path in agg_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
