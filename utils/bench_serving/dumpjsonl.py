import json
import sys
from copy import deepcopy
from typing import Any


def load_dumpjsonl_payloads(dataset_path: str) -> list[dict[str, Any]]:
    """Load valid JSON object payloads from a JSONL file."""
    payloads = []
    with open(dataset_path, encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                print(
                    f"Warning: skipping empty JSONL record at line {line_number}.",
                    file=sys.stderr,
                )
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSONL record at line "
                    f"{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(payload, dict):
                print(
                    f"Warning: skipping non-object JSONL record at line "
                    f"{line_number}.",
                    file=sys.stderr,
                )
                continue
            payloads.append(payload)

    if not payloads:
        raise ValueError(
            f"No valid JSON object records found in dataset: {dataset_path}"
        )
    return payloads


def prepare_dumpjsonl_payload(
    payload: dict[str, Any],
    model_name: str,
    output_len: int,
) -> dict[str, Any]:
    """Copy and apply mandatory benchmark overrides to a dumped payload."""
    prepared_payload = deepcopy(payload)
    prepared_payload["model"] = model_name
    prepared_payload["max_tokens"] = output_len
    prepared_payload.update({
        "stream": True,
        "stream_options": prepared_payload.get(
            "stream_options", {"include_usage": True}
        ),
    })
    return prepared_payload


def parse_usage_tokens(usage: Any) -> tuple[int, int]:
    """Return validated prompt and completion token counts from SSE usage."""
    if not isinstance(usage, dict):
        raise ValueError("SSE response did not contain a usage object.")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or prompt_tokens < 0
        or not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens < 0
    ):
        raise ValueError(
            "SSE usage must contain non-negative integer prompt_tokens "
            "and completion_tokens."
        )
    return prompt_tokens, completion_tokens
