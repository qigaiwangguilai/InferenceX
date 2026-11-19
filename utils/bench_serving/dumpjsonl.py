import json
import sys
from copy import deepcopy
from typing import Any


def validate_dumpjsonl_endpoint(backend: str, endpoint: str) -> None:
    """Require a matching OpenAI backend and endpoint for raw payload replay."""
    if backend == "openai-chat" and endpoint.endswith("/chat/completions"):
        return
    if (
        backend == "openai"
        and endpoint.endswith("/completions")
        and not endpoint.endswith("/chat/completions")
    ):
        return
    raise ValueError(
        "dumpjsonl requires --backend=openai-chat with /v1/chat/completions "
        "or --backend=openai with /v1/completions."
    )


def validate_completion_payload(payload: dict[str, Any]) -> None:
    """Accept one text prompt or one token-ID sequence per benchmark request."""
    if "messages" in payload:
        raise ValueError(
            "Completions JSONL records must use 'prompt', not 'messages'. "
            "Apply the chat template before exporting token IDs."
        )
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) and not (
        isinstance(prompt, list)
        and prompt
        and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0
                for token in prompt)
    ):
        raise ValueError(
            "Completions JSONL 'prompt' must be a string or a non-empty flat "
            "list of non-negative integer token IDs; batched prompts are unsupported."
        )
    if payload.get("n", 1) != 1:
        raise ValueError("Completions JSONL replay requires n=1 for per-request metrics.")


def load_dumpjsonl_payloads(
    dataset_path: str,
    num_prompts: int | None = None,
) -> list[dict[str, Any]]:
    """Load valid JSON objects, truncating or cycling to the requested count."""
    if num_prompts is not None and num_prompts <= 0:
        raise ValueError("--num-prompts must be a positive integer.")
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
    if num_prompts is not None:
        return [payloads[index % len(payloads)] for index in range(num_prompts)]
    return payloads


def prepare_dumpjsonl_payload(
    payload: dict[str, Any],
    model_name: str,
    output_len: int,
    *,
    ignore_eos: bool = False,
    backend: str = "openai-chat",
) -> dict[str, Any]:
    """Copy and apply mandatory benchmark overrides to a dumped payload."""
    # Completion records are validated before the timed run. Only top-level
    # fields change here: do not rescan or deepcopy large, read-only token lists
    # for every replayed request inside the throughput measurement.
    prepared_payload = dict(payload) if backend == "openai" else deepcopy(payload)
    prepared_payload["model"] = model_name
    prepared_payload["max_tokens"] = output_len
    if backend == "openai":
        prepared_payload.pop("max_completion_tokens", None)
    elif "max_completion_tokens" in prepared_payload:
        prepared_payload["max_completion_tokens"] = output_len
    if ignore_eos:
        prepared_payload["ignore_eos"] = True
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
