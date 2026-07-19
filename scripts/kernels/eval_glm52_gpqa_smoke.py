#!/usr/bin/env python3
"""Run a reproducible GPQA-Diamond smoke eval against an OpenAI-compatible API.

This intentionally matches the sampling, option permutation, prompt, and direct
answer extraction in zai-org/glm-simple-evals.  It is a small regression smoke,
not a replacement for the official eight-repeat full-dataset evaluation.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Iterable


QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*([A-D])", re.IGNORECASE)
DATASET_SEED = 0


class RequestCompletionError(RuntimeError):
    """Request failure annotated with the number of attempts made."""

    def __init__(self, cause: Exception, *, attempts: int):
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts


def prepare_examples(
    rows: Iterable[dict[str, str]], *, sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Apply the official Random(seed) subset and answer permutation."""
    examples = list(rows)
    if not 0 < sample_size <= len(examples):
        raise ValueError(f"sample_size must be in [1, {len(examples)}]")
    rng = random.Random(seed)
    examples = rng.sample(examples, sample_size)
    prepared = []
    for row in examples:
        permutation = rng.sample(range(4), 4)
        original_choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        choices = [original_choices[index] for index in permutation]
        correct_answer = "ABCD"[choices.index(row["Correct Answer"])]
        prompt = QUERY_TEMPLATE.format(
            Question=row["Question"],
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
        prepared.append(
            {
                "row": row,
                "permutation": permutation,
                "correct_answer": correct_answer,
                "prompt": prompt,
            }
        )
    return prepared


def prepared_example_identity(index: int, example: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "record_id": example["row"].get("Record ID"),
        "domain": example["row"].get("High-level domain"),
        "prompt": example["prompt"],
        "permutation": example["permutation"],
        "correct_answer": example["correct_answer"],
    }


def prepared_examples_fingerprint(examples: list[dict[str, Any]]) -> str:
    identities = [
        prepared_example_identity(index, example)
        for index, example in enumerate(examples)
    ]
    canonical = json.dumps(
        identities, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def extract_answer(response: str) -> str | None:
    """Match the official direct-regex path over the last 1024 characters."""
    match = ANSWER_PATTERN.search(response[-1024:])
    return match.group(1) if match else None


def build_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def wilson_interval(*, correct: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval as fractions."""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("correct and total must satisfy 0 <= correct <= total")
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - margin, center + margin


def request_completion(
    *, api_base: str, api_key: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], int]:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                try:
                    return json.load(response), attempt
                except json.JSONDecodeError as error:
                    raise RequestCompletionError(error, attempts=attempt) from error
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if retryable and attempt < 3:
                time.sleep(attempt)
                continue
            raise RequestCompletionError(error, attempts=attempt) from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < 3:
                time.sleep(attempt)
                continue
            raise RequestCompletionError(error, attempts=attempt) from error
    raise AssertionError("request retry loop exited unexpectedly")


def evaluate_one(
    index: int,
    example: dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    payload = build_payload(
        model=model,
        prompt=example["prompt"],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed + index,
    )
    started = time.perf_counter()
    base_result = {
        "index": index,
        "record_id": example["row"].get("Record ID"),
        "domain": example["row"].get("High-level domain"),
        "prompt": example["prompt"],
        "permutation": example["permutation"],
        "correct_answer": example["correct_answer"],
    }
    attempts = 0
    try:
        response, attempts = request_completion(
            api_base=api_base, api_key=api_key, payload=payload, timeout=timeout
        )
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        extracted_answer = extract_answer(content)
        return base_result | {
            "extracted_answer": extracted_answer,
            "score": int(extracted_answer == example["correct_answer"]),
            "format_failure": extracted_answer is None,
            "content": content,
            "reasoning_content": message.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage") or {},
            "latency_seconds": time.perf_counter() - started,
            "request_attempts": attempts,
            "request_error": None,
        }
    except Exception as error:
        cause = error
        if isinstance(error, RequestCompletionError):
            attempts = error.attempts
            cause = error.cause
        return base_result | {
            "extracted_answer": None,
            "score": 0,
            "format_failure": True,
            "content": "",
            "reasoning_content": "",
            "finish_reason": None,
            "usage": {},
            "latency_seconds": time.perf_counter() - started,
            "request_attempts": attempts,
            "request_error": {
                "type": type(cause).__name__,
                "message": str(cause),
            },
        }


def summarize_results(
    results: list[dict[str, Any]], *, expected_total: int, wall_seconds: float
) -> dict[str, Any]:
    """Report official-style scoring plus truncation-aware diagnostics."""
    total = len(results)
    scored_results = [
        result for result in results if result.get("request_error") is None
    ]
    scored = len(scored_results)
    correct = sum(result["score"] for result in scored_results)
    completed_results = [
        result for result in scored_results if result.get("finish_reason") == "stop"
    ]
    completed = len(completed_results)
    completed_correct = sum(result["score"] for result in completed_results)

    if scored:
        lower, upper = wilson_interval(correct=correct, total=scored)
        official_accuracy = 100 * correct / scored
        official_wilson: list[float] | None = [100 * lower, 100 * upper]
        median_request_seconds: float | None = statistics.median(
            result["latency_seconds"] for result in results
        )
    else:
        official_accuracy = None
        official_wilson = None
        median_request_seconds = (
            statistics.median(result["latency_seconds"] for result in results)
            if results
            else None
        )

    if completed:
        completed_lower, completed_upper = wilson_interval(
            correct=completed_correct, total=completed
        )
        completed_accuracy: float | None = 100 * completed_correct / completed
        completed_wilson: list[float] | None = [
            100 * completed_lower,
            100 * completed_upper,
        ]
    else:
        completed_accuracy = None
        completed_wilson = None

    incomplete_finish_reasons = collections.Counter(
        str(result.get("finish_reason") or "unknown")
        for result in scored_results
        if result.get("finish_reason") not in {"stop", "length"}
    )
    return {
        "correct": correct,
        "total": total,
        "scored": scored,
        "expected_total": expected_total,
        "remaining": expected_total - total,
        "accuracy_percent": official_accuracy,
        "wilson_95_percent": official_wilson,
        "official_style_accuracy_percent": official_accuracy,
        "official_style_wilson_95_percent": official_wilson,
        "format_failures": sum(result["format_failure"] for result in results),
        "length_truncations": sum(
            result.get("finish_reason") == "length" for result in results
        ),
        "request_errors": sum(
            result.get("request_error") is not None for result in results
        ),
        "completed": completed,
        "completed_correct": completed_correct,
        "completed_only_accuracy_percent": completed_accuracy,
        "completed_only_wilson_95_percent": completed_wilson,
        "incomplete_finish_reasons": dict(sorted(incomplete_finish_reasons.items())),
        "wall_seconds": wall_seconds,
        "median_request_seconds": median_request_seconds,
    }


def write_report_atomic(output: pathlib.Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to gpqa_diamond.csv")
    parser.add_argument("--output", required=True, help="JSON result path")
    parser.add_argument("--api-base", default="http://127.0.0.1:30280/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="/models/GLM-5.2")
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--request-seed",
        "--seed",
        dest="request_seed",
        type=int,
        default=0,
        help="Per-request sampling seed; --seed is retained as an alias.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful/truncated records from an existing output checkpoint.",
    )
    return parser.parse_args()


def build_report(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    *,
    wall_seconds: float,
    examples_fingerprint: str,
) -> dict[str, Any]:
    results = sorted(results, key=lambda result: result["index"])
    summary = summarize_results(
        results,
        expected_total=args.sample_size,
        wall_seconds=wall_seconds,
    )
    incomplete_count = sum(summary["incomplete_finish_reasons"].values())
    if len(results) < args.sample_size:
        status = "partial"
    elif summary["scored"] == 0 or (
        summary["completed"] == 0 and incomplete_count
    ):
        status = "failed"
    elif summary["request_errors"] or incomplete_count:
        status = "complete_with_errors"
    else:
        status = "complete"
    return {
        "schema_version": 1,
        "benchmark": "GPQA-Diamond smoke",
        "status": status,
        "prepared_examples_sha256": examples_fingerprint,
        "official_harness_semantics": {
            "subset_seed": DATASET_SEED,
            "n_repeats": 1,
            "direct_answer_regex": ANSWER_PATTERN.pattern,
            "direct_answer_regex_flags": ["IGNORECASE"],
            "response_tail_chars": 1024,
        },
        "request": {
            "api_base": args.api_base,
            "model": args.model,
            "sample_size": args.sample_size,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "request_seed": args.request_seed,
            "enable_thinking": True,
        },
        "summary": summary,
        "results": results,
    }


def report_has_operational_errors(report: dict[str, Any]) -> bool:
    summary = report["summary"]
    return bool(
        summary["request_errors"]
        or sum(summary["incomplete_finish_reasons"].values())
    )


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
) -> None:
    fingerprint = prepared_examples_fingerprint(examples)
    expected = build_report(
        args,
        [],
        wall_seconds=0.0,
        examples_fingerprint=fingerprint,
    )
    if checkpoint.get("schema_version") != expected["schema_version"]:
        raise ValueError("resume checkpoint schema version does not match")
    if checkpoint.get("official_harness_semantics") != expected[
        "official_harness_semantics"
    ]:
        raise ValueError("resume checkpoint harness semantics do not match")
    if checkpoint.get("request") != expected["request"]:
        raise ValueError("resume checkpoint request settings do not match")
    if checkpoint.get("prepared_examples_sha256") != fingerprint:
        raise ValueError("resume checkpoint prepared examples do not match")

    seen_indices: set[int] = set()
    for result in checkpoint.get("results", []):
        index = result.get("index")
        if not isinstance(index, int) or not 0 <= index < len(examples):
            raise ValueError(f"resume checkpoint result index out of range: {index!r}")
        if index in seen_indices:
            raise ValueError(f"resume checkpoint has duplicate result index: {index}")
        seen_indices.add(index)
        expected_identity = prepared_example_identity(index, examples[index])
        actual_identity = {
            key: result.get(key) for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                f"resume checkpoint result identity mismatch at index {index}"
            )


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise ValueError("concurrency, max-tokens, and timeout must be positive")
    with open(args.data, newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    examples = prepare_examples(
        rows, sample_size=args.sample_size, seed=DATASET_SEED
    )
    examples_fingerprint = prepared_examples_fingerprint(examples)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    output = pathlib.Path(args.output)
    previous_wall_seconds = 0.0
    if args.resume and output.exists():
        checkpoint = json.loads(output.read_text(encoding="utf-8"))
        validate_resume_checkpoint(checkpoint, args, examples)
        previous_wall_seconds = checkpoint.get("summary", {}).get("wall_seconds", 0.0)
        results = [
            result
            for result in checkpoint.get("results", [])
            if result.get("request_error") is None
        ]
    finished_indices = {result["index"] for result in results}
    pending = [
        (index, example)
        for index, example in enumerate(examples)
        if index not in finished_indices
    ]

    write_report_atomic(
        output,
        build_report(
            args,
            results,
            wall_seconds=previous_wall_seconds,
            examples_fingerprint=examples_fingerprint,
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                index,
                example,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.request_seed,
                timeout=args.timeout,
            ): index
            for index, example in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda item: item["index"])
            print(
                f"[{len(results)}/{len(examples)}] index={result['index']} "
                f"answer={result['extracted_answer']} expected={result['correct_answer']} "
                f"score={result['score']} error={result['request_error']} "
                f"latency={result['latency_seconds']:.1f}s",
                flush=True,
            )
            elapsed = previous_wall_seconds + time.perf_counter() - started
            write_report_atomic(
                output,
                build_report(
                    args,
                    results,
                    wall_seconds=elapsed,
                    examples_fingerprint=examples_fingerprint,
                ),
            )

    wall_seconds = previous_wall_seconds + time.perf_counter() - started
    report = build_report(
        args,
        results,
        wall_seconds=wall_seconds,
        examples_fingerprint=examples_fingerprint,
    )
    write_report_atomic(output, report)
    print(
        json.dumps(report["summary"], indent=2),
        flush=True,
    )
    if report_has_operational_errors(report):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
