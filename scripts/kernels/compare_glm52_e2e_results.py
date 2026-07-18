#!/usr/bin/env python3
"""Compare deterministic native /generate responses from GLM-5.2 E2E runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _response_list(payload: Any) -> list[dict[str, Any]]:
    responses = payload if isinstance(payload, list) else [payload]
    if not responses or not all(isinstance(response, dict) for response in responses):
        raise ValueError("response payload must be a response object or a non-empty list")
    return responses


def _validate_response(
    response: dict[str, Any],
    *,
    expected_topk_width: int,
    label: str,
) -> tuple[list[str], list[int], list[tuple[float, int]], list[list[tuple[float, int]]]]:
    errors: list[str] = []
    if response.get("error") is not None:
        errors.append(f"{label}: response contains error={response['error']!r}")

    output_ids = response.get("output_ids")
    if not isinstance(output_ids, list) or not output_ids:
        errors.append(f"{label}: output_ids must be a non-empty list")
        output_ids = []
    elif not all(isinstance(token, int) for token in output_ids):
        errors.append(f"{label}: output_ids must contain integers")
        output_ids = []

    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        errors.append(f"{label}: meta_info must be an object")
        meta = {}
    completion_tokens = meta.get("completion_tokens")
    if completion_tokens != len(output_ids):
        errors.append(
            f"{label}: completion_tokens={completion_tokens!r} does not match "
            f"output_ids={len(output_ids)}"
        )

    token_logprobs: list[tuple[float, int]] = []
    raw_token_logprobs = meta.get("output_token_logprobs")
    if not isinstance(raw_token_logprobs, list) or len(raw_token_logprobs) != len(output_ids):
        errors.append(f"{label}: expected one output token logprob per output id")
    else:
        for index, row in enumerate(raw_token_logprobs):
            try:
                value = float(row[0])
                token = int(row[1])
            except (IndexError, TypeError, ValueError):
                errors.append(f"{label}: malformed output token logprob at position {index}")
                continue
            if not math.isfinite(value):
                errors.append(f"{label}: non-finite output token logprob at position {index}")
            if token != output_ids[index]:
                errors.append(
                    f"{label}: output token logprob id {token} does not match "
                    f"output id {output_ids[index]} at position {index}"
                )
            token_logprobs.append((value, token))

    top_logprob_rows: list[list[tuple[float, int]]] = []
    raw_top_rows = meta.get("output_top_logprobs")
    if not isinstance(raw_top_rows, list) or len(raw_top_rows) != len(output_ids):
        errors.append(f"{label}: expected one top-logprob row per output id")
    else:
        for row_index, raw_row in enumerate(raw_top_rows):
            if not isinstance(raw_row, list) or len(raw_row) != expected_topk_width:
                errors.append(
                    f"{label}: top-logprob row {row_index} must have width "
                    f"{expected_topk_width}"
                )
                continue
            parsed_row: list[tuple[float, int]] = []
            for item_index, item in enumerate(raw_row):
                try:
                    value = float(item[0])
                    token = int(item[1])
                except (IndexError, TypeError, ValueError):
                    errors.append(
                        f"{label}: malformed top-logprob item {row_index}:{item_index}"
                    )
                    continue
                if not math.isfinite(value):
                    errors.append(
                        f"{label}: non-finite top-logprob value at {row_index}:{item_index}"
                    )
                parsed_row.append((value, token))
            if len({token for _, token in parsed_row}) != len(parsed_row):
                errors.append(f"{label}: duplicate token ids in top-logprob row {row_index}")
            top_logprob_rows.append(parsed_row)

    return errors, output_ids, token_logprobs, top_logprob_rows


def compare_responses(
    candidate: Any,
    baseline: Any,
    *,
    max_logprob_abs_error: float,
    min_topk_overlap: float,
    expected_topk_width: int,
) -> dict[str, Any]:
    """Return a JSON-compatible precision report for one or more responses."""
    candidate_responses = _response_list(candidate)
    baseline_responses = _response_list(baseline)
    response_count_equal = len(candidate_responses) == len(baseline_responses)
    schema_errors: list[str] = []
    candidate_parsed = []
    baseline_parsed = []
    for side, responses, parsed in (
        ("candidate", candidate_responses, candidate_parsed),
        ("baseline", baseline_responses, baseline_parsed),
    ):
        for index, response in enumerate(responses):
            result = _validate_response(
                response,
                expected_topk_width=expected_topk_width,
                label=f"{side}[{index}]",
            )
            schema_errors.extend(result[0])
            parsed.append(result[1:])
    if not response_count_equal:
        schema_errors.append(
            "candidate and baseline response counts differ: "
            f"{len(candidate_responses)} != {len(baseline_responses)}"
        )
    schema_valid = not schema_errors
    finite_output_logprobs = not any("non-finite" in error for error in schema_errors)
    output_ids_equal = schema_valid and all(
        candidate_item[0] == baseline_item[0]
        for candidate_item, baseline_item in zip(candidate_parsed, baseline_parsed, strict=True)
    )

    output_errors: list[float] = []
    overlap_values: list[float] = []
    shared_topk_errors: list[float] = []
    if schema_valid and output_ids_equal:
        for candidate_item, baseline_item in zip(
            candidate_parsed, baseline_parsed, strict=True
        ):
            _, candidate_logprobs, candidate_top_rows = candidate_item
            _, baseline_logprobs, baseline_top_rows = baseline_item
            output_errors.extend(
                abs(candidate_logprob[0] - baseline_logprob[0])
                for candidate_logprob, baseline_logprob in zip(
                    candidate_logprobs, baseline_logprobs, strict=True
                )
            )
            for candidate_row, baseline_row in zip(
                candidate_top_rows, baseline_top_rows, strict=True
            ):
                candidate_by_token = {token: value for value, token in candidate_row}
                baseline_by_token = {token: value for value, token in baseline_row}
                shared_tokens = candidate_by_token.keys() & baseline_by_token.keys()
                overlap_values.append(len(shared_tokens) / expected_topk_width)
                shared_topk_errors.extend(
                    abs(candidate_by_token[token] - baseline_by_token[token])
                    for token in shared_tokens
                )

    maximum_output_error = max(output_errors) if output_errors else None
    minimum_topk_overlap = min(overlap_values) if overlap_values else None
    report = {
        "passed": False,
        "response_count": len(candidate_responses),
        "response_count_equal": response_count_equal,
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "output_ids_equal": output_ids_equal,
        "finite_output_logprobs": finite_output_logprobs,
        "max_output_logprob_abs_error": maximum_output_error,
        "min_topk_overlap": minimum_topk_overlap,
        "topk_rows_compared": len(overlap_values),
        "max_shared_topk_logprob_abs_error": (
            max(shared_topk_errors) if shared_topk_errors else None
        ),
        "thresholds": {
            "max_logprob_abs_error": max_logprob_abs_error,
            "min_topk_overlap": min_topk_overlap,
            "expected_topk_width": expected_topk_width,
        },
    }
    report["passed"] = (
        schema_valid
        and output_ids_equal
        and finite_output_logprobs
        and maximum_output_error is not None
        and minimum_topk_overlap is not None
        and maximum_output_error <= max_logprob_abs_error
        and minimum_topk_overlap >= min_topk_overlap
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--max-logprob-abs-error", type=float, default=0.05)
    parser.add_argument("--min-topk-overlap", type=float, default=0.90)
    parser.add_argument("--expected-topk-width", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare_responses(
        json.loads(args.candidate.read_text(encoding="utf-8")),
        json.loads(args.baseline.read_text(encoding="utf-8")),
        max_logprob_abs_error=args.max_logprob_abs_error,
        min_topk_overlap=args.min_topk_overlap,
        expected_topk_width=args.expected_topk_width,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
