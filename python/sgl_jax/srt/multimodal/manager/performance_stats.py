"""Performance statistics tracking for sglang-jax multimodal pipeline.

This module provides performance tracking comparable to vLLM-Omni's stats system,
enabling end-to-end benchmark comparisons.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageStats:
    """Statistics for a single stage processing a request."""

    stage_id: int
    batch_id: int = 0
    batch_size: int = 1
    num_tokens_in: int = 0
    num_tokens_out: int = 0
    stage_gen_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    audio_generated_frames: int = 0
    video_generated_frames: int = 0


@dataclass
class TransferStats:
    """Statistics for inter-stage data transfer."""

    from_stage: int
    to_stage: int
    size_bytes: int = 0
    tx_time_ms: float = 0.0
    rx_decode_time_ms: float = 0.0
    in_flight_time_ms: float = 0.0

    @property
    def size_kbytes(self) -> float:
        return self.size_bytes / 1024.0


@dataclass
class RequestStats:
    """Complete statistics for a single request through the pipeline."""

    request_id: str
    e2e_start_time: float = field(default_factory=time.perf_counter)
    e2e_end_time: float | None = None
    stage_stats: dict[int, StageStats] = field(default_factory=dict)
    transfer_stats: list[TransferStats] = field(default_factory=list)

    @property
    def e2e_total_ms(self) -> float:
        if self.e2e_end_time is None:
            return 0.0
        return (self.e2e_end_time - self.e2e_start_time) * 1000

    @property
    def e2e_total_tokens(self) -> int:
        return sum(s.num_tokens_out for s in self.stage_stats.values())

    @property
    def transfers_total_kbytes(self) -> float:
        return sum(t.size_kbytes for t in self.transfer_stats)

    @property
    def transfers_total_time_ms(self) -> float:
        return sum(t.tx_time_ms + t.rx_decode_time_ms for t in self.transfer_stats)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "e2e_total_ms": self.e2e_total_ms,
            "e2e_total_tokens": self.e2e_total_tokens,
            "transfers_total_kbytes": self.transfers_total_kbytes,
            "transfers_total_time_ms": self.transfers_total_time_ms,
            "stage_stats": {
                stage_id: {
                    "stage_id": s.stage_id,
                    "batch_id": s.batch_id,
                    "batch_size": s.batch_size,
                    "num_tokens_in": s.num_tokens_in,
                    "num_tokens_out": s.num_tokens_out,
                    "stage_gen_time_ms": s.stage_gen_time_ms,
                    "postprocess_time_ms": s.postprocess_time_ms,
                    "audio_generated_frames": s.audio_generated_frames,
                    "video_generated_frames": s.video_generated_frames,
                }
                for stage_id, s in self.stage_stats.items()
            },
            "transfer_stats": [
                {
                    "from_stage": t.from_stage,
                    "to_stage": t.to_stage,
                    "size_kbytes": t.size_kbytes,
                    "tx_time_ms": t.tx_time_ms,
                    "rx_decode_time_ms": t.rx_decode_time_ms,
                    "in_flight_time_ms": t.in_flight_time_ms,
                }
                for t in self.transfer_stats
            ],
        }


@dataclass
class OverallStats:
    """Overall statistics for all requests."""

    e2e_requests: int = 0
    e2e_wall_time_ms: float = 0.0
    e2e_total_tokens: int = 0
    stage_wall_times_ms: dict[int, float] = field(default_factory=dict)

    @property
    def e2e_avg_time_per_request_ms(self) -> float:
        if self.e2e_requests == 0:
            return 0.0
        return self.e2e_wall_time_ms / self.e2e_requests

    @property
    def e2e_avg_tokens_per_s(self) -> float:
        if self.e2e_wall_time_ms == 0:
            return 0.0
        return self.e2e_total_tokens / (self.e2e_wall_time_ms / 1000.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "e2e_requests": self.e2e_requests,
            "e2e_wall_time_ms": self.e2e_wall_time_ms,
            "e2e_total_tokens": self.e2e_total_tokens,
            "e2e_avg_time_per_request_ms": self.e2e_avg_time_per_request_ms,
            "e2e_avg_tokens_per_s": self.e2e_avg_tokens_per_s,
        }
        for stage_id, wall_time in self.stage_wall_times_ms.items():
            result[f"e2e_stage_{stage_id}_wall_time_ms"] = wall_time
        return result


class PerformanceTracker:
    """Tracks performance statistics for the multimodal pipeline."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.request_stats: dict[str, RequestStats] = {}
        self.overall_start_time: float | None = None
        self.overall_end_time: float | None = None

    def start_tracking(self):
        """Start overall tracking."""
        if not self.enabled:
            return
        self.overall_start_time = time.perf_counter()
        logger.info("Performance tracking started")

    def stop_tracking(self):
        """Stop overall tracking."""
        if not self.enabled:
            return
        self.overall_end_time = time.perf_counter()
        logger.info("Performance tracking stopped")

    def start_request(self, request_id: str) -> RequestStats:
        """Start tracking a new request."""
        if not self.enabled:
            return RequestStats(request_id=request_id)

        stats = RequestStats(request_id=request_id)
        self.request_stats[request_id] = stats
        return stats

    def end_request(self, request_id: str):
        """Mark request as completed."""
        if not self.enabled or request_id not in self.request_stats:
            return

        self.request_stats[request_id].e2e_end_time = time.perf_counter()

    def add_stage_stats(self, request_id: str, stage_stats: StageStats):
        """Add stage statistics for a request."""
        if not self.enabled or request_id not in self.request_stats:
            return

        self.request_stats[request_id].stage_stats[stage_stats.stage_id] = stage_stats

    def add_transfer_stats(self, request_id: str, transfer_stats: TransferStats):
        """Add transfer statistics for a request."""
        if not self.enabled or request_id not in self.request_stats:
            return

        self.request_stats[request_id].transfer_stats.append(transfer_stats)

    def compute_overall_stats(self) -> OverallStats:
        """Compute overall statistics from all requests."""
        if not self.enabled:
            return OverallStats()

        overall = OverallStats()
        overall.e2e_requests = len(self.request_stats)

        if self.overall_start_time and self.overall_end_time:
            overall.e2e_wall_time_ms = (self.overall_end_time - self.overall_start_time) * 1000

        # Aggregate per-request stats
        stage_times: dict[int, list[float]] = {}
        for req_stats in self.request_stats.values():
            overall.e2e_total_tokens += req_stats.e2e_total_tokens

            for stage_id, stage_stat in req_stats.stage_stats.items():
                if stage_id not in stage_times:
                    stage_times[stage_id] = []
                stage_times[stage_id].append(stage_stat.stage_gen_time_ms)

        # Compute stage wall times (max time across all requests in batch)
        for stage_id, times in stage_times.items():
            overall.stage_wall_times_ms[stage_id] = max(times) if times else 0.0

        return overall

    def print_summary(self):
        """Print performance summary to console."""
        if not self.enabled:
            return

        overall = self.compute_overall_stats()

        print("\n" + "=" * 80)
        print("[Overall Summary]")
        print("=" * 80)
        print(f"{'e2e_requests':<35} {overall.e2e_requests:>12}")
        print(f"{'e2e_wall_time_ms':<35} {overall.e2e_wall_time_ms:>12,.3f}")
        print(f"{'e2e_total_tokens':<35} {overall.e2e_total_tokens:>12,}")
        print(f"{'e2e_avg_time_per_request_ms':<35} {overall.e2e_avg_time_per_request_ms:>12,.3f}")
        print(f"{'e2e_avg_tokens_per_s':<35} {overall.e2e_avg_tokens_per_s:>12,.2f}")

        for stage_id in sorted(overall.stage_wall_times_ms.keys()):
            wall_time = overall.stage_wall_times_ms[stage_id]
            print(f"{'e2e_stage_' + str(stage_id) + '_wall_time_ms':<35} {wall_time:>12,.3f}")

        print("=" * 80)

        # Print per-request details
        for req_id, req_stats in self.request_stats.items():
            print(f"\n[RequestE2EStats [request_id={req_id}]]")
            print("-" * 80)
            print(f"{'e2e_total_ms':<35} {req_stats.e2e_total_ms:>12,.3f}")
            print(f"{'e2e_total_tokens':<35} {req_stats.e2e_total_tokens:>12,}")
            print(f"{'transfers_total_kbytes':<35} {req_stats.transfers_total_kbytes:>12,.3f}")
            print(f"{'transfers_total_time_ms':<35} {req_stats.transfers_total_time_ms:>12,.3f}")

    def save_to_file(self, output_dir: str | Path):
        """Save statistics to JSON files."""
        if not self.enabled:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save overall stats
        overall = self.compute_overall_stats()
        overall_path = output_dir / "overall_stats.json"
        with open(overall_path, "w") as f:
            json.dump(overall.to_dict(), f, indent=2)
        logger.info("Saved overall stats to %s", overall_path)

        # Save per-request stats
        requests_path = output_dir / "request_stats.json"
        with open(requests_path, "w") as f:
            json.dump(
                {rid: stats.to_dict() for rid, stats in self.request_stats.items()},
                f,
                indent=2,
            )
        logger.info("Saved request stats to %s", requests_path)

        # Save JSONL format for compatibility with vLLM-Omni
        overall_jsonl = output_dir / "overall.stats.jsonl"
        with open(overall_jsonl, "w") as f:
            f.write(json.dumps(overall.to_dict()) + "\n")

        requests_jsonl = output_dir / "requests.stats.jsonl"
        with open(requests_jsonl, "w") as f:
            for req_stats in self.request_stats.values():
                f.write(json.dumps(req_stats.to_dict()) + "\n")

        logger.info("Saved stats to %s", output_dir)
