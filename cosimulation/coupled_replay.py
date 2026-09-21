#!/usr/bin/env python3
"""Replay compute/issue events and post-cache traffic as one HBFSim DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_HBFSIM = REPO_ROOT / "third_party" / "hbfsim"
sys.dont_write_bytecode = True
sys.path.insert(0, str(VENDORED_HBFSIM))

from hbfsim_client import (  # noqa: E402
    ResolvedSystemConfig,
    SimulationSession,
    Transaction,
)


STATUS = "PASS_HBFSIM_CAUSAL_COSIM_DIAGNOSTIC"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nonnegative_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{description} must be finite and non-negative")
    return result


def nonnegative_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be an integer >= 0")
    return value


def safe_identifier(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value or any(
        not (character.isalnum() or character in "_-.:/") for character in value
    ):
        raise ValueError(f"{description} is not protocol-safe: {value!r}")
    return value


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number} is not JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            required = {
                "event_id", "phase", "kernel_id", "cta_id", "sm", "warp", "ready_ns",
                "compute_ns", "wait_for", "memory",
            }
            if set(row) != required:
                raise ValueError(
                    f"line {line_number} fields differ; expected {sorted(required)}"
                )
            event_id = safe_identifier(row["event_id"], f"line {line_number} event_id")
            if event_id in seen:
                raise ValueError(f"line {line_number} repeats event_id {event_id}")
            phase = row["phase"]
            if not isinstance(phase, str) or not phase:
                raise ValueError(f"line {line_number} phase must be non-empty")
            kernel_id = nonnegative_integer(row["kernel_id"], f"line {line_number} kernel_id")
            cta_id = safe_identifier(row["cta_id"], f"line {line_number} cta_id")
            sm = nonnegative_integer(row["sm"], f"line {line_number} sm")
            warp = nonnegative_integer(row["warp"], f"line {line_number} warp")
            ready_ns = nonnegative_number(row["ready_ns"], f"line {line_number} ready_ns")
            compute_ns = nonnegative_number(row["compute_ns"], f"line {line_number} compute_ns")

            raw_wait = row["wait_for"]
            if not isinstance(raw_wait, list):
                raise ValueError(f"line {line_number} wait_for must be a list")
            wait_for = tuple(
                safe_identifier(value, f"line {line_number} wait_for entry")
                for value in raw_wait
            )
            if len(wait_for) != len(set(wait_for)):
                raise ValueError(f"line {line_number} repeats a wait_for event")
            unknown = [value for value in wait_for if value not in seen]
            if unknown:
                raise ValueError(
                    f"line {line_number} wait_for must name prior events: {unknown}"
                )

            raw_memory = row["memory"]
            if not isinstance(raw_memory, list):
                raise ValueError(f"line {line_number} memory must be a list")
            if not raw_memory:
                raise ValueError(
                    f"line {line_number} has no memory transaction; this runner "
                    "models memory-issue events, not compute-only events"
                )
            memory: list[dict[str, Any]] = []
            for item_index, item in enumerate(raw_memory):
                if not isinstance(item, dict) or set(item) != {"op", "addr", "bytes"}:
                    raise ValueError(
                        f"line {line_number} memory {item_index} must have op/addr/bytes"
                    )
                op = item["op"]
                if op not in {"R", "W"}:
                    raise ValueError(f"line {line_number} memory {item_index} op is invalid")
                address = nonnegative_integer(
                    item["addr"], f"line {line_number} memory {item_index} addr"
                )
                byte_count = nonnegative_integer(
                    item["bytes"], f"line {line_number} memory {item_index} bytes"
                )
                if byte_count == 0 or address % 32 or byte_count % 32:
                    raise ValueError(
                        f"line {line_number} memory {item_index} must use positive "
                        "32-byte-aligned sector traffic"
                    )
                memory.append({"op": op, "addr": address, "bytes": byte_count})

            events.append({
                "event_id": event_id,
                "phase": phase,
                "kernel_id": kernel_id,
                "cta_id": cta_id,
                "sm": sm,
                "warp": warp,
                "ready_ns": ready_ns,
                "compute_ns": compute_ns,
                "wait_for": wait_for,
                "memory": tuple(memory),
            })
            seen.add(event_id)
    if not events:
        raise ValueError("event stream is empty")
    return events


def verify_backend_pin(simulator: Path, configs: list[Path], pin_path: Path) -> dict[str, Any]:
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    if pin.get("schema") != "memgen.hbfsim_backend_pin.v1":
        raise ValueError("unsupported HBFSim backend pin")
    actual_binary = sha256(simulator)
    expected_binary = pin["external_simulator"]["sha256"]
    if actual_binary != expected_binary:
        raise ValueError(f"HBFSim binary SHA-256 differs: {actual_binary}")
    expected_configs = pin["system_config_sha256_by_basename"]
    seen_names: set[str] = set()
    for path in configs:
        name = path.name
        if name in seen_names or name not in expected_configs:
            raise ValueError(f"unrecognized or repeated pinned config: {name}")
        seen_names.add(name)
        actual = sha256(path)
        if actual != expected_configs[name]:
            raise ValueError(f"HBFSim config SHA-256 differs for {name}: {actual}")
    missing_required = sorted(set(pin["required_configs"]) - seen_names)
    if missing_required:
        raise ValueError(f"required pinned configs were not supplied: {missing_required}")
    return pin


def lower(events: list[dict[str, Any]]) -> tuple[list[Transaction], list[dict[str, Any]]]:
    transactions: list[Transaction] = []
    terminals: dict[str, tuple[str, ...]] = {}
    metadata: list[dict[str, Any]] = []
    for event in events:
        event_id = event["event_id"]
        compute_id = f"event.{event_id}.compute"
        dependencies = tuple(
            identifier
            for parent in event["wait_for"]
            for identifier in terminals[parent]
        )
        transactions.append(Transaction(
            id=compute_id,
            target="BARRIER",
            op=None,
            addr=0,
            bytes=0,
            issue_ns=event["ready_ns"],
            duration_ns=event["compute_ns"],
            dependencies=dependencies,
        ))
        memory_ids: list[str] = []
        for index, memory in enumerate(event["memory"]):
            memory_id = f"event.{event_id}.memory.{index}"
            transactions.append(Transaction(
                id=memory_id,
                target="HBM",
                op=memory["op"],
                addr=memory["addr"],
                bytes=memory["bytes"],
                issue_ns=event["ready_ns"] + event["compute_ns"],
                dependencies=(compute_id,),
            ))
            memory_ids.append(memory_id)
        terminals[event_id] = tuple(memory_ids)
        metadata.append({
            **event,
            "compute_transaction_id": compute_id,
            "memory_transaction_ids": tuple(memory_ids),
            "terminal_transaction_ids": terminals[event_id],
        })
    return transactions, metadata


def rate(byte_count: int, elapsed_ns: float) -> float | None:
    return None if elapsed_ns <= 0 else byte_count / elapsed_ns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--simulator", required=True, type=Path)
    parser.add_argument("--system-config", required=True, action="append", type=Path)
    parser.add_argument(
        "--backend-pin",
        type=Path,
        default=REPO_ROOT / "cosimulation" / "config" / "backend-pin.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--read-timeout-s", type=float, default=30.0)
    args = parser.parse_args()

    simulator = args.simulator.resolve(strict=True)
    configs = [path.resolve(strict=True) for path in args.system_config]
    pin = verify_backend_pin(simulator, configs, args.backend_pin.resolve(strict=True))
    events = load_events(args.events)
    transactions, metadata = lower(events)

    started = time.monotonic()
    session = SimulationSession(
        simulator_path=simulator,
        system_config=ResolvedSystemConfig.load(configs),
        enable_hbm=True,
        enable_hbf=False,
        enable_external=False,
        read_timeout_s=args.read_timeout_s,
    )
    try:
        batch = session.run(transactions, completions=True)
    finally:
        session.close()
    cpu_wall_seconds = time.monotonic() - started
    source_receipt = session.source_receipt()
    engine_source = source_receipt["engine_source"]
    expected_engine = pin["external_simulator"]
    if (
        engine_source.get("git_commit") != expected_engine["source_revision"]
        or engine_source.get("git_dirty") is not False
        or engine_source.get("tree_hash") != expected_engine["engine_tree_hash"]
        or engine_source.get("source_sha256")
        != expected_engine["engine_source_sha256"]
    ):
        raise RuntimeError("HBFSim runtime source receipt differs from the clean backend pin")
    completions = {record.id: record for record in batch.completions}
    expected_completion_ids = {
        transaction.id for transaction in transactions if transaction.target != "BARRIER"
    }
    if set(completions) != expected_completion_ids:
        raise RuntimeError("HBFSim completion coverage differs from the lowered DAG")

    event_results: list[dict[str, Any]] = []
    memory_records = []
    traffic = Counter()
    phase_traffic: dict[str, Counter[str]] = {}
    for event in metadata:
        memories = [completions[value] for value in event["memory_transaction_ids"]]
        memory_records.extend(memories)
        compute_finish = min(record.arrival_ns for record in memories)
        compute_start = compute_finish - event["compute_ns"]
        event_finish = max(record.finish_ns for record in memories)
        causal_stall = compute_start - event["ready_ns"]
        if causal_stall < -1e-9:
            raise RuntimeError(f"event {event['event_id']} started before ready_ns")
        per_phase = phase_traffic.setdefault(event["phase"], Counter())
        for spec, record in zip(event["memory"], memories, strict=True):
            if record.logical_bytes != spec["bytes"]:
                raise RuntimeError(
                    f"event {event['event_id']} lost logical memory bytes"
                )
            key = "read_bytes" if spec["op"] == "R" else "write_bytes"
            traffic[key] += spec["bytes"]
            traffic["physical_bytes"] += record.physical_bytes
            per_phase[key] += spec["bytes"]
            per_phase["physical_bytes"] += record.physical_bytes
        event_results.append({
            "event_id": event["event_id"],
            "phase": event["phase"],
            "kernel_id": event["kernel_id"],
            "cta_id": event["cta_id"],
            "sm": event["sm"],
            "warp": event["warp"],
            "wait_for": list(event["wait_for"]),
            "nominal_ready_ns": event["ready_ns"],
            "compute_start_ns": compute_start,
            "compute_finish_ns": compute_finish,
            "causal_stall_ns": max(0.0, causal_stall),
            "memory_first_arrival_ns": (
                None if not memories else min(record.arrival_ns for record in memories)
            ),
            "memory_finish_ns": (
                None if not memories else max(record.finish_ns for record in memories)
            ),
            "event_finish_ns": event_finish,
            "memory_transactions": len(memories),
        })

    first_ready = min(event["ready_ns"] for event in metadata)
    final_finish = max(row["event_finish_ns"] for row in event_results)
    closed_loop_span = final_finish - first_ready
    memory_active_span = (
        0.0
        if not memory_records
        else max(record.finish_ns for record in memory_records)
        - min(record.arrival_ns for record in memory_records)
    )
    logical_bytes = traffic["read_bytes"] + traffic["write_bytes"]
    stalls = [row["causal_stall_ns"] for row in event_results]
    result = {
        "status": STATUS,
        "hardware_accuracy_accepted": False,
        "claim_boundary": pin["claim_boundary"],
        "inputs": {
            "events": str(args.events.resolve()),
            "events_sha256": sha256(args.events),
            "simulator": str(simulator),
            "simulator_sha256": sha256(simulator),
            "system_configs": [
                {"path": str(path), "sha256": sha256(path)} for path in configs
            ],
            "backend_pin": str(args.backend_pin.resolve()),
            "backend_pin_sha256": sha256(args.backend_pin),
        },
        "event_count": len(events),
        "transaction_count": len(transactions),
        "memory_transaction_count": len(memory_records),
        "traffic": {
            "read_bytes": traffic["read_bytes"],
            "write_bytes": traffic["write_bytes"],
            "logical_bytes": logical_bytes,
            "physical_bytes": traffic["physical_bytes"],
            "input_to_completion_logical_bytes_conserved": True,
            "by_phase": {phase: dict(counts) for phase, counts in sorted(phase_traffic.items())},
        },
        "timing": {
            "closed_loop_span_ns": closed_loop_span,
            "memory_active_span_ns": memory_active_span,
            "closed_loop_logical_GBps": rate(logical_bytes, closed_loop_span),
            "memory_active_logical_GBps": rate(logical_bytes, memory_active_span),
            "total_causal_stall_ns": sum(stalls),
            "max_causal_stall_ns": max(stalls),
            "stalled_event_count": sum(value > 1e-9 for value in stalls),
            "cpu_wall_seconds": cpu_wall_seconds,
        },
        "events": event_results,
        "hbf_simulation": {
            "batch_id": batch.batch_id,
            "batch_origin_ns": batch.batch_origin_ns,
            "finish_ns": batch.finish_ns,
            "elapsed_ns": batch.elapsed_ns,
            "transaction_latency_by_target": batch.receipt["transaction_latency_by_target"],
            "source_receipt": source_receipt,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": STATUS,
        "events": len(events),
        "transactions": len(transactions),
        "closed_loop_span_ns": closed_loop_span,
        "closed_loop_logical_GBps": result["timing"]["closed_loop_logical_GBps"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
