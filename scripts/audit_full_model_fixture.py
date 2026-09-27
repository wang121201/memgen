#!/usr/bin/env python3
"""Fail-closed audit for a native full-model replay fixture."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


KERNEL_PHASE = re.compile(r"^-kernel_([0-9]+)_llama_phase\s+\S+\s*$")
ADMITTED = "PASS_ALL_KERNEL_PROFILES_ADMITTED_NOT_CACHE_REPLAYED"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def audit(root: Path) -> dict[str, Any]:
    reasons: list[str] = []
    finish_path = root / "finish.json"
    manifest_path = root / "manifest.json"
    index_path = root / "profiles.index.jsonl"
    app_path = root / "app.config"
    issue_path = root / "issue.config"
    result: dict[str, Any] = {
        "schema": "memgen.full_model_fixture_audit_v1",
        "fixture": str(root.resolve()),
        "accepted": False,
        "status": "STARTED",
        "blocking_reasons": reasons,
    }

    if finish_path.is_file():
        finish = load(finish_path)
        result["source_status"] = finish.get("status")
        result["claim_boundary"] = finish.get("claim_boundary")
        if "prefix" in str(finish.get("schema", "")).lower():
            reasons.append("fixture schema is a prefix view")
        if "not a full inference result" in str(finish.get("claim_boundary", "")).lower():
            reasons.append("fixture claim boundary rejects full inference")

    for path in (manifest_path, index_path, app_path, issue_path):
        if not path.is_file():
            reasons.append(f"missing {path.name}")

    if manifest_path.is_file():
        manifest = load(manifest_path)
        result["profile_status"] = manifest.get("status")
        if "complete_full_model" in manifest:
            result["fixture_kind"] = "expanded"
            required = {
                "complete_full_model": True,
                "fully_exact": True,
                "full_native_address_coverage": True,
                "modeled_launches": 0,
                "unsupported_launches": 0,
                "unknown_private_allocations": 0,
            }
            for key, expected in required.items():
                actual = manifest.get(key)
                if actual != expected:
                    reasons.append(f"expanded manifest {key} is {actual!r}, expected {expected!r}")
            if manifest.get("hardware_accuracy_accepted") is True:
                reasons.append("fixture incorrectly claims hardware accuracy")
        else:
            result["fixture_kind"] = "profile"
            if manifest.get("status") != ADMITTED:
                reasons.append(f"profile manifest status is {manifest.get('status')!r}")
            if manifest.get("modeled_launches", 0):
                reasons.append("profile manifest contains modeled launches")
            if manifest.get("unsupported_launches", 0):
                reasons.append("profile manifest contains unsupported launches")

    if index_path.is_file():
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result["profile_count"] = len(rows)
        if [int(row.get("kernel_id", -1)) for row in rows] != list(range(1, len(rows) + 1)):
            reasons.append("profile index is not dense")
        non_native = [row.get("status") for row in rows if not str(row.get("status", "")).startswith("PASS_")]
        if non_native:
            reasons.append("profile index contains non-PASS records")

    if app_path.is_file():
        kernel_ids = {
            int(match.group(1))
            for line in app_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if (match := KERNEL_PHASE.match(line))
        }
        result["kernel_count"] = len(kernel_ids)
        if index_path.is_file() and len(kernel_ids) != result.get("profile_count"):
            reasons.append("app.config and profile index kernel counts differ")

    result["status"] = "PASS_NATIVE_FULL_MODEL_FIXTURE" if not reasons else "BLOCKED_NOT_NATIVE_FULL_MODEL"
    result["accepted"] = not reasons
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.fixture.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())