#!/usr/bin/env python3
"""Create a deterministic two-kernel fixture for the streaming cache bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(
    ordinal: int,
    opcode: str,
    delta: int,
    rule: dict[str, object],
    *,
    mask: str = "ffffffff",
    pair: str = "4:31",
    exact_delta: dict[str, int] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "ordinal": ordinal,
        "pc": f"{0x10 + ordinal * 0x10:x}",
        "opcode": opcode,
        "mask": mask,
        "sampled_timestamp_delta": delta,
        "groups": [{"pairs": [pair]}],
        "address_rules": [rule],
    }
    if exact_delta is not None:
        result["sampled_timestamp_delta_by_cta"] = exact_delta
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = output / "configs"
    configs.mkdir()

    exact_bases = {
        "kind": "exact_cta_base_table",
        "bases_by_cta": {"0": 0x1000, "2": 0x1100},
    }
    profile1 = {
        "schema": {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 9},
        "status": "PASS_DETERMINISTIC_STREAM_BRIDGE_SMOKE",
        "kernel": {"id": 1, "name": "smoke_structural", "grid_size": 3},
        "structural_classes": [
            {
                "class_id": "active",
                "template": [
                    entry(
                        0,
                        "LDG.E.32",
                        5,
                        exact_bases,
                        exact_delta={"0": 7, "2": 3},
                    ),
                    entry(
                        1,
                        "STG.E.32",
                        10,
                        exact_bases,
                        exact_delta={"0": 2, "2": 8},
                    ),
                ],
            },
            {"class_id": "empty", "template": []},
        ],
        "cta_class_by_id": ["active", "empty", "active"],
    }
    profile2 = {
        "schema": {"name": "hbserve.hyfiss_sampled_sass_profile", "version": 7},
        "status": "PASS_DETERMINISTIC_STREAM_BRIDGE_SMOKE",
        "kernel": {"id": 2, "name": "smoke_affine", "grid_size": 2},
        "template": [
            entry(
                0,
                "LDG.E.64",
                1,
                {
                    "intercept": 0x2000,
                    "cta_x_stride": 128,
                    "cta_y_stride": 0,
                    "cta_z_stride": 0,
                },
                mask="0000000f",
                # Four active lanes walk backwards across distinct sectors.
                # The compact path must reproduce the legacy coalescer's
                # (reference id, sector address) sort, not lane encounter order.
                pair="-32:31",
            )
        ],
    }
    app = """\
-kernel_1_kernel_name smoke_structural
-kernel_1_llama_phase prefill
-kernel_1_grid_size 3
-kernel_1_block_size 32
-kernel_1_grid_dim_x 3
-kernel_1_grid_dim_y 1
-kernel_1_grid_dim_z 1
-kernel_2_kernel_name smoke_affine
-kernel_2_llama_phase decode_step_0
-kernel_2_grid_size 2
-kernel_2_block_size 32
-kernel_2_grid_dim_x 2
-kernel_2_grid_dim_y 1
-kernel_2_grid_dim_z 1
"""
    (configs / "app.config").write_text(app, encoding="utf-8")
    issue = """\
-trace_issued_sm_id_0 (1,0,100) (1,1,200) (2,0,200)
-trace_issued_sm_id_1 (1,2,80) (2,1,180)
"""
    (configs / "issue.config").write_text(issue, encoding="utf-8")
    (output / "objects.tsv").write_text("0 0 4294967296\n", encoding="utf-8")

    pack_path = output / "profiles.pack.jsonl"
    index_path = output / "profiles.index.jsonl"
    with pack_path.open("wb") as pack, index_path.open("w", encoding="utf-8") as stream:
        for kernel, profile in enumerate((profile1, profile2), 1):
            encoded = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            offset = pack.tell()
            pack.write(encoded)
            stream.write(
                json.dumps(
                    {
                        "kernel_id": kernel,
                        "path": str(pack_path),
                        "offset": offset,
                        "bytes": len(encoded),
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "status": "PASS_DETERMINISTIC_STREAM_BRIDGE_SMOKE",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    write_json(output / "manifest.json", {
        "schema": "hbserve_full_inference_profile_census_v1",
        "status": "PASS_ALL_KERNEL_PROFILES_ADMITTED_NOT_CACHE_REPLAYED",
        "workload_id": "smoke",
        "kernel_count": 2,
        "completed_kernels": 2,
        "inputs": {
            "app_config": {
                "path": str(configs / "app.config"),
                "bytes": (configs / "app.config").stat().st_size,
                "sha256": sha256(configs / "app.config"),
            },
            "issue_config": {
                "path": str(configs / "issue.config"),
                "bytes": (configs / "issue.config").stat().st_size,
                "sha256": sha256(configs / "issue.config"),
            },
        },
        "profile_pack": {
            "path": str(pack_path),
            "bytes": pack_path.stat().st_size,
            "sha256": sha256(pack_path),
            "records": 2,
            "format": "one compact JSON profile plus newline per indexed byte range",
        },
        "profile_index": {
            "path": str(index_path),
            "bytes": index_path.stat().st_size,
            "sha256": sha256(index_path),
        },
        "claim_boundary": "deterministic stream-cache bridge smoke fixture only",
    })

    expected = {
        "generated_memory_instructions": 6,
        "generated_lane_addresses": 136,
        "sector_requests": 24,
        "read_sector_requests": 16,
        "write_sector_requests": 8,
        "compact_bytes": 288,
        "compact_sequence": [
            *[[address, 0, 1] for address in range(0x1100, 0x1180, 0x20)],
            *[[address, 1, 1] for address in range(0x1100, 0x1180, 0x20)],
            *[[address, 1, 1] for address in range(0x1000, 0x1080, 0x20)],
            *[[address, 0, 1] for address in range(0x1000, 0x1080, 0x20)],
            *[[address, 0, 2] for address in range(0x2020, 0x20A0, 0x20)],
            *[[address, 0, 2] for address in range(0x1FA0, 0x2020, 0x20)],
        ],
    }
    write_json(output / "expected.json", expected)
    print(json.dumps({"status": "PASS_FIXTURE_CREATED", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
