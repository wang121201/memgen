"""Explicit baseline, detailed-reference, and direct-compact trace options.

These routes do not replace the serving compiler or enable experimental
count-preserving executors. Their input and fidelity contracts remain distinct.
"""

ROUTES = {
    "simple": {
        "role": "original object-range baseline",
        "requires_capture": False,
        "boundary": "semantic object ranges; not hardware post-cache traffic",
    },
    "reference": {
        "role": "capture-driven detailed reference",
        "requires_capture": True,
        "boundary": "generated addresses through a named reference GPU cache",
    },
    "compact": {
        "role": "direct compact production candidate",
        "stability_status": "qualified cases only; no generic full-model or mixed-placement stable release",
        "requires_capture": True,
        "boundary": "legacy decode: logical ranges plus source physical work; structured source: explicit pre-cache patterns and one named cache",
        "native_placement_scope": "legacy decode all-HBF; opt-in structured prefill supports reciprocal placements as unvalidated candidates",
        "calibrated_execution_placements": ["all-hbf"],
        "semantic_only_placements": ["weights-hbm-kv-hbf", "weights-hbf-kv-hbm", "all-hbm"],
        "full_model_option": "--full-model: 7B decode component envelopes with HBM transients; native execution, not full GPU-cache fidelity",
        "structured_source_option": "--structured-source: small source bundle and shape; access structure before continuous cache by default",
        "structured_plan_option": "--structured-plan: opaque objects, ranges/affine/matrices, explicit cache boundary and media",
        "source_schedule_option": "--source-schedule: pinned external p40 source-order descriptor; retained p48 gate/down HBM read streams only, not full-model support",
        "structured_native_requires": "--allow-exploratory and explicit work budgets; API support is not fidelity certification",
    },
}
