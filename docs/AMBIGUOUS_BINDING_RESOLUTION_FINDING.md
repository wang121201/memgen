# Why layer rebinding refused, and the view it was attributed to

Status: `MEASURED_2026-09-24_ON_THIS_HOST`. This is a finding about the
layer-to-layer rebinding step, measured on the archived P32D2 run with no new GPU
work. It changes no admission: `hardware_accuracy_accepted` stays `false`
everywhere it appears, and `validation/p32d2_branch_status.csv` remains the
authority.

## 1. What was refused

`expand_profiles.py` copies one sampled layer's packed profile onto every other
layer of the model. For each address rule it translates the rule by the
difference between the source tensor view and the target tensor view, so the
target launch addresses the model's own objects rather than the sampled layer's.

The archived Qwen2.5-1.5B P32D2 run (`out/collect-20260923T165342Z`) packs 1360 of
2060 target launches. 700 were refused, in two equal halves:

| Count | Reason as recorded | Where it comes from |
| ---: | --- | --- |
| 350 | `Source profile missing/unsupported` | the class never produced a packed profile (sampler rule fit) |
| 350 | `Ambiguous observed tensor binding: conflicting target deltas` | the profile exists, the rebinding refuses to choose |

This document is about the second half.

## 2. Why it was ambiguous

`Binder.contexts()` walks the sampled launch's call, then its parent call, then
the parent's parent, and so on. Every input and output view of every ancestor is
collected under a key whose layer index is neutralised, so
`model.layers.3.post_attention_layernorm.inputs.args[1]` and
`model.layers.3.inputs.args[1]` are *different* keys but a single ancestor can
inherit the very buffer a child uses.

`Binder.mapping()` then pairs source views with target views that have the same
layout, and `rebind()` requires every rule to be contained in the candidate
views of exactly one delta. When an inherited ancestor view and the kernel's own
operand view are both contained, and their counterparts in the target layer sit
at different addresses, the union holds two deltas and the target is refused.

Measured on the run, 162 of the 350 ambiguous launches are
`flashinfer::norm::FusedAddRMSNormKernel`, and one of its rules shows the shape
of the problem exactly:

```text
pc=0xc60 LDG.E.128 span=[139954157926400, 139954158022144] bytes=95744
  cand kind=activation size=98304 off=0 delta=0           ('model.layers.<L>.post_attention_layernorm','inputs','args[1]')
  cand kind=activation size=98304 off=0 delta=0           ('model.layers.<L>.post_attention_layernorm','outputs','output[1]')
  cand kind=activation size=98304 off=0 delta=2084352     ('model.layers.<L>','inputs','args[1]')
  cand kind=activation size=98304 off=0 delta=0           ('model.layers.<L>','outputs','output[1]')
```

Three views agree on `delta=0`; one ancestor view disagrees. The rule is not
ambiguous at the kernel; it is ambiguous only when the kernel's operand and an
ancestor's inherited buffer are pooled.

## 3. The rule now applied

When the containing views disagree, keep only the views reached in the fewest
call hops, and require their delta to be unique. Hop 0 is the launch's own call,
which is where its operands are; higher hops are ancestor modules that inherited
and reused buffers. If the innermost views still disagree, the rule is refused as
before — nothing is guessed, and the refusal message is unchanged.

The refinement is entered only when the union of containing views already holds
more than one delta, so a rule that was unambiguous before keeps exactly the
delta it had. Each rebinding that used the rule records
`ambiguous_binding_resolution` in its binding receipt and in the packed profile's
`model` object, so a reader can see that a translated address came from the
kernel's own operand view.

## 4. What was measured

Prototype over every rule of every cross-layer binding of the P32D2 run:

| Quantity | Value |
| --- | ---: |
| rules unique before | 548,718 |
| rules ambiguous before | 35,544 |
| ambiguous rules resolved by the hop rule | 35,544 (100%) |
| ambiguous rules still refused | 0 |
| rules unique before that the hop rule would have made ambiguous | 0 |
| hop at which the resolution happened | 0 for every one of them |
| resolutions into weights or KV | 0 (all activations) |

Full expansion, control and patched, on the same sample and with no GPU work:

| | control (unpatched) | patched |
| --- | ---: | ---: |
| target launches | 2060 | 2060 |
| packed launches | 1360 | **1710** |
| unsupported launches | 700 | **350** |
| of which ambiguous binding | 350 | **0** |
| coverage | 66.0% | **83.0%** |

Regression check, comparing every binding that the control accepted against the
same binding under the patch:

| | rows |
| --- | ---: |
| identical template | 902 |
| differing only in the synthetic private-arena bases | 428 |
| both rows `numeric_modeled` | 30 |
| **differing in a hardware-bound address** | **0** |

The 428 moved addresses are the `unobserved_private_object` rules, whose bases
come from one run-wide monotone cursor (`Binder.unknown_cursor`). That cursor
advances once per accepted binding, so accepting 350 more bindings shifts the
bases of later ones — a bookkeeping shift in an address range that the manifest
already labels `target-kernel private allocation`, not a change in any address
taken from an observed tensor.

## 5. What was not established

- The rule is validated on this run's own evidence. It is not a claim that a
  target operand is *always* at hop 0; a class whose operand is genuinely
  inherited would now be refused the same way it was before.
- No hardware comparison. The 350 newly packed launches raise the coverage of a
  partial expansion; they produce no admission, and the remaining 350 launches
  still hold the model short of `complete_full_model`.
- The synthetic-arena coupling in section 4 is measured, not fixed. A target's
  private-arena base still depends on how many bindings were accepted before it.

## 6. Reproduce

```bash
python3 integrations/sglang/memgen-adapter/expand_profiles.py \
  --sample-output  out/collect-20260923T165342Z/runs/qwen25_1p5b-p32-d2-collect/followthrough/sample \
  --layer-bindings out/collect-20260923T165342Z/runs/qwen25_1p5b-p32-d2-collect/followthrough/plan/layer-bindings.json \
  --output         out/<fresh-directory>/expanded \
  --model-uncovered refuse
```

The expansion writes 1710 profiles and reports 350 unsupported launches, all
`Source profile missing/unsupported`. The control value above is the same command
run from `integrations/sglang/revisions.json`'s previous `current_sha256` for
`expand_profiles.py`.

## 7. Confirmed end to end on a fresh run

Section 4 measured the change by re-expanding the archived sample. It was then
confirmed on a fresh collection of the same point, with the patch in the working
tree and with its own census, sampler build, sparse sample and expansion, so the
numbers below do not come from the earlier run's artifacts:

```bash
./memgen collect --model qwen25_1p5b --prefill-length 32 --decode-steps 2 --gpu-index 1
```

Run directory `out/collect-20260924T140203Z`. Its expansion reports 1710 of 2060
launches packed and 350 unsupported, all `Source profile missing/unsupported`,
and every binding that used the new rule records
`ambiguous_binding_resolution.deepest_callsite_unique` at hop 0, 35544 times.

| | archived sample (section 4) | fresh run |
| --- | ---: | ---: |
| target launches | 2060 | 2060 |
| packed launches | 1710 | 1710 |
| unsupported launches | 350 | 350 |
| of which ambiguous binding | 0 | 0 |

The chain then stopped with `STOP_UNSUPPORTED_PROFILES_NOT_FULL_MODEL_TRAFFIC`
and exit code 2, which is the documented refusal to publish a full-model number
while launches remain unmodelled; the receipt keeps
`hardware_accuracy_accepted: false` and `raw_trace_persisted: false`.

So the coverage gain reproduces from a fresh census and sample, and it changes
nothing else in the chain. The remaining 350 launches are the sampler's
address-rule-fit refusals, which this change does not address: 168 RoPE, 112
paged attention, 56 elementwise, 6 unrolled elementwise, 6 reduction and 2
gather launches at this point.
