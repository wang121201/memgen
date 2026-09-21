# Accuracy baselines and acceptance boundary

HBServe is the sampled-address generator. Memgen is the functional cache
backend. L1 and L2 mean Level-1 and Level-2 cache. DRAM means Dynamic
Random-Access Memory traffic after the cache model. NCU means NVIDIA Nsight
Compute. `P<n>D<m>` means batch-one inference with `n` prefill tokens followed
by `m` decode steps.

Cache relative error is `abs(model_hit_rate - NCU_hit_rate) / NCU_hit_rate`.
Traffic relative error is `abs(model_bytes - NCU_bytes) / NCU_bytes`. The
historical table below reports magnitudes; the current SGLang table retains the
sign, where positive means model over-estimation.

## Historical frozen regression: llama.cpp/Q8_0

This is the requested five-point table. Each row is an independent
Qwen2.5-1.5B-Instruct Q8_0 + llama.cpp full inference on RTX 4000 Ada, with an
independent sparse sample, profile, HBServe expansion and NCU application
range. It is not SGLang evidence.

| Historical workload | naïve DRAM read/write error | two-level Memgen L1/L2 hit-rate error | two-level Memgen DRAM read/write error |
|---|---:|---:|---:|
| P64D32 | 0.48% / 4.55% | 1.02% / 22.71% | 0.06% / 4.55% |
| P128D32 | 0.90% / 3.64% | 1.09% / 12.53% | 0.05% / 3.63% |
| P256D32 | 1.85% / 2.66% | 1.48% / 8.31% | 0.05% / 2.65% |
| P512D32 | 3.85% / 25.13% | 2.32% / 5.62% | 0.21% / 25.13% |
| P1024D32 | 7.81% / 23.07% | 4.10% / 1.14% | 1.08% / 23.07% |

The canonical machine-readable copy is
`validation/historical_prefill_d32.csv`. The complete address-free analysis is
under `evidence/historical-prefill-d32/`. The result supports the read path at
the measured points, but write traffic fails at P512D32 and P1024D32.

## Current acceptance: SGLang/BF16

The current target is Qwen2.5-1.5B-Instruct BF16, SGLang 0.4.10 + FlashInfer,
batch 1, tensor parallelism 1, eager execution, CUDA Graph disabled and an RTX
4000 Ada Generation GPU with 48 streaming multiprocessors. Each independent
workload uses its own profile and three NCU measurements; the table uses the
median. The cache policy is ordinary LRU with no additional active dirty
cleaner.

| Workload/range | DRAM read error | DRAM write error | status |
|---|---:|---:|---|
| P128D2 whole | +0.54% | +0.91% | numeric traffic gate passes; model domain remains incomplete |
| P128D2 prefill | +0.70% | +0.40% | numeric traffic gate passes |
| P128D2 decode 2 | +0.44% | +39.17% | write fails |
| P128D16 whole | +0.20% | +14.55% | write fails |
| P128D16 prefill | +0.69% | -0.80% | numeric traffic gate passes |
| P128D16 decode 16 | +0.16% | +648.63% | write fails |

Current SGLang NCU runs do not provide a same-range, same-denominator L1/L2
hit-rate oracle, so no SGLang L1/L2 accuracy value is fabricated. P128D4 and
P128D8 prefix sums are diagnostics from P128D16, not independent workload
acceptance. P32D2 is also absent under this exact model/framework contract.

The stable acceptance rule is workload-scoped: report whole, prefill and
decode separately; require absolute read and write error below 10% for every
accepted range; preserve input/source hashes and three-run NCU medians; and do
not promote internal conservation to hardware accuracy.

