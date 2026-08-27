## Summary

This PR adds `rnn_tanh.input` and `rnn_tanh.data` implementations in the generic FlagGems operator directory. NVIDIA, Ascend, T-Head, and MetaX were revalidated on source commit `d5b375a69bb4df7e12aeed38498723627eee19ea`. Hygon and Iluvatar retain generic dispatch fallback but are excluded from this retest; no new validation claim is made for them.

The implementation covers bias/no-bias, multiple layers, bidirectional execution, batch-first dense input, training dropout, packed sequences, and Triton backward. Dense forward correctness covers FP16, FP32, and BF16; the packed and gradient tests in this validation use FP32.

## Changes

- Add the dense `rnn_tanh` and packed `rnn_tanh_data` implementations in `src/flag_gems/ops/rnn_tanh.py`, without calling native Torch/backend RNN for computation.
- Register and export both ATen overloads and include the packed overload when selecting the public `rnn_tanh` configuration group.
- Use the common `rnn_tanh` pytest marker for both overloads.
- Remove identical T-Head, Hygon, MetaX, and Iluvatar runtime copies and rely on generic ops fallback when the higher-priority runtime directory has no implementation.
- Select persistent dot, vector reduction, or split input-projection/recurrent paths by platform, dtype, and feature dimensions.
- Keep small NVIDIA hidden states on the vector path and use the dot path for applicable larger BF16 states, with FP32 accumulation.
- Precompute input projections for applicable NVIDIA non-BF16 H=128 and MetaX FP32 H=128 cases.
- Implement Ascend-specific matrix and activation kernels in the same generic source file. Reuse the existing FlagGems tanh scalar function inside Triton kernels.
- Fuse three recurrent timesteps per chunk for applicable Ascend BF16 H=128 cases, with separate first/final/remainder handling.
- Use recurrent K=128 reduction tiles for the applicable Ascend H=128 path, while retaining activation tiles of at most 512 elements.
- Resolve the current Ascend stream once per invocation and reuse bound launch runners for compatible middle steps, without caching streams across calls or reusing incompatible scalar-specialized kernels.
- Use FlagGems Philox seed/offset management for reproducible training dropout. PyTorch allocation and metadata access are not computational fallback.
- Add regression checks against launcher composition using native Torch mm, matmul, addmm, stack, cat, or tanh.
- Add functional coverage for forward modes, dense and packed backward, packed dispatch, dropout, large hidden states, non-default streams, and BF16 long bidirectional sequences.
- Retain the original `benchmark/base.py` timing implementation and the same three comprehensive shapes and three dtypes. Do not change the profiler or averaging definition for acceptance.
- Update source code via local commits and remote pulls only. The T-Head main-checkout branch switch and the existing MetaX container start were explicitly authorized; no dependency installation or persistent environment/proxy reconfiguration was needed.

## Test Command

```bash
python -m pytest tests/test_rnn_tanh.py -m rnn_tanh -q
python -m pytest tests/test_rnn_tanh.py -m rnn_tanh --collect-only -q
python -m pytest -s benchmark/test_rnn_tanh.py --mode kernel --level comprehensive --metrics latency_base --metrics latency --metrics speedup --warmup 100 --iter 100 -q
python -m pytest -s benchmark/test_rnn_tanh.py --mode operator --level comprehensive --metrics latency_base --metrics latency --metrics speedup --warmup 100 --iter 100 -q
```

All four retested platforms passed 40 functional tests. Tests use each device's native Torch reference; T-Head uses its available generic decomposition. No CPU-reference coverage is claimed: the current test module skips accelerator cases under `--ref=cpu`.

Run inside the container or venv specified by the login file. NVIDIA used device 1, Ascend used NPU 1, and T-Head/MetaX used device 0. Test discovery uses the repository's existing `pytest.ini` (`pythonpath=src`); no package installation was added.

## Benchmark

Each mode uses three shapes and three dtypes: nine distinct dense forward cases per platform. First average the three per-shape baseline/Gems latency ratios for each dtype, then average the FP16, FP32, and BF16 means. This is an equal-weight arithmetic mean, not a geometric mean or the ratio of summed latencies.

All cases use one layer, one direction, biases, dropout=0, train=False, batch_first=False, and input_size=hidden_size. Packed and backward performance are not part of this acceptance. The threshold applies independently to each platform's overall kernel and operator means, not to every individual dtype or shape.

Measurements were collected on 2026-08-27 at source commit `d5b375a6`. Ascend ran three complete repetitions per mode; NVIDIA, T-Head, and MetaX each ran one. Thus there are 108 measurement records but only nine unique cases per platform/mode. Tables use three-decimal display values; statistics are calculated from six-decimal reported latencies before rounding speedups. No slow cases were excluded.

The benchmark base blob is `276a5ff28d161d03bd20d4f793f1f9a63e59c177`, unchanged from the pre-optimization branch timing implementation. Warmup/iteration flags are interpreted by the original framework; operator-mode loop counts are estimated by the framework.

### NPU

Ascend 910B; Torch 2.6.0+cpu with torch_npu, Triton 3.2. Each row below is the arithmetic mean of that case's three complete-round speedups.

kernel comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|0.891x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|0.987x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|0.950x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|0.858x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|0.896x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|0.883x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|0.749x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|0.881x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|0.257x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 0.943x, the FP32 arithmetic mean speedup is 0.879x, and the BF16 arithmetic mean speedup is 0.629x. Using the unrounded dtype means, the final Ascend kernel speedup is 0.817x, which is greater than 0.8x.

operator comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|0.629x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|0.803x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|1.035x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|0.656x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|0.817x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|0.984x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|0.760x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|0.855x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|3.388x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 0.822x, the FP32 arithmetic mean speedup is 0.819x, and the BF16 arithmetic mean speedup is 1.668x. Using the unrounded dtype means, the final Ascend operator speedup is 1.103x, which is greater than 0.8x.

Per-round overall kernel means are 0.815x, 0.810x, and 0.826x; operator means are 1.120x, 1.101x, and 1.088x. Every complete round passes both modes. The kernel margin is small and results may vary on shared hardware.

The installed Ascend profiler averages selected CSV kernel records rather than summing all kernels in one RNN invocation. It returns Duration(us), although the benchmark labels the field as ms. These results deliberately retain that original collector and compare ratios within the same measurement convention. The kernel speedup is not an end-to-end device-call speedup; operator timing measures the whole call including host dispatch and synchronization. Do not compare these kernel absolute latency values directly with other platforms.

### GPU

NVIDIA H20; Torch 2.11.0a0+eb65b36914.nv26.02, Triton 3.6.

kernel comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|0.283x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|0.408x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|0.520x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|0.294x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|0.303x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|0.299x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|4.287x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|4.937x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|4.414x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 0.404x, the FP32 arithmetic mean speedup is 0.298x, and the BF16 arithmetic mean speedup is 4.546x. Using the unrounded dtype means, the final NVIDIA kernel speedup is 1.749x, which is greater than 0.8x.

operator comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|0.385x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|0.466x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|0.465x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|0.523x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|0.496x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|0.285x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|3.374x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|4.843x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|5.357x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 0.439x, the FP32 arithmetic mean speedup is 0.435x, and the BF16 arithmetic mean speedup is 4.525x. Using the unrounded dtype means, the final NVIDIA operator speedup is 1.799x, which is greater than 0.8x.

The overall means pass, but NVIDIA FP16 and FP32 dtype means are below 0.8x in both modes. BF16 gains drive the aggregate result; this is not a claim that all NVIDIA cases or dtypes meet the target. Native RNN weight-compaction warnings were retained rather than changing the reference path.

### T-Head

PPU-ZW810E; Torch 2.9.0, Triton 3.5. The login-file venv imports the main checkout, which was switched to `rnn_tanh` with explicit authorization before testing.

kernel comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|45.016x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|36.415x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|26.292x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|28.938x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|22.000x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|19.667x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|44.835x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|36.750x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|25.573x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 35.907x, the FP32 arithmetic mean speedup is 23.535x, and the BF16 arithmetic mean speedup is 35.719x. Using the unrounded dtype means, the final T-Head kernel speedup is 31.721x, which is greater than 0.8x.

operator comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|9.797x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|23.046x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|29.544x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|9.335x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|18.525x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|22.431x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|10.983x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|21.243x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|28.612x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 20.796x, the FP32 arithmetic mean speedup is 16.763x, and the BF16 arithmetic mean speedup is 20.280x. Using the unrounded dtype means, the final T-Head operator speedup is 19.280x, which is greater than 0.8x.

The installed ACDNN path does not support tanh RNN. The existing test and benchmark modules therefore disable the cudnn-compatible fast path and compare with Torch's runnable generic decomposition. These large speedups are not measurements against a fused ACDNN RNN implementation.

### Hygon

Hygon is excluded from this retest. The runtime duplicate was removed in favor of generic fallback, but neither current correctness nor performance is asserted here.

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16/fp32/bf16|N/A|Not retested in either mode|

The packed overload was not performance-tested:

|Operator|FP16 Mean|FP32 Mean|BF16 Mean|Final Mean|Minimum|Maximum|
|---|---:|---:|---:|---:|---:|---:|
|rnn_tanh_data|N/A|N/A|N/A|N/A|N/A|N/A|

### MetaX

MetaX C550; Torch 2.8.0+metax3.7.0.7, Triton 3.0. The existing login-file container `flagtree-dev-elzat` was started with authorization, and its checkout was updated by `git pull --ff-only`. Source provenance and file hashes match the tested commit.

kernel comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|16.356x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|16.298x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|1.080x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|12.735x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|2.977x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|3.478x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|5.808x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|4.499x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|2.086x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 11.245x, the FP32 arithmetic mean speedup is 6.397x, and the BF16 arithmetic mean speedup is 4.131x. Using the unrounded dtype means, the final MetaX kernel speedup is 7.257x, which is greater than 0.8x.

operator comprehensive:

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16|6.323x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp16|14.352x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp16|1.101x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|fp32|10.467x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|fp32|3.071x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|fp32|3.625x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|
|rnn_tanh.input|bf16|8.422x|`input=(16,4,32), hx=(1,4,32), params=[(32,32),(32,32),(32,),(32,)]`|
|rnn_tanh.input|bf16|15.695x|`input=(32,8,64), hx=(1,8,64), params=[(64,64),(64,64),(64,),(64,)]`|
|rnn_tanh.input|bf16|5.839x|`input=(64,16,128), hx=(1,16,128), params=[(128,128),(128,128),(128,),(128,)]`|

The FP16 arithmetic mean speedup is 7.259x, the FP32 arithmetic mean speedup is 5.721x, and the BF16 arithmetic mean speedup is 9.986x. Using the unrounded dtype means, the final MetaX operator speedup is 7.655x, which is greater than 0.8x.

These ratios are relative to the native Torch path on this MetaX environment. They should not be interpreted as equivalent speedups over NVIDIA cuDNN or over a different vendor's RNN library. Operator and kernel timings include different host-dispatch effects, so their ratios need not match.

### Iluvatar

Iluvatar is excluded from this retest. Generic fallback remains available, but no new validation claim is made.

|Operator|Dtype|Speedup|Size Detail|
|---|---|---:|---|
|rnn_tanh.input|fp16/fp32/bf16|N/A|Not retested in either mode|

The packed overload was not performance-tested:

|Operator|FP16 Mean|FP32 Mean|BF16 Mean|Final Mean|Minimum|Maximum|
|---|---:|---:|---:|---:|---:|---:|
|rnn_tanh_data|N/A|N/A|N/A|N/A|N/A|N/A|

### Final Speedup Summary

kernel comprehensive:

|Platform|FP16 Mean Speedup|FP32 Mean Speedup|BF16 Mean Speedup|Final Speedup|Requirement|
|---|---:|---:|---:|---:|---|
|Ascend|0.943x|0.879x|0.629x|0.817x|PASS (>0.8x)|
|NVIDIA|0.404x|0.298x|4.546x|1.749x|PASS (>0.8x)|
|T-Head|35.907x|23.535x|35.719x|31.721x|PASS (>0.8x)|
|Hygon|Not retested|Not retested|Not retested|N/A|Excluded from this retest|
|MetaX|11.245x|6.397x|4.131x|7.257x|PASS (>0.8x)|
|Iluvatar|Not retested|Not retested|Not retested|N/A|Excluded from this retest|

operator comprehensive:

|Platform|FP16 Mean Speedup|FP32 Mean Speedup|BF16 Mean Speedup|Final Speedup|Requirement|
|---|---:|---:|---:|---:|---|
|Ascend|0.822x|0.819x|1.668x|1.103x|PASS (>0.8x)|
|NVIDIA|0.439x|0.435x|4.525x|1.799x|PASS (>0.8x)|
|T-Head|20.796x|16.763x|20.280x|19.280x|PASS (>0.8x)|
|Hygon|Not retested|Not retested|Not retested|N/A|Excluded from this retest|
|MetaX|7.259x|5.721x|9.986x|7.655x|PASS (>0.8x)|
|Iluvatar|Not retested|Not retested|Not retested|N/A|Excluded from this retest|

All four required platforms pass both overall arithmetic-mean thresholds. This conclusion uses the documented Ascend collector convention and T-Head generic reference, and does not imply every case passes or that packed/backward performance has been validated.

Raw local evidence is retained under `output/rnn_tanh_retest_20260827/`: `ascend_stream_cached_validation.log`, `nvidia_final_d5b375a6.log`, `thead_final_d5b375a6.log`, `metax_final_d5b375a6.log`, and the per-record `four_platform_final_results.json`. These are local test artifacts, not additional uploaded PR attachments.
