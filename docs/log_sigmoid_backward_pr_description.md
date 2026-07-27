## Summary

This PR adds `log_sigmoid_backward` and `log_sigmoid_backward.grad_input` support for NVIDIA GPUs, Huawei Ascend NPUs, Hygon DCUs, and T-Head PPUs.

The implementation follows PyTorch semantics for empty and populated forward buffers, contiguous and non-contiguous inputs, float16, float32, bfloat16, autograd, and `grad_input` handling.

## Changes

- Add Triton implementations for `log_sigmoid_backward` and `log_sigmoid_backward.grad_input`.
- Register and export the default and `grad_input` overloads as `log_sigmoid_backward` and `log_sigmoid_backward_out`.
- Allocate default outputs through the Triton pointwise dynamic path without changing the public `flag_gems.empty` implementation.
- Add handwritten contiguous Triton kernels for the `grad_input` overload and Triton pointwise paths for default outputs and non-contiguous layouts on all four platforms.
- Reuse populated forward buffers for NVIDIA and Hygon float16 and bfloat16 inputs.
- Recompute the derivative with fused Triton sigmoid for NVIDIA and Hygon float32 and empty-buffer inputs.
- Add pure Triton Ascend kernels without calling the native torch_npu `log_sigmoid_backward` implementation.
- Add an Ascend pointwise configuration with 4096-element tiles and a bounded grid, plus a persistent-grid `grad_input` kernel with 8192-element blocks.
- Avoid redundant Ascend device guards when the input already uses the current device.
- Add Hygon backend implementations using the optimized 1024-element contiguous kernel and pointwise fallback.
- Add T-Head backend implementations that recompute sigmoid for every supported dtype to avoid buffer-read overhead.
- Add separate correctness markers for `log_sigmoid_backward` and `log_sigmoid_backward_out` covering empty and populated buffers, 0D through 5D tensors, contiguous and non-contiguous layouts, special values, autograd, and return-object identity.
- Add common benchmark coverage for float16, float32, and bfloat16 across 1D, 2D, and 3D throughput shapes from 1M to 67M elements.

## Test Command

```bash
pytest -q tests/test_log_sigmoid.py -m log_sigmoid_backward
pytest -q tests/test_log_sigmoid.py -m log_sigmoid_backward_out
pytest -s -q benchmark/test_log_sigmoid_backward.py --mode kernel --level core --warmup 100 --iter 100
pytest -s -q benchmark/test_log_sigmoid_backward.py --mode kernel --level comprehensive
```

The functional test selected 30 cases and passed on all four platforms. The comprehensive benchmark selected 24 cases and passed on all four platforms. The benchmark uses the same eight shapes and three dtypes on every platform.

## Benchmark

Mode: kernel, level: comprehensive

For each platform, the speedups of the eight shapes are averaged within each dtype. The final speedup is the arithmetic mean of the float16, float32, and bfloat16 averages.

|Shape|Elements|
|---|---:|
|`(33554432,)`|33554432|
|`(4096,4096)`|16777216|
|`(64,512,512)`|16777216|
|`(128,512,512)`|33554432|
|`(1024,4096)`|4194304|
|`(1024,65536)`|67108864|
|`(64,64,256)`|1048576|
|`(64,64,4096)`|16777216|

### NVIDIA GPU

|Dtype|Shape Speedups|Minimum|Maximum|Average|
|---|---|---:|---:|---:|
|float16|1.072x, 1.013x, 1.002x, 1.040x, 0.419x, 1.185x, 0.243x, 1.012x|0.243x|1.185x|0.873x|
|float32|1.007x, 1.014x, 0.997x, 1.087x, 0.603x, 1.139x, 0.300x, 0.997x|0.300x|1.139x|0.893x|
|bfloat16|1.017x, 0.994x, 1.005x, 1.018x, 0.422x, 1.276x, 0.246x, 1.014x|0.246x|1.276x|0.874x|

Final speedup: 0.880x.

### Ascend NPU

|Dtype|Shape Speedups|Minimum|Maximum|Average|
|---|---|---:|---:|---:|
|float16|1.357x, 0.949x, 0.926x, 0.910x, 1.191x, 0.917x, 0.195x, 0.925x|0.195x|1.357x|0.921x|
|float32|0.927x, 1.054x, 1.001x, 0.678x, 0.756x, 1.158x, 0.933x, 1.051x|0.678x|1.158x|0.945x|
|bfloat16|1.321x, 0.913x, 0.928x, 0.923x, 1.190x, 0.912x, 0.205x, 0.910x|0.205x|1.321x|0.913x|

Final speedup: 0.926x.

### Hygon DCU

|Dtype|Shape Speedups|Minimum|Maximum|Average|
|---|---|---:|---:|---:|
|float16|0.879x, 0.889x, 0.890x, 0.880x, 0.713x, 0.873x, 0.271x, 0.893x|0.271x|0.893x|0.786x|
|float32|0.998x, 0.996x, 0.998x, 0.997x, 1.000x, 0.998x, 0.559x, 0.997x|0.559x|1.000x|0.943x|
|bfloat16|0.946x, 0.953x, 0.953x, 0.945x, 0.736x, 0.941x, 0.283x, 0.955x|0.283x|0.955x|0.839x|

Final speedup: 0.856x.

### T-Head PPU

|Dtype|Shape Speedups|Minimum|Maximum|Average|
|---|---|---:|---:|---:|
|float16|1.030x, 1.029x, 1.029x, 1.024x, 1.088x, 1.006x, 1.192x, 1.028x|1.006x|1.192x|1.053x|
|float32|1.009x, 1.000x, 1.002x, 1.005x, 1.005x, 0.994x, 1.070x, 0.999x|0.994x|1.070x|1.011x|
|bfloat16|1.029x, 1.033x, 1.042x, 1.031x, 1.083x, 1.013x, 1.159x, 1.026x|1.013x|1.159x|1.052x|

Final speedup: 1.039x.

All four platform-level final speedups exceed 0.8x.
