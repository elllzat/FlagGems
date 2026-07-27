## Summary

This PR adds `log_sigmoid_backward` and `log_sigmoid_backward.grad_input` support for NVIDIA GPUs, Huawei Ascend NPUs, Hygon DCUs, and T-Head PPUs.

The implementation follows PyTorch semantics for empty and populated forward buffers, contiguous and non-contiguous inputs, float16, float32, bfloat16, autograd, and `grad_input` handling.

## Changes

- Add Triton implementations for `log_sigmoid_backward` and `log_sigmoid_backward.grad_input`.
- Register and export the default and `grad_input` overloads as `log_sigmoid_backward` and `log_sigmoid_backward_out`.
- Allocate default outputs with the supported `flag_gems.empty` operator.
- Replace the zero-initializing `flag_gems.empty` implementation with BackendSelect redispatch to the native uninitialized device allocator, avoiding a redundant zero-fill kernel while preserving allocation arguments.
- Add contiguous Triton kernels and Triton pointwise paths for non-contiguous layouts on all four platforms.
- Reuse populated forward buffers for NVIDIA and Hygon float16 and bfloat16 inputs.
- Recompute the derivative with fused Triton sigmoid for NVIDIA and Hygon float32 and empty-buffer inputs.
- Add pure Triton Ascend kernels without calling the native torch_npu `log_sigmoid_backward` implementation.
- Add an Ascend persistent-grid kernel with 8192-element blocks and bounded grid size.
- Avoid redundant Ascend device guards when the input already uses the current device.
- Add Hygon backend implementations using the optimized 1024-element contiguous kernel and pointwise fallback.
- Add T-Head backend implementations that recompute sigmoid for every supported dtype to avoid buffer-read overhead.
- Add separate correctness markers for `log_sigmoid_backward` and `log_sigmoid_backward_out` covering empty and populated buffers, 0D through 5D tensors, contiguous and non-contiguous layouts, special values, autograd, and return-object identity.
- Add benchmark coverage for float16, float32, and bfloat16, including backend-specific launch-overhead cases, bounded-memory core cases, and comprehensive shape coverage.

## Test Command

```bash
pytest -q tests/test_log_sigmoid.py -m log_sigmoid_backward
pytest -q tests/test_log_sigmoid.py -m log_sigmoid_backward_out
pytest -q -s benchmark/test_empty.py --level core
pytest -s -q benchmark/test_log_sigmoid_backward.py --mode kernel --level core --warmup 100 --iter 100
pytest -s -q benchmark/test_log_sigmoid_backward.py --mode kernel --level comprehensive
```

The `empty` core benchmark completed 15 of 15 cases on each of NVIDIA, Ascend, and T-Head. NVIDIA additionally passed tuple-size, dtype/device, channels-last memory-format, and `use_gems(include=["empty"])` smoke checks. These checks validate the exercised allocation paths; they do not assume or guarantee that unrelated callers may depend on zero-filled contents, which would violate `torch.empty` semantics.

## Benchmark

### Ascend NPU

Mode: kernel, level: comprehensive

|Operator|Dtype|Torch Latency (ms)|Gems Latency (ms)|Speedup|Size Detail|
|---|---|---:|---:|---:|---|
|log_sigmoid_backward|float16|0.858800|0.550880|1.559x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float16|0.646000|0.355240|1.818x|`[torch.Size([4608, 4608]), torch.Size([4608, 4608]), torch.Size([4608, 4608])]`|
|log_sigmoid_backward|float16|0.475180|0.292260|1.626x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float16|0.474680|0.288600|1.645x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float16|1.000760|0.459240|2.179x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float16|0.002360|0.003160|0.747x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float16|0.003840|0.003360|1.143x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float16|0.007860|0.007400|1.062x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float16|0.131380|0.077160|1.703x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float16|1.741560|1.038980|1.676x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float16|0.002560|0.003280|0.780x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float16|0.008100|0.004280|1.893x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float16|0.035440|0.021680|1.635x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float16|0.425020|0.287900|1.476x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|float32|1.150140|0.635020|1.811x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float32|0.666100|0.456140|1.460x|`[torch.Size([4608, 4608]), torch.Size([4608, 4608]), torch.Size([4608, 4608])]`|
|log_sigmoid_backward|float32|0.541860|0.354060|1.530x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float32|0.545500|0.350400|1.557x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float32|1.053500|0.627020|1.680x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float32|0.002000|0.003280|0.610x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float32|0.003860|0.004140|0.932x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float32|0.010220|0.010760|0.950x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float32|0.185440|0.098420|1.884x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float32|2.211560|1.319540|1.676x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float32|0.003120|0.002960|1.054x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float32|0.005480|0.004420|1.240x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float32|0.038240|0.025200|1.517x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float32|0.550240|0.365760|1.504x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|bfloat16|0.902420|0.554940|1.626x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|bfloat16|0.651620|0.354180|1.840x|`[torch.Size([4608, 4608]), torch.Size([4608, 4608]), torch.Size([4608, 4608])]`|
|log_sigmoid_backward|bfloat16|0.478740|0.286600|1.670x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|bfloat16|0.425800|0.272200|1.564x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.907160|0.537060|1.689x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.002200|0.002880|0.764x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|bfloat16|0.003620|0.003540|1.023x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|bfloat16|0.008240|0.008720|0.945x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|bfloat16|0.126840|0.060940|2.081x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|bfloat16|1.801460|1.054800|1.708x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|bfloat16|0.002180|0.002980|0.732x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|bfloat16|0.007200|0.004860|1.481x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|bfloat16|0.038580|0.022800|1.692x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|bfloat16|0.332740|0.270800|1.229x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|

The Ascend results cover 42 comprehensive kernel-mode cases. The measured speedup range is 0.610x to 2.179x. Thirty-seven cases exceed 0.8x; the five lower ratios contain only 1024 or 4096 elements and compare against native measurements of 0.002000 ms to 0.002560 ms, where launch and timing-floor effects dominate.

### NVIDIA GPU

Mode: kernel, level: comprehensive

|Operator|Dtype|Torch Latency (ms)|Gems Latency (ms)|Speedup|Size Detail|
|---|---|---:|---:|---:|---|
|log_sigmoid_backward|float16|0.080512|0.069888|1.152x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float16|0.005696|0.005472|1.041x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float16|0.041760|0.039456|1.058x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float16|0.041632|0.039232|1.061x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float16|0.076800|0.069792|1.100x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float16|0.005920|0.005408|1.095x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float16|0.005824|0.005568|1.046x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float16|0.006432|0.006080|1.058x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float16|0.014912|0.014752|1.011x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float16|0.188928|0.130976|1.442x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float16|0.005664|0.005440|1.041x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float16|0.006208|0.005760|1.078x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float16|0.008720|0.007968|1.094x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float16|0.041648|0.041200|1.011x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|float32|0.100496|0.099360|1.011x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float32|0.005664|0.005472|1.035x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float32|0.054688|0.054112|1.011x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float32|0.054368|0.053824|1.010x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float32|0.102368|0.099328|1.031x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float32|0.005632|0.005152|1.093x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float32|0.005984|0.005632|1.062x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float32|0.006912|0.006592|1.049x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float32|0.019040|0.018848|1.010x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float32|0.271904|0.191360|1.421x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float32|0.005664|0.005472|1.035x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float32|0.006048|0.005600|1.080x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float32|0.009056|0.008768|1.033x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float32|0.054336|0.054080|1.005x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|bfloat16|0.083904|0.070272|1.194x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|bfloat16|0.005664|0.005472|1.035x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|bfloat16|0.041248|0.040320|1.023x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|bfloat16|0.041536|0.040160|1.034x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.076832|0.070016|1.097x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.005856|0.005152|1.137x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|bfloat16|0.006048|0.005408|1.118x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|bfloat16|0.006400|0.006240|1.026x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|bfloat16|0.014976|0.014944|1.002x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|bfloat16|0.187008|0.131280|1.424x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|bfloat16|0.005920|0.005216|1.135x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|bfloat16|0.006048|0.005600|1.080x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|bfloat16|0.008736|0.008192|1.066x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|bfloat16|0.041440|0.039552|1.048x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|

The NVIDIA results cover 42 comprehensive kernel-mode cases. Every case exceeds 0.8x, with a measured speedup range of 1.002x to 1.442x.

### Hygon DCU

Mode: kernel, level: comprehensive

|Operator|Dtype|Torch Latency (ms)|Gems Latency (ms)|Speedup|Size Detail|
|---|---|---:|---:|---:|---|
|log_sigmoid_backward|float16|0.193601|0.219521|0.882x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float16|0.007840|0.009280|0.845x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float16|0.101120|0.112960|0.895x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float16|0.100960|0.112960|0.894x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float16|0.193601|0.219520|0.882x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float16|0.007840|0.009280|0.845x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float16|0.007840|0.008160|0.961x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float16|0.009120|0.008960|1.018x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float16|0.031360|0.033120|0.947x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float16|0.378721|0.432481|0.876x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float16|0.007840|0.009280|0.845x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float16|0.007680|0.009280|0.828x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float16|0.013280|0.012960|1.025x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float16|0.101120|0.112800|0.896x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|float32|0.308801|0.309921|0.996x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float32|0.007840|0.009280|0.845x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float32|0.156641|0.156960|0.998x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float32|0.156961|0.156961|1.000x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float32|0.308640|0.309760|0.996x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float32|0.008000|0.007840|1.020x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float32|0.007680|0.009280|0.828x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float32|0.008960|0.009440|0.949x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float32|0.044160|0.043360|1.018x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float32|0.612561|0.616481|0.994x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float32|0.009120|0.007840|1.163x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float32|0.009120|0.008000|1.140x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float32|0.015840|0.014880|1.065x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float32|0.156961|0.156960|1.000x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|bfloat16|0.210080|0.220961|0.951x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|bfloat16|0.009280|0.009280|1.000x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|bfloat16|0.109280|0.113920|0.959x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|bfloat16|0.109280|0.113920|0.959x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.210080|0.220961|0.951x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.009280|0.009280|1.000x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|bfloat16|0.007680|0.009280|0.828x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|bfloat16|0.009120|0.009280|0.983x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|bfloat16|0.033600|0.033441|1.005x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|bfloat16|0.411521|0.434881|0.946x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|bfloat16|0.007680|0.009280|0.828x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|bfloat16|0.007840|0.009120|0.860x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|bfloat16|0.014400|0.012640|1.139x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|bfloat16|0.109281|0.113601|0.962x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|

The Hygon results cover 42 comprehensive kernel-mode cases. The measured speedup range is 0.828x to 1.163x across all supported dtypes and shapes.

### T-Head PPU

Mode: kernel, level: comprehensive

|Operator|Dtype|Torch Latency (ms)|Gems Latency (ms)|Speedup|Size Detail|
|---|---|---:|---:|---:|---|
|log_sigmoid_backward|float16|0.119880|0.120220|0.997x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float16|0.003560|0.002560|1.391x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float16|0.074020|0.073360|1.009x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float16|0.074200|0.073520|1.009x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float16|0.125600|0.124000|1.013x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float16|0.003560|0.002320|1.534x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float16|0.003640|0.002560|1.422x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float16|0.004600|0.003480|1.322x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float16|0.025160|0.023160|1.086x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float16|0.209400|0.214560|0.976x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float16|0.003600|0.002560|1.406x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float16|0.003800|0.002760|1.377x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float16|0.009200|0.007960|1.156x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float16|0.073680|0.072240|1.020x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|float32|0.216200|0.215800|1.002x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float32|0.003560|0.002760|1.290x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float32|0.120360|0.119520|1.007x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float32|0.120200|0.119400|1.007x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float32|0.213960|0.213000|1.005x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float32|0.003440|0.002320|1.483x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float32|0.003600|0.002760|1.304x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float32|0.005720|0.004840|1.182x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float32|0.044440|0.043720|1.016x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float32|0.410560|0.408720|1.005x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float32|0.003600|0.002560|1.406x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float32|0.003840|0.003000|1.280x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float32|0.013520|0.012600|1.073x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float32|0.121000|0.120040|1.008x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|bfloat16|0.120520|0.119840|1.006x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|bfloat16|0.003400|0.002760|1.232x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|bfloat16|0.074380|0.073440|1.013x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|bfloat16|0.074540|0.073560|1.013x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.125720|0.124000|1.014x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.003360|0.002320|1.448x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|bfloat16|0.003440|0.002760|1.246x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|bfloat16|0.004520|0.003480|1.299x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|bfloat16|0.025120|0.023200|1.083x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|bfloat16|0.211280|0.214800|0.984x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|bfloat16|0.003400|0.002760|1.232x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|bfloat16|0.003640|0.002760|1.319x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|bfloat16|0.009040|0.007960|1.136x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|bfloat16|0.073800|0.072360|1.020x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|

The T-Head results cover 42 comprehensive kernel-mode cases. Every case exceeds 0.8x, with a measured speedup range of 0.976x to 1.534x.
