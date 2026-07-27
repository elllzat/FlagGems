## Summary

This PR adds `log_sigmoid_backward` and `log_sigmoid_backward.grad_input` support for NVIDIA GPUs, Huawei Ascend NPUs, Hygon DCUs, and T-Head PPUs.

The implementation follows PyTorch semantics for empty and populated forward buffers, contiguous and non-contiguous inputs, float16, float32, bfloat16, autograd, and `grad_input` handling.

## Changes

- Add Triton implementations for `log_sigmoid_backward` and `log_sigmoid_backward.grad_input`.
- Register and export the default and `grad_input` overloads as `log_sigmoid_backward` and `log_sigmoid_backward_out`.
- Allocate default outputs with the supported `flag_gems.empty` operator.
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
pytest -s -q benchmark/test_log_sigmoid_backward.py --mode kernel --level core --warmup 100 --iter 100
pytest -s -q benchmark/test_log_sigmoid_backward.py --mode kernel --level comprehensive
```

## Benchmark

### Ascend NPU

Mode: kernel

|Operator|Dtype|Torch Latency (ms)|Gems Latency (ms)|Speedup|Size Detail|
|---|---|---:|---:|---:|---|
|log_sigmoid_backward|float16|0.365340|0.271720|1.345x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float16|0.235780|0.174300|1.353x|`[torch.Size([4608, 4608]), torch.Size([4608, 4608]), torch.Size([4608, 4608])]`|
|log_sigmoid_backward|float16|0.186440|0.139360|1.338x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float16|0.187000|0.139720|1.338x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float16|0.363900|0.271840|1.339x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float32|0.475680|0.441120|1.078x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float32|0.327660|0.308340|1.063x|`[torch.Size([4608, 4608]), torch.Size([4608, 4608]), torch.Size([4608, 4608])]`|
|log_sigmoid_backward|float32|0.274440|0.268840|1.021x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float32|0.279000|0.270100|1.033x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float32|0.472300|0.443320|1.065x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.364360|0.269740|1.351x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|bfloat16|0.233960|0.173580|1.348x|`[torch.Size([4608, 4608]), torch.Size([4608, 4608]), torch.Size([4608, 4608])]`|
|log_sigmoid_backward|bfloat16|0.185840|0.139520|1.332x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|bfloat16|0.187440|0.139220|1.346x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.362580|0.270760|1.339x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|

The NPU results are measured in kernel mode. The Triton implementation uses a persistent grid, 8192-element blocks, and fused sigmoid recomputation to avoid the extra buffer read and division path. The backend-specific launch-overhead shape keeps native measurements stable while the remaining 1D, 2D, and 3D cases cover sustained throughput without excessive shared-device memory use.

### NVIDIA GPU

Mode: kernel

|Operator|Dtype|Torch Latency (ms)|Gems Latency (ms)|Speedup|Size Detail|
|---|---|---:|---:|---:|---|
|log_sigmoid_backward|float16|0.074752|0.069728|1.072x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float16|0.005760|0.005280|1.091x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float16|0.040800|0.038272|1.066x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float16|0.040800|0.038528|1.059x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float16|0.077984|0.069696|1.119x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float32|0.101408|0.099008|1.024x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float32|0.005728|0.005344|1.072x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float32|0.053952|0.053536|1.008x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float32|0.054080|0.053264|1.015x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float32|0.100032|0.098976|1.011x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.076736|0.070080|1.095x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|bfloat16|0.005568|0.005088|1.094x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|bfloat16|0.040832|0.038304|1.066x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|bfloat16|0.040928|0.038640|1.059x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.076928|0.070016|1.099x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|

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
|log_sigmoid_backward|float16|0.119760|0.120280|0.996x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float16|0.003600|0.002560|1.406x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float16|0.074040|0.073400|1.009x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float16|0.074240|0.073520|1.010x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float16|0.125680|0.124000|1.014x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float16|0.003600|0.002320|1.552x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float16|0.003600|0.002560|1.406x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float16|0.004560|0.003480|1.310x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float16|0.025120|0.023160|1.085x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float16|0.209420|0.214680|0.975x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float16|0.003600|0.002560|1.406x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float16|0.003800|0.002760|1.377x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float16|0.009240|0.007960|1.161x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float16|0.073720|0.072280|1.020x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|float32|0.216120|0.215840|1.001x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|float32|0.003600|0.002760|1.304x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|float32|0.120440|0.119520|1.008x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|float32|0.120240|0.119320|1.008x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|float32|0.214000|0.213000|1.005x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|float32|0.003400|0.002320|1.466x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|float32|0.003600|0.002760|1.304x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|float32|0.005720|0.004880|1.172x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|float32|0.044440|0.043800|1.015x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|float32|0.410300|0.408920|1.003x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|float32|0.003600|0.002760|1.304x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|float32|0.003840|0.003000|1.280x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|float32|0.013520|0.012600|1.073x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|float32|0.121000|0.119960|1.009x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|
|log_sigmoid_backward|bfloat16|0.120480|0.119880|1.005x|`[torch.Size([33554432]), torch.Size([33554432]), torch.Size([33554432])]`|
|log_sigmoid_backward|bfloat16|0.003400|0.002560|1.328x|`[torch.Size([64, 64]), torch.Size([64, 64]), torch.Size([64, 64])]`|
|log_sigmoid_backward|bfloat16|0.074360|0.073400|1.013x|`[torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096])]`|
|log_sigmoid_backward|bfloat16|0.074560|0.073560|1.014x|`[torch.Size([64, 512, 512]), torch.Size([64, 512, 512]), torch.Size([64, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.125800|0.124000|1.015x|`[torch.Size([128, 512, 512]), torch.Size([128, 512, 512]), torch.Size([128, 512, 512])]`|
|log_sigmoid_backward|bfloat16|0.003360|0.002320|1.448x|`[torch.Size([1024, 1]), torch.Size([1024, 1]), torch.Size([1024, 1])]`|
|log_sigmoid_backward|bfloat16|0.003440|0.002760|1.246x|`[torch.Size([1024, 16]), torch.Size([1024, 16]), torch.Size([1024, 16])]`|
|log_sigmoid_backward|bfloat16|0.004520|0.003480|1.299x|`[torch.Size([1024, 256]), torch.Size([1024, 256]), torch.Size([1024, 256])]`|
|log_sigmoid_backward|bfloat16|0.025040|0.023200|1.079x|`[torch.Size([1024, 4096]), torch.Size([1024, 4096]), torch.Size([1024, 4096])]`|
|log_sigmoid_backward|bfloat16|0.211120|0.214640|0.984x|`[torch.Size([1024, 65536]), torch.Size([1024, 65536]), torch.Size([1024, 65536])]`|
|log_sigmoid_backward|bfloat16|0.003400|0.002560|1.328x|`[torch.Size([64, 64, 1]), torch.Size([64, 64, 1]), torch.Size([64, 64, 1])]`|
|log_sigmoid_backward|bfloat16|0.003640|0.002760|1.319x|`[torch.Size([64, 64, 16]), torch.Size([64, 64, 16]), torch.Size([64, 64, 16])]`|
|log_sigmoid_backward|bfloat16|0.009040|0.007960|1.136x|`[torch.Size([64, 64, 256]), torch.Size([64, 64, 256]), torch.Size([64, 64, 256])]`|
|log_sigmoid_backward|bfloat16|0.073800|0.072360|1.020x|`[torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096]), torch.Size([64, 64, 4096])]`|

The T-Head results cover 42 comprehensive kernel-mode cases. Every case exceeds 0.8x, with a measured speedup range of 0.975x to 1.552x.
