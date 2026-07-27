# log_sigmoid_backward开发文档
## 接口介绍
### 算子名字
本次开发包含两个接口：
- `log_sigmoid_backward`
- `log_sigmoid_backward.grad_input`
对应的算子id分别为：
- `log_sigmoid_backward`
- `log_sigmoid_backward_out`
### 输入输出参数
#### log_sigmoid_backward
接口形式：
```python
torch.ops.aten.log_sigmoid_backward(grad_output, self, buffer)
```
|参数|方向|类型|说明|
|---|---|---|---|
|grad_output|输入|Tensor|上游传入的梯度Tensor，与`self`具有相同shape和dtype|
|self|输入|Tensor|log sigmoid前向计算的输入Tensor|
|buffer|输入|Tensor|前向计算保存的`exp(-abs(self))`，设备前向未保存buffer时可以为空Tensor|
约束：
- `grad_output`和`self`必须位于同一设备。
- `grad_output`和`self`必须具有相同的shape和dtype。
- NVIDIA和Hygon连续kernel仅在非空`buffer`元素数量、dtype和连续性符合要求时复用buffer，否则重新计算导数。
- Ascend和T-Head不读取buffer，统一重新计算导数。
- NVIDIA、Ascend、Hygon和T-Head支持`float16`、`float32`和`bfloat16`。
#### log_sigmoid_backward.grad_input
接口形式：
```python
torch.ops.aten.log_sigmoid_backward.grad_input(grad_output, self, buffer, *, grad_input=grad_input)
```
|参数|方向|类型|说明|
|---|---|---|---|
|grad_output|输入|Tensor|上游传入的梯度Tensor，与`self`具有相同shape和dtype|
|self|输入|Tensor|log sigmoid前向计算的输入Tensor|
|buffer|输入|Tensor|前向计算保存的`exp(-abs(self))`，设备前向未保存buffer时可以为空Tensor|
|grad_input|输入/输出|Tensor|接收输入梯度的Tensor，必须与输入位于同一设备并具有相同shape和dtype|
约束：
- 连续输入和连续`grad_input`使用手写Triton kernel，其他布局由Triton pointwise kernel处理。
- 返回值与传入的`grad_input`是同一个Tensor对象。
## 算子功能介绍
`log_sigmoid_backward`计算log sigmoid函数关于输入的梯度。
```text
z=exp(-abs(self))
derivative=self<0?1/(1+z):z/(1+z)
grad_input=grad_output*derivative
```
当不复用buffer时，等价计算为：
```text
grad_input=grad_output*sigmoid(-self)
```
简单示例：
```python
self=torch.randn(2,3)
grad_output=torch.randn_like(self)
buffer=torch.exp(-torch.abs(self))
grad_input=torch.ops.aten.log_sigmoid_backward(grad_output,self,buffer)
```
输出shape为`(2,3)`。每个元素由对应的`grad_output`乘以log sigmoid导数得到。
## 实现方案
默认返回变体通过已支持的`flag_gems.empty`分配输出。原实现先调用`torch.zeros`分配并额外启动写零Triton kernel，不符合`empty`无需初始化的性能预期。当前实现通过BackendSelect redispatch调用设备原生未初始化分配器，避免额外kernel，同时保留tuple size、dtype、device、layout、pin_memory和memory_format参数传递。该公共修改已在NVIDIA、Ascend和T-Head平台通过`benchmark/test_empty.py --level core`验证，NVIDIA另行验证了tuple size、channels-last memory format和`use_gems(include=["empty"])`注册路径。
### NVIDIA GPU平台实现方案
NVIDIA GPU平台使用Triton kernel实现。核心流程是：
1. 检查输入shape、dtype和连续性。
2. 连续输入选择手写Triton kernel，非连续输入选择Triton pointwise kernel。
3. FP16和BF16在有效buffer存在时复用`exp(-abs(self))`。
4. FP32和空buffer路径使用Triton sigmoid稳定重算导数。
5. 默认变体使用`flag_gems.empty`分配输出，out变体直接写入传入的`grad_input`。
实现流程：
```text
grad_output/self/buffer
   |
   |--校验输入布局
   |
   |--选择连续或pointwise Triton kernel
   |
   |--复用buffer或重算sigmoid导数
   |
   |--写入grad_input
```
性能优化手段：
- 连续布局使用单次Triton kernel完成读取、导数计算和梯度写回。
- FP16和BF16复用有效buffer，避免重复指数计算。
- FP32避免额外buffer读访问，使用融合的`tl.sigmoid`重算导数。
- 空buffer路径直接重算，兼容CUDA前向返回空buffer的行为。
- 使用1024元素block兼顾小shape启动开销和大shape吞吐性能。
- 非连续布局使用Triton pointwise kernel，避免回退到Torch。
NVIDIA GPU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|
|---|---|---:|---:|
|float16|kernel comprehensive|1.011x|1.442x|
|float32|kernel comprehensive|1.005x|1.421x|
|bfloat16|kernel comprehensive|1.002x|1.424x|
### Ascend NPU平台实现方案
Ascend NPU平台使用Triton kernel实现，不回调torch_npu原生`log_sigmoid_backward`。
实现流程：
```text
NPU grad_output/self
   |
   |--校验输入布局
   |
   |--选择连续或pointwise Triton kernel
   |      |--连续布局使用libentry手写kernel
   |      |--非连续布局使用pointwise kernel
   |
   |--使用Triton sigmoid重算导数
   |
   |--写入grad_input
```
性能优化手段：
- 连续布局使用8192元素block进行合并访存。
- 使用持久化grid和`TILES_PER_PROGRAM`循环处理超大Tensor。
- 将grid数量限制为65535，避免超出Ascend启动范围。
- 使用`tl.sigmoid`融合重算，不读取buffer。
- 当前设备已经匹配时省略重复device guard，降低启动开销。
- 非连续布局由Triton pointwise kernel完成，不调用原生Torch算子。
Ascend NPU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|
|---|---|---:|---:|
|float16|kernel comprehensive|0.747x|2.179x|
|float32|kernel comprehensive|0.610x|1.884x|
|bfloat16|kernel comprehensive|0.732x|2.081x|
Ascend comprehensive共42个case，其中37个case超过0.8x。低于0.8x的5个case只有1024或4096个元素，Torch基线为0.002000ms至0.002560ms，性能比主要受设备计时和kernel启动下限影响。
### Hygon DCU平台实现方案
Hygon DCU平台使用独立后端文件中的Triton实现，计算路径与NVIDIA版本一致。
性能优化手段：
- 连续布局使用1024元素block的手写Triton kernel。
- FP16和BF16在buffer有效时复用buffer，FP32重新计算sigmoid导数。
- 空buffer和无效buffer路径直接重算导数。
- 非连续布局使用Triton pointwise kernel，不回退到Torch。
- 默认输出通过`flag_gems.empty`分配，out变体直接写入`grad_input`。
Hygon DCU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|
|---|---|---:|---:|
|float16|kernel comprehensive|0.828x|1.025x|
|float32|kernel comprehensive|0.828x|1.163x|
|bfloat16|kernel comprehensive|0.828x|1.139x|
### T-Head PPU平台实现方案
T-Head PPU平台使用独立后端文件中的Triton实现。
性能优化手段：
- 连续布局使用1024元素block的手写Triton kernel。
- 所有支持的dtype均使用`tl.sigmoid`重新计算导数，避免额外buffer读取。
- 非连续布局使用Triton pointwise kernel，不回退到Torch。
- 默认输出通过`flag_gems.empty`分配，out变体直接写入`grad_input`。
T-Head PPU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|
|---|---|---:|---:|
|float16|kernel comprehensive|0.976x|1.534x|
|float32|kernel comprehensive|1.002x|1.483x|
|bfloat16|kernel comprehensive|0.984x|1.448x|
## 功能性测试
功能测试主要覆盖以下场景：
- `float16`、`float32`和`bfloat16`。
- 0维、1维、2维、3维、4维和5维Tensor。
- 空buffer和有效buffer。
- 连续输入和转置得到的非连续输入。
- `log_sigmoid_backward`默认返回变体。
- `log_sigmoid_backward.grad_input`输出变体及独立的`log_sigmoid_backward_out`测试marker。
- `grad_input`返回对象和数值正确性。
- 正无穷、负无穷、大负数和零等特殊值。
- log sigmoid前向自动反向传播触发已实现的`log_sigmoid_backward`。
- NVIDIA、Ascend、Hygon和T-Head后端注册路径。
功能性测试命令：
```bash
pytest -q tests/test_log_sigmoid.py -m log_sigmoid_backward
pytest -q tests/test_log_sigmoid.py -m log_sigmoid_backward_out
```
## 性能测试
core级别性能测试覆盖：
|平台|grad_output shape|self shape|buffer shape|
|---|---|---|---|
|NVIDIA/Hygon/T-Head|`(33554432,)`|`(33554432,)`|`(33554432,)`|
|Ascend|`(33554432,)`|`(33554432,)`|`(33554432,)`|
|NVIDIA/Hygon/T-Head|`(64,64)`|`(64,64)`|`(64,64)`|
|Ascend|`(4608,4608)`|`(4608,4608)`|`(4608,4608)`|
|四个平台|`(4096,4096)`|`(4096,4096)`|`(4096,4096)`|
|四个平台|`(64,512,512)`|`(64,512,512)`|`(64,512,512)`|
|四个平台|`(128,512,512)`|`(128,512,512)`|`(128,512,512)`|
comprehensive级别在core case之外增加：
|grad_output shape|self shape|buffer shape|
|---|---|---|
|`(1024,1)`|`(1024,1)`|`(1024,1)`|
|`(1024,16)`|`(1024,16)`|`(1024,16)`|
|`(1024,256)`|`(1024,256)`|`(1024,256)`|
|`(1024,4096)`|`(1024,4096)`|`(1024,4096)`|
|`(1024,65536)`|`(1024,65536)`|`(1024,65536)`|
|`(64,64,1)`|`(64,64,1)`|`(64,64,1)`|
|`(64,64,16)`|`(64,64,16)`|`(64,64,16)`|
|`(64,64,256)`|`(64,64,256)`|`(64,64,256)`|
|`(64,64,4096)`|`(64,64,4096)`|`(64,64,4096)`|
设计原因：
- NVIDIA、Hygon和T-Head使用`(64,64)`覆盖kernel启动开销占主导的小shape场景。
- Ascend使用`(4608,4608)`覆盖Triton启动延迟仍占主导且原生Torch计时稳定的场景。
- 一维、二维和三维case覆盖不同rank下的连续扁平化处理。
- 两个十亿元素默认case替换为3355万元素等价case，避免共享设备FP32单case约16GiB的内存峰值。
- comprehensive case补充短内层、长内层和不同三维布局下的启动与吞吐场景。
- `float16`、`float32`和`bfloat16`是四个平台共同支持的dtype。
性能测试命令：
```bash
pytest -q -s benchmark/test_log_sigmoid_backward.py --mode kernel --level core --warmup 100 --iter 100
pytest -q -s benchmark/test_log_sigmoid_backward.py --mode kernel --level comprehensive
```
