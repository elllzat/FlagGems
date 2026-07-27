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
- NVIDIA和Hygon仅在非空`buffer`元素数量、dtype和连续性符合要求时复用buffer，否则重新计算导数。
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
默认返回变体使用Triton pointwise动态生成路径分配输出并完成计算，不修改公共`flag_gems.empty`实现。`log_sigmoid_backward.grad_input`连续布局使用手写Triton kernel直接写入`grad_input`，其他布局通过Triton pointwise输出参数路径写入`grad_input`。四个平台的数值计算均由Triton完成，不回退到Torch。
### NVIDIA GPU平台实现方案
NVIDIA GPU平台使用Triton kernel实现。核心流程是：
1. 检查输入shape、dtype、连续性和buffer有效性。
2. 默认返回变体使用Triton pointwise动态生成路径自动分配输出。
3. FP16和BF16在有效buffer存在时复用`exp(-abs(self))`。
4. FP32和空buffer路径使用Triton sigmoid稳定重算导数。
5. out变体的连续输入使用手写Triton kernel，其他布局使用Triton pointwise kernel并直接写入`grad_input`。
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
- 默认返回变体使用Triton pointwise kernel在单次kernel中完成读取、导数计算和梯度写回。
- out变体连续布局使用1024元素block的手写Triton kernel。
- FP16和BF16复用有效buffer，避免重复指数计算。
- FP32避免额外buffer读访问，使用融合的`tl.sigmoid`重算导数。
- 空buffer路径直接重算，兼容CUDA前向返回空buffer的行为。
- 非连续布局使用Triton pointwise kernel，避免回退到Torch。
NVIDIA GPU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|平均加速比|
|---|---|---:|---:|---:|
|float16|kernel comprehensive|0.243x|1.185x|0.873x|
|float32|kernel comprehensive|0.300x|1.139x|0.893x|
|bfloat16|kernel comprehensive|0.246x|1.276x|0.874x|
NVIDIA三个dtype平均加速比的平均值为0.880x。
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
- 默认返回变体使用最大4096元素tile、最大65535 grid和一维tile优先的Triton pointwise配置。
- out变体连续布局使用8192元素block、持久化grid和`TILES_PER_PROGRAM`循环处理超大Tensor。
- 将grid数量限制为65535，避免超出Ascend启动范围。
- 使用`tl.sigmoid`融合重算，不读取buffer。
- 当前设备已经匹配时省略重复device guard，降低启动开销。
- 非连续布局由Triton pointwise kernel完成，不调用原生Torch算子。
Ascend NPU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|平均加速比|
|---|---|---:|---:|---:|
|float16|kernel comprehensive|0.195x|1.357x|0.921x|
|float32|kernel comprehensive|0.678x|1.158x|0.945x|
|bfloat16|kernel comprehensive|0.205x|1.321x|0.913x|
Ascend三个dtype平均加速比的平均值为0.926x。
### Hygon DCU平台实现方案
Hygon DCU平台使用独立后端文件中的Triton实现，计算路径与NVIDIA版本一致。
性能优化手段：
- 连续布局使用1024元素block的手写Triton kernel。
- FP16和BF16在buffer有效时复用buffer，FP32重新计算sigmoid导数。
- 空buffer和无效buffer路径直接重算导数。
- 非连续布局使用Triton pointwise kernel，不回退到Torch。
- 默认输出由Triton pointwise动态生成路径分配，out变体直接写入`grad_input`。
Hygon DCU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|平均加速比|
|---|---|---:|---:|---:|
|float16|kernel comprehensive|0.271x|0.893x|0.786x|
|float32|kernel comprehensive|0.559x|1.000x|0.943x|
|bfloat16|kernel comprehensive|0.283x|0.955x|0.839x|
Hygon三个dtype平均加速比的平均值为0.856x。
### T-Head PPU平台实现方案
T-Head PPU平台使用独立后端文件中的Triton实现。
性能优化手段：
- 连续布局使用1024元素block的手写Triton kernel。
- 所有支持的dtype均使用`tl.sigmoid`重新计算导数，避免额外buffer读取。
- 非连续布局使用Triton pointwise kernel，不回退到Torch。
- 默认输出由Triton pointwise动态生成路径分配，out变体直接写入`grad_input`。
T-Head PPU性能对比Torch baseline：
|dtype|测试模式|最低加速比|最高加速比|平均加速比|
|---|---|---:|---:|---:|
|float16|kernel comprehensive|1.006x|1.192x|1.053x|
|float32|kernel comprehensive|0.994x|1.070x|1.011x|
|bfloat16|kernel comprehensive|1.013x|1.159x|1.052x|
T-Head三个dtype平均加速比的平均值为1.039x。
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
|grad_output shape|self shape|buffer shape|
|---|---|---|
|`(33554432,)`|`(33554432,)`|`(33554432,)`|
|`(4096,4096)`|`(4096,4096)`|`(4096,4096)`|
|`(64,512,512)`|`(64,512,512)`|`(64,512,512)`|
|`(128,512,512)`|`(128,512,512)`|`(128,512,512)`|
comprehensive级别在core case之外增加：
|grad_output shape|self shape|buffer shape|
|---|---|---|
|`(1024,4096)`|`(1024,4096)`|`(1024,4096)`|
|`(1024,65536)`|`(1024,65536)`|`(1024,65536)`|
|`(64,64,256)`|`(64,64,256)`|`(64,64,256)`|
|`(64,64,4096)`|`(64,64,4096)`|`(64,64,4096)`|
设计原因：
- 四个平台使用完全相同的shape和dtype集合。
- 性能测试保留不少于1048576个元素的吞吐场景，小shape由功能测试覆盖。
- 一维、二维和三维case覆盖不同rank下的连续扁平化处理，元素规模覆盖1M、4M、16M、32M和67M。
- 两个十亿元素默认case替换为3355万元素等价case，避免共享设备FP32单case约16GiB的内存峰值。
- `float16`、`float32`和`bfloat16`是四个平台共同支持的dtype。
- 每个dtype先对8个shape的加速比求算术平均，再对三个dtype平均加速比求算术平均，得到平台最终加速比。
性能测试命令：
```bash
pytest -q -s benchmark/test_log_sigmoid_backward.py --mode kernel --level core --warmup 100 --iter 100
pytest -q -s benchmark/test_log_sigmoid_backward.py --mode kernel --level comprehensive
```
