# rnn_tanh开发文档

## 接口介绍

### 算子名字

本次开发包含两个接口：

- `rnn_tanh.input`
- `rnn_tanh.data`

对应的算子配置ID和测试marker为：

|接口|配置ID|测试marker|
|---|---|---|
|`rnn_tanh.input`|`rnn_tanh`|`rnn_tanh`|
|`rnn_tanh.data`|`rnn_tanh_data`|`rnn_tanh`|

两个接口分别导出为 `flag_gems.rnn_tanh` 和 `flag_gems.rnn_tanh_data`。选择 `rnn_tanh` 配置组时同时注册两种ATen重载。

### 输入输出参数

#### rnn_tanh.input

接口形式：

```python
torch.ops.aten.rnn_tanh.input(
    input, hx, params, has_biases, num_layers,
    dropout, train, bidirectional, batch_first
)
```

|参数|方向|类型|说明|
|---|---|---|---|
|input|输入|Tensor|形状为`(T,B,I)`，batch_first=True时为`(B,T,I)`|
|hx|输入|Tensor|初始隐藏状态，形状为`(L*D,B,H)`|
|params|输入|Tensor[]|按层和方向排列的weight_ih、weight_hh及可选bias_ih、bias_hh|
|has_biases|输入|bool|是否包含输入偏置和隐藏状态偏置|
|num_layers|输入|int|层数L，必须大于零|
|dropout|输入|float|层间dropout概率，范围为[0,1]|
|train|输入|bool|是否启用训练模式下的层间dropout|
|bidirectional|输入|bool|是否双向；D为2或1|
|batch_first|输入|bool|是否以batch作为input和output的第一维|
|output|输出|Tensor|最后一层全部时间步输出，形状为`(T,B,D*H)`或`(B,T,D*H)`|
|hidden|输出|Tensor|各层、各方向的最终隐藏状态，形状为`(L*D,B,H)`|

约束：

- T、B、I、H分别为序列长度、批大小、输入特征数和隐藏特征数，本文性能测试使用正尺寸。
- input、hx和params要求使用相同设备及兼容的浮点dtype；本次验证FP16、FP32和BF16。
- 每层每方向包含2个权重Tensor，有偏置时额外包含2个偏置Tensor。
- 第一层weight_ih形状为`(H,I)`，后续层为`(H,D*H)`；weight_hh为`(H,H)`，每个偏置为`(H,)`。
- dropout仅在train=True、层数大于1时作用于层间输入，不作用于最后一层输出之后。
- 输入布局通过stride读取，batch_first输出由Triton复制kernel转换；本次测试不代表覆盖所有任意stride组合。

#### rnn_tanh.data

接口形式：

```python
torch.ops.aten.rnn_tanh.data(
    data, batch_sizes, hx, params, has_biases,
    num_layers, dropout, train, bidirectional
)
```

|参数|方向|类型|说明|
|---|---|---|---|
|data|输入|Tensor|PackedSequence数据，形状为`(sum(batch_sizes),I)`|
|batch_sizes|输入|Tensor|CPU上的一维int64元数据，表示每个时间步的有效batch数|
|hx|输入|Tensor|初始隐藏状态，形状为`(L*D,batch_sizes[0],H)`|
|params|输入|Tensor[]|与input重载相同的按层、方向排列的权重和可选偏置|
|has_biases|输入|bool|是否包含偏置|
|num_layers|输入|int|层数L|
|dropout|输入|float|层间dropout概率，范围为[0,1]|
|train|输入|bool|是否启用训练dropout|
|bidirectional|输入|bool|是否双向|
|output|输出|Tensor|打包输出，形状为`(sum(batch_sizes),D*H)`|
|hidden|输出|Tensor|各层、方向的最终隐藏状态，形状与hx相同|

约束：

- batch_sizes必须非空、为正数且非递增，其总和必须等于data的行数。
- hx的batch维必须等于batch_sizes[0]。
- batch_sizes只用于主机端计算启动范围和偏移，设备端张量计算由Triton完成。
- PackedSequence的排序、反排序由调用方处理；该底层接口不接收sorted_indices或unsorted_indices。
- 本次packed前向、反向和dispatch功能测试使用FP32；不将其扩展解释为packed低精度性能验收。

## 算子功能介绍

`rnn_tanh`实现使用tanh激活的多层Elman RNN，并提供前向和反向传播。

```text
h_t = tanh(x_t @ weight_ih.T + bias_ih
           + h_previous @ weight_hh.T + bias_hh)
output = 最后一层各时间步、各方向隐藏状态的拼接
hidden = 各层、各方向的最终隐藏状态
```

无偏置时省略bias项。反向方向按时间倒序递推；层间输入由前一层输出及可选dropout构成。

反向传播使用tanh导数并沿时间链传递梯度，其中g_t包含当前步从输出、上层及最终隐藏状态传入的梯度：

```text
delta_t = (g_t + delta_next @ weight_hh) * (1 - h_t * h_t)
grad_x_t = delta_t @ weight_ih
```

简单示例：

```python
import torch
import flag_gems

device = flag_gems.device
module = torch.nn.RNN(32, 32, nonlinearity="tanh").to(device)
input = torch.randn(16, 4, 32, device=device)
hx = torch.randn(1, 4, 32, device=device)
with flag_gems.use_gems():
    output, hidden = torch.rnn_tanh(
        input, hx, tuple(module._flat_weights),
        True, 1, 0.0, False, False, False
    )
```

输出shape分别为`(16,4,32)`和`(1,4,32)`。在T-Head当前ACDNN环境下，构造reference模块前需要与测试文件一致地禁用不支持tanh RNN的cudnn兼容快速路径。

## 实现方案

### GPU 平台实现方案

GPU类平台使用通用文件`src/flag_gems/ops/rnn_tanh.py`中的Triton kernel。核心流程是：

1. 检查输入rank、层数、参数数量、隐藏状态层维和dropout范围。
2. 按平台、dtype和特征大小选择持久化dot、向量归约或输入投影与递推分离路径。
3. 逐层、逐方向计算输入投影和隐藏状态递推，将偏置、激活及可选dropout融合到相应kernel。
4. 写入各时间步输出和最终隐藏状态，必要时转换batch_first布局。
5. 反向通过BPTT及梯度归约kernel计算input、hx、权重和偏置梯度；packed路径按各步有效batch处理。

实现流程：

```text
input/data + hx + params
   |
   |--校验参数、解析层/方向/packed元数据
   |
   |--按平台、dtype、特征大小选择Triton路径
   |      |--持久化dot或vector递推
   |      |--输入投影 + 持久化递推
   |
   |--写入output和hidden，按需转换布局
   |
   |--autograd调用Triton BPTT和参数梯度kernel
```

性能优化手段：

- 持久化路径在kernel内部遍历时间步，复用权重及隐藏状态，减少主机端逐步启动。
- NVIDIA较小隐藏状态使用vector路径，较大的BF16隐藏状态使用dot路径；矩阵累加采用FP32。
- NVIDIA非BF16且H=128的适用形状使用输入投影与持久化递推分离路径。
- MetaX的适用FP32、H=128形状复用输入投影与持久化递推分离路径。
- T-Head和Hygon共享通用文件中的持久化路径选择逻辑；本次未复测Hygon。
- 启动函数不调用原生Torch的mm、matmul、addmm、stack、cat或tanh来拼装RNN；张量分配、元数据读取和设备/随机数管理不属于计算回退。
- 训练dropout使用FlagGems现有Philox随机数管理，保证重复设置manual_seed时可复现，不承诺与Torch生成位级相同的随机mask。

T-Head、Hygon、MetaX和Iluvatar没有保留相同的runtime副本；按runtime优先、通用ops回退的dispatch规则复用该文件。Ascend差异化kernel同样保留在通用文件中并按vendor选择。代码复用不等于所有平台均已完成验证，Hygon和Iluvatar按要求排除本次复测。

GPU类平台性能对比原生Torch baseline：

|平台|测试模式|FP16均值|FP32均值|BF16均值|最终均值|最低加速比|最高加速比|
|---|---|---:|---:|---:|---:|---:|---:|
|NVIDIA|kernel comprehensive|0.404x|0.298x|4.546x|1.749x|0.283x|4.937x|
|NVIDIA|operator comprehensive|0.439x|0.435x|4.525x|1.799x|0.285x|5.357x|
|T-Head|kernel comprehensive|35.907x|23.535x|35.719x|31.721x|19.667x|45.016x|
|T-Head|operator comprehensive|20.796x|16.763x|20.280x|19.280x|9.335x|29.544x|
|Hygon|kernel comprehensive|未复测|未复测|未复测|不纳入验收|—|—|
|Hygon|operator comprehensive|未复测|未复测|未复测|不纳入验收|—|—|
|MetaX|kernel comprehensive|11.245x|6.397x|4.131x|7.257x|1.080x|16.356x|
|MetaX|operator comprehensive|7.259x|5.721x|9.986x|7.655x|1.101x|15.695x|
|Iluvatar|kernel comprehensive|未复测|未复测|未复测|不纳入验收|—|—|
|Iluvatar|operator comprehensive|未复测|未复测|未复测|不纳入验收|—|—|

T-Head在测试文件中禁用不支持tanh RNN的ACDNN快速路径，baseline是Torch通用分解实现，因此31.721x/19.280x不是相对ACDNN融合RNN的加速比。MetaX结果仅针对当前设备原生Torch路径，不可直接解释为相对其他平台cuDNN的同等倍数。NVIDIA的FP16和FP32均值小于0.8x，整体均值达标主要来自BF16收益。

packed重载仅完成本次四平台的功能验证，未单独测量性能：

|平台|接口|FP16均值|FP32均值|BF16均值|最终均值|最低加速比|最高加速比|
|---|---|---:|---:|---:|---:|---:|---:|
|NVIDIA、T-Head、MetaX|`rnn_tanh_data`|未测|未测|未测|未测|—|—|
|Hygon、Iluvatar|`rnn_tanh_data`|未复测|未复测|未复测|不纳入验收|—|—|

### NPU 平台实现方案

NPU平台使用同一文件内的Ascend专用Triton路径，不回调torch_npu原生RNN。激活kernel复用FlagGems已实现的`tanh_kernel._scalar_fn`，不是在启动函数中调用Torch tanh。

实现流程：

```text
NPU input + hx + params
   |
   |--按当前调用获取stream，解析层和方向
   |
   |--Triton批量输入投影
   |
   |--选择递推路径
   |      |--常规：递推矩阵计算 + FlagGems tanh标量函数
   |      |--BF16 H=128：每chunk融合3个时间步
   |
   |--复用适用的已编译中间步launch runner
   |
   |--写入output和hidden，支持Triton反向
```

性能优化手段：

- 预计算各时间步的输入投影，降低递推阶段重复计算。
- 对H=128的常规递推使用K=128归约tile，减少两段K=64计算的中间搬运。
- 常规激活使用不超过512元素的tile；128元素tile尝试没有整体收益，未保留。
- BF16、H=128适用路径每个chunk融合3个时间步，batch tile为8；首块、末块及不足3步的尾块分别处理。
- 每次算子调用只解析一次当前stream，并复用已绑定grid的中间步runner，减少Python启动开销；不跨调用缓存stream。
- 仅复用首步/尾步标记与动态索引specialization匹配的中间kernel，避免0、1索引特化影响结果。
- 使用FP32 H=64、BF16 H=128非默认stream测试以及多层双向长序列测试覆盖runner复用和chunk尾部。
- 优化保留原始benchmark计时实现；不通过修改公共benchmark、profiler或统计方式提高结果。

Ascend性能对比原生Torch baseline：

|测试模式|FP16均值|FP32均值|BF16均值|最终均值|最低加速比|最高加速比|
|---|---:|---:|---:|---:|---:|---:|
|kernel comprehensive|0.943x|0.879x|0.629x|0.817x|0.253x|1.001x|
|operator comprehensive|0.822x|0.819x|1.668x|1.103x|0.607x|3.460x|

表中均值来自3轮完整复测，最低/最高为全部27条记录的极值。三轮kernel算术平均分别为0.815x、0.810x、0.826x；operator分别为1.120x、1.101x、1.088x，两种模式每轮均超过0.8x。kernel裕量较小，不保证共享设备上任意后续测量都保持相同数值。

当前安装的Ascend profiler对选中的CSV kernel记录取平均，而非对一次完整RNN调用的所有kernel耗时求和；返回字段为Duration(us)，但公共benchmark表头标为ms。本报告保留原始口径，只比较同一口径的延迟比。Ascend kernel加速比不是完整调用的端到端设备加速比，也不宜与其他平台kernel绝对延迟横向比较。operator是包含主机调度和同步的整次调用计时。packed性能未单独测量。

## 功能性测试

功能测试主要覆盖以下场景：

- FP16、FP32和BF16前向dtype。
- 单层/双层、有偏置/无偏置、单向/双向和batch_first布局。
- comprehensive性能形状对应的全部9组前向正确性。
- BF16中等隐藏状态、17/18/20步双层双向序列，覆盖chunk完整块和尾块。
- FP32 H=64与BF16 H=128的非默认stream调用。
- FP32 dense反向的输入、初始隐藏状态、权重和偏置梯度。
- packed FP32单层/双层、单向/双向的前向、反向及ATen dispatch。
- H=320大隐藏状态回退路径、dropout=1和manual_seed可复现性。
- 禁止启动函数调用原生Torch数学算子拼装RNN的回归测试。

主功能测试shape为：

|shape|rank|覆盖场景|
|---|---:|---|
|`(5,3,16)`，H=24；batch_first时`(3,5,16)`|3|前向层数、偏置、方向及布局组合|
|`(16,4,32)`、`(32,8,64)`、`(64,16,128)`|3|全部comprehensive形状和三种dtype|
|`(17,4,64)`、`(17,4,128)`|3|非默认stream，H分别为64和128|
|`(3,2,64)`，H=128|3|BF16中等隐藏状态|
|`(17/18/20,3,64)`，H=128|3|BF16双层双向及chunk尾块|
|`(4,2,16)`，H=16|3|双层双向FP32反向|
|`(3,2,32)`，H=320|3|大隐藏状态回退|
|`(5,3,16)`，H=16|3|训练dropout|
|`(12,16)`与`(8,16)`的packed data|2|packed前向/反向及dispatch|

功能测试使用设备原生Torch作为reference；T-Head使用其可运行的通用分解reference。本次未建立CPU reference覆盖：当前测试文件在`--ref=cpu`时会跳过设备测试，不能将skip当作通过。

在同一源代码提交`d5b375a69bb4df7e12aeed38498723627eee19ea`上，NVIDIA、Ascend、T-Head、MetaX均为40 passed。40是功能测试项数；性能每模式为9个不同case，并不是300个不同case。

功能性测试命令：

```bash
python -m pytest -q tests/test_rnn_tanh.py -m rnn_tanh
python -m pytest -q tests/test_rnn_tanh.py -m rnn_tanh --collect-only
```

## 性能测试

性能测试覆盖：

|平台|input shape|hx shape|params shape|
|---|---|---|---|
|NVIDIA、Ascend、T-Head、MetaX|`(16,4,32)`|`(1,4,32)`|两份`(32,32)`权重和两份`(32,)`偏置|
|NVIDIA、Ascend、T-Head、MetaX|`(32,8,64)`|`(1,8,64)`|两份`(64,64)`权重和两份`(64,)`偏置|
|NVIDIA、Ascend、T-Head、MetaX|`(64,16,128)`|`(1,16,128)`|两份`(128,128)`权重和两份`(128,)`偏置|

设计原因：

- 四个平台使用相同3个shape、3种dtype及原始comprehensive参数，未删减慢case或替换计时框架。
- 固定单层、单向、有偏置、dropout=0、train=False、batch_first=False，输入特征数等于隐藏特征数。
- T、B、H同步增大，用于观察小RNN启动开销及较长序列的递推成本；该集合不代表所有实际模型。
- 每平台每模式9个不同case。Ascend重复3轮，得到每模式27条记录；其他平台各1轮，总计108条性能记录。
- 先对同一dtype的3个shape加速比取算术平均，再平均3个dtype均值，等价于9个case等权平均。重复轮次也等权平均，不使用几何平均或总延迟之比。
- 加速比由日志中6位小数延迟相除后计算，表中最终显示3位小数；不对已显示为3位小数的speedup再求均值。
- 验收要求是每平台kernel、operator各自的整体算术平均超过0.8x，不是每个dtype、shape都超过0.8x。
- 仅验收dense input前向性能，packed及反向未测性能；Hygon和Iluvatar按要求未复测。
- 2026-08-27验证设备为NVIDIA H20、Ascend 910B、T-Head PPU-ZW810E和MetaX C550。分别使用登录文件指定的容器或venv；平头哥切换主仓分支、沐曦启动已有容器均经授权。
- NVIDIA使用设备1，Ascend使用NPU1，T-Head和MetaX使用设备0。测试前检查设备占用；远端仅拉取本地提交，不修改依赖。
- NVIDIA环境Torch 2.11.0a0+eb65b36914.nv26.02/Triton 3.6；Ascend Torch 2.6.0+cpu及torch_npu/Triton 3.2；T-Head Torch 2.9.0/Triton 3.5；MetaX Torch 2.8.0+metax3.7.0.7/Triton 3.0。
- `benchmark/base.py`保持分支原始计时实现，blob为`276a5ff28d161d03bd20d4f793f1f9a63e59c177`。参数warmup/iter按原始框架解释，operator的实际循环次数由框架估算。

性能测试命令：

```bash
python -m pytest -s benchmark/test_rnn_tanh.py --mode kernel --level comprehensive --metrics latency_base --metrics latency --metrics speedup --warmup 100 --iter 100 -q
python -m pytest -s benchmark/test_rnn_tanh.py --mode operator --level comprehensive --metrics latency_base --metrics latency --metrics speedup --warmup 100 --iter 100 -q
```

复测原始日志保存在本地`output/rnn_tanh_retest_20260827/`：`ascend_stream_cached_validation.log`、`nvidia_final_d5b375a6.log`、`thead_final_d5b375a6.log`和`metax_final_d5b375a6.log`；逐条延迟、加速比及汇总见同目录`four_platform_final_results.json`。
