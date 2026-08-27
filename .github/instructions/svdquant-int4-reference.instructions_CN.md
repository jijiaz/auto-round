---
description: "用于实现或运行 FLUX.1-dev SVDQuant INT4 PyTorch reference 精度实验、block size 扫描、model-level 质量评估及 go/no-go 分析。"
name: "SVDQuant INT4 Reference 精度实验"
applyTo: "test/e2e/test_cuda/svdquant_int4_reference/**"
---

# SVDQuant INT4 Reference 精度实验流程

## 实验目的

在投入 packing、export 或 runtime kernel 开发之前，确定满足 model-level
质量要求的 INT4 scale 共享 block size 上限。

Reference 路径保持为纯 PyTorch 实现。复用 AutoRound 现有的 SVDQuant
transform、量化注册机制、diffusion calibration 和 diffusion 质量指标。
不能用 kernel 吞吐量或 export round trip 代替 reference 精度结果。

主要实验对象为 FLUX.1-dev，因为它是目前 AutoRound 中唯一完成
SVDQuant 端到端验证的模型。

## 现有参考实现与结果

应复用以下实现，不要重复开发：

- SVD 分解与 residual iteration：
  `auto_round/algorithms/transforms/svdquant/residual.py`
- 结构变换与 projection grouping：
  `auto_round/algorithms/transforms/svdquant/apply.py`
- residual 与 low-rank 分支的 PyTorch forward：
  `auto_round/algorithms/transforms/svdquant/wrapper.py`
- 已注册的 INT 和 MXFP QDQ：
  `auto_round/data_type/`
- Diffusion 图片生成及 CLIP、CLIP-IQA、ImageReward 评估：
  `auto_round/compressors/diffusion/eval.py`
- 已有 SVDQuant 配置与质量结果：
  `docs/step_by_step_CN.md` 和 `docs/svdquant_details_CN.md`

仓库中已有的 FLUX.1-dev 参考值为：

| 配置 | CLIP | CLIP-IQA | ImageReward |
|---|---:|---:|---:|
| BF16 | 26.0189 | 0.954360 | 1.018340 |
| MXFP4，smooth + SVDQuant + SignRound | 26.1039 | 0.962655 | 1.021020 |
| MXFP4，no smooth + SVDQuant + SignRound | 26.0727 | 0.959363 | 1.002380 |
| MXFP4，smooth + SVDQuant + RTN | 25.9719 | 0.947763 | 0.939392 |
| MXFP4，no smooth + SVDQuant + RTN | 25.9624 | 0.946939 | 0.934579 |

这些数值只能作为历史参考，不能默认当前环境可以直接复现。每次完整
INT4 扫描都必须在同一环境中重新运行 BF16 和 MXFP4 control。

## 固定实验配置

除非某一阶段明确研究对应变量，否则固定以下配置：

- 模型：`black-forest-labs/FLUX.1-dev`
- 模型及 low-rank dtype：BF16
- Low-rank rank：32
- Calibration samples：128 条 COCO2017 captions
- Inference steps：50
- SignRound tuning iterations：200
- Residual iterations：20
- 指标：CLIP、CLIP-IQA、ImageReward
- Prompt 文件及顺序、生成 seed、图片尺寸、guidance scale、scheduler、
  batch size 和软件版本

BF16、MXFP4 control 和所有 INT4 block size 必须使用相同 prompt 与逐
prompt seed。每次运行都要保存最终生效的配置和代码/依赖版本。如果要
声称与历史数值直接可比，必须先补齐并匹配尚未记录的配置，尤其是评估
prompt 集合、sample limit、图片尺寸、guidance scale 和 seed。

## 步骤 1：冻结 INT4 数值规范

**目标：**确保 PyTorch QDQ 精确模拟计划开发的 kernel，而不是笼统的
INT4 定义。

需要明确、记录并测试：

- Signed code 范围，例如 `[-8, 7]` 或 `[-7, 7]`
- Scale 计算公式及 scale dtype
- Rounding mode 与 clamp 顺序
- Weight 和 activation 的 block 维度
- Weight 与 activation block size 是否绑定
- Padding 和不完整 block 的行为
- 全零 block、NaN、Inf、underflow 和 overflow 行为
- Scale 是仅沿 K 维共享，还是按 M-by-K 二维 block 共享

只有目标 kernel 语义与 `int.py` 一致时，才复用其中的 K 维对称 INT4。
如果使用 M-by-K scale，应新增并注册独立的 `block_int` PyTorch QDQ。
不能为了本实验修改现有 `int` datatype 的语义。

**完成标准：**单元测试覆盖人工可计算 tensor、全零 block、维度不整除、
BF16/FP16 scale materialization 以及全部候选 block shape。

## 步骤 2：验证 SVDQuant workflow 接入

**目标：**确认 INT4 QDQ 仅作用于 SVDQuant residual 分支，而 low-rank
分支保持 BF16。

使用以下算法组合构造 AutoRound：

```python
alg_configs=[
    SVDQuantConfig(
        rank=32,
        smooth_enabled=...,
        residual_iters=20,
        low_rank_dtype="bf16",
        model_adapter="flux",
    ),
    RTNConfig(disable_opt_rtn=True),
]
```

为每个候选 block size 构造 `QuantizationScheme` 并直接传给 AutoRound。
调用 `quantize()` 后评估内存中的 pipeline。不要调用
`svdquant_nunchaku` exporter；该 exporter 按设计仅接受 group size 32
的 MXFP4，不属于本次 reference 实验。

抽查具有代表性的 transformed layer：

- 原 Linear 已替换为 `SVDQuantLinear`
- Residual weight 已执行 INT4 QDQ
- 研究 W4A4 时，activation 使用 dynamic INT4 QDQ
- `lora_down` 和 `lora_up` 保持 BF16
- 输出均为有限值，且模型输出 shape 不变

**完成标准：**切片 FLUX 的单元/集成测试完成内存 forward，并与独立计算
的 `Q(R) + U @ V` reference 数值一致。

## 步骤 3：复现 control

**目标：**在扫描 INT4 之前，证明当前环境和评估工具能够产生可信的
model-level 结果。

在同一环境中运行：

1. BF16
2. MXFP4，smooth + SVDQuant + SignRound
3. MXFP4，no smooth + SVDQuant + SignRound
4. 可选运行两个已有 RTN 配置，用于诊断比较

所有配置都使用 `diffusion_eval()` 计算三个质量指标。可以保留
`test_diffusion_quantize_e2e.py` 中的图片 sanity check，但绝不能把图片
shape、dtype 或标准差当作质量验收指标。

将新 control 与上述历史值比较。若差异明显，应先查明原因，再执行昂贵的
完整扫描；不能通过临时放宽阈值掩盖 control 不一致。

**完成标准：**BF16 和至少一个 MXFP4 control 在重复运行中保持稳定，并
记录其与历史结果存在差异时的原因。

## 步骤 4：粗粒度扫描 block size

**目标：**使用较低成本定位精度转折区间，再进入完整 model-level 矩阵。

对于沿 K 维共享 scale 的方案，可从以下 kernel 合法候选开始：

```text
16, 32, 64, 128, 256, per-channel
```

对于二维 scale，应改为扫描硬件支持的 M-by-K shape。先固定 M 寻找 K
上限，再围绕通过的 K 值扫描 M。

这一阶段可以使用固定的小 prompt 子集和较少 inference steps，但结果只能
用于筛选，不能作为最终精度。所有候选必须使用相同 prompt 和 seed。一次
只改变一个变量：

1. 固定 activation block size，扫描 weight block size
2. 固定 weight block size，扫描 activation block size
3. 在通过边界附近扫描少量联合组合

**完成标准：**找到最可能通过的最大 block size，以及第一个明显失败的
block size，交由完整评估确认。

## 步骤 5：完整 model-level 扫描

**目标：**按照已有 SVDQuant 质量标准确定可支持的 block size 边界。

使用固定实验配置运行新的 BF16 control、MXFP4 control 及候选 INT4
block size。至少完整评估粗扫边界两侧的两个 size，并增加一个更小的
control size。资源允许时使用多个配对 seed，报告均值、标准差和最差结果。

主要可行性配置采用 smooth SVDQuant + SignRound，与 AutoRound 现有最佳
结果一致。如果产品目标是 RTN，还必须在最终候选边界运行 no-smooth + RTN；
不能仅根据 SignRound 结果推断 RTN 可行性。

每条结构化记录对应一个 model、method、block size、seed 和 metric。记录
原始指标、相对本次 BF16 的 delta、相对本次 MXFP4 control 的 delta、
wall time、peak memory 和完整配置。

**完成标准：**每个最终候选都具有来自同一 prompt/seed 集合的 CLIP、
CLIP-IQA 和 ImageReward 完整结果。

## 步骤 6：应用精度门槛

**目标：**使用实验前确定的标准选择最大 block size，不能看过结果后修改
验收条件。

主要质量 control 使用同一轮、同配置的 MXFP4 结果。候选只有在汇总结果的
三个指标都通过预注册阈值，且重复实验中没有明显异常失败时才算通过。

在正式 INT4 阈值获批前，暂时使用现有已验证 MXFP4 RTN envelope 作为保守
下限：

```text
CLIP        >= 25.9624
CLIP-IQA    >= 0.946939
ImageReward >= 0.934579
```

该 envelope 对应仓库现有 MXFP4 SVDQuant RTN 相对 BF16 的最差结果。
一旦 control 可以稳定复现，应优先采用相对本次 control 的门槛。完整扫描
开始前，必须在实验配置中记录已批准的绝对值和相对值门槛。

Block size 上限是满足以下全部条件的最大已测 size：

- 三个 model-level 质量门槛全部通过
- 所有输出均为有限值，图片 sanity check 通过
- 在要求的 seed 上可复现
- 预期单调的所有更小硬件合法 size 均已通过；若出现非单调结果，已重跑并解释

不能用一个明显更好的无关指标平均掉另一个失败指标。

## 步骤 7：Kernel go/no-go 决策

**目标：**判断通过精度要求的 block size 是否具有足够系统收益，值得进入
产品开发。

对每个通过的 block size 估算：

- INT4 weight bytes 与 scale metadata bytes
- Dynamic activation scale 流量
- Residual INT4 GEMM 成本
- BF16 low-rank down/up GEMM 及中间 tensor 成本
- 硬件 tile 和 memory layout 兼容性
- 预期 bandwidth、occupancy、latency 和 throughput 收益
- Export、loader、backend 与 parity test 的开发成本

只有至少一个通过精度的 block size 同时具备硬件效率，并且相对现有 MXFP4
路径能显著改善目标 workload 时，才建议开发 kernel。小 block 即使精度通过，
如果 scale 开销抵消了内存或吞吐收益，也应判定为“精度可行、产品 no-go”。

## 必须产出的结果

生成的 checkpoint 和图片不能提交到 git。工作目录只保存可复用脚本、测试
和小型配置文件。实验必须产出：

- 机器可读的实验 manifest
- 每个 seed 的原始指标记录
- 汇总后的 block size 对比表
- 最大通过 block size
- 包含性能假设的 go/no-go 结论

新增 Python、YAML 或 shell 文件必须包含仓库 Apache 2.0 header。提交任何
Markdown 修改时，必须同步维护对应的 `_CN` 翻译。
