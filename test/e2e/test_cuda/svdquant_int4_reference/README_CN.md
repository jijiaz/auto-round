# FLUX.1-dev SVDQuant INT4 参考精度研究

用于确定 SVDQuant 在 FLUX.1-dev 上仍能满足模型级质量要求的最大 INT4 共享缩放块
大小的 PyTorch 参考研究，需在投入打包、导出或内核开发之前完成。

本目录中的所有内容都保持在 PyTorch 中，并复用 AutoRound 的生产实现：SVDQuant
变换（`auto_round/algorithms/transforms/svdquant/`）、已注册的量化/反量化函数
（`auto_round/data_type/`）以及扩散模型评测指标
（`auto_round/compressors/diffusion/eval.py`）。不使用 ARK INT4 内核，也不用
`svdquant_nunchaku` 导出往返来替代参考精度结果。

## 目录内容

| 文件 | 阶段 | 用途 |
|---|---|---|
| `contract.py` | 1、6 | 冻结的 INT4 数值契约、固定实验设置、候选块大小、预先登记的验收门限、运行清单 |
| `test_int4_reference_contract.py` | 1、2 | 契约测试与 SVDQuant 工作流集成测试（CPU，切片 FLUX 块） |
| `proxy_block_sweep.py` | 4 | 低成本的层级筛查扫描，用于定位过渡区间 |
| `run_reference_sweep.py` | 3、5 | 全新的 BF16/MXFP4 对照组以及 FLUX.1-dev 上的模型级 INT4 块大小扫描 |
| `aggregate_results.py` | 6 | 应用预先登记的门限并选出最大通过块大小 |

生成的检查点、图像与记录文件不纳入 git（忽略 `records/` 与 `images/`）。

## 步骤 1：INT4 数值契约

参考 QDQ 必须精确表示所提议的内核，而不是对 INT4 的通用解释，因此契约在
`contract.py::INT4_CONTRACT` 中显式声明，并由两个已注册函数实现：

| 属性 | 取值 |
|---|---|
| 编码范围 | 有符号 `[-8, 7]`（full range） |
| 缩放 | `scale = -sign(dominant) * max(abs(w_min), abs(w_max)) / 8`，幅值下限钳位到 `1e-5` |
| 缩放 dtype | 默认 FP16，使用前实体化 |
| 舍入 | 四舍六入五成双，然后钳位（钳位不会反馈影响缩放） |
| 权重轴 | `(out_features, in_features)` 上的 `(M, K)`；`K` 为规约轴 |
| 激活轴 | 一维 `K` 分组，逐 token 动态 |
| 耦合关系 | 权重与激活块大小相互独立，分别单独扫描 |
| 填充 | 部分块补零；填充不会扩大块范围，并在输出时裁剪 |
| 全零块 | 缩放钳位为 `+1e-5`，所有编码为 0，输出恰为 0 |
| 缩放布局 | `block_int_sym` 为二维 `(ceil(M_out/M), ceil(K/K_block))`；`int_sym` 为一维 `K` 分组 |

`auto_round/data_type/int.py` 新增了 `block_int_sym`（别名 `block_int_sym`、
`block_int4_sym`）用于 M×K 缩放。当块高度为 1 时，它与现有 `int_sym`
**逐比特一致**，因此 K 轴布局与块布局属于同一个契约而非两个契约，且现有 `int`
数据类型的语义未被改动。当 `group_size` 为元组时，`get_quant_func` 现在会在
`rtn_*`/`opt_rtn_*` 别名**之前**解析 `block_*` 实现，因为这些别名的 K 轴语义与
二维块契约并不匹配。

单元测试：`test/unit/test_cpu/data_type/test_block_int.py`（精确的手工计算张量、
全零块、不可整除维度、FP16/BF16/FP32 缩放实体化，以及全部候选块形状）。

## 步骤 2：工作流集成

`test_int4_reference_contract.py` 使用如下配置构建 AutoRound 的算法组合器

```
SVDQuantConfig(rank=..., smooth_enabled=..., residual_iters=..., low_rank_dtype="bf16", model_adapter="flux")
RTNConfig(disable_opt_rtn=True)
```

以及由 `contract.Candidate.to_scheme()` 生成的 `QuantizationScheme`，然后在切片
的 FLUX 形状块上检查：该层被替换为 `SVDQuantLinear`、残差权重精确落在 INT4 网格
上（QDQ 幂等）、`lora_down`/`lora_up` 保持 BF16 且秩符合要求、前向输出有限且形状
不变，并且等效权重与独立计算的 `Q(R) + U @ V` 一致。

在 CPU 上运行：

```bash
PYTHONPATH=. pytest test/e2e/test_cuda/svdquant_int4_reference/ -q
PYTHONPATH=. pytest test/unit/test_cpu/data_type/test_block_int.py -q
```

## 步骤 4：粗粒度筛查扫描（已执行）

`proxy_block_sweep.py` 使用生产实现的 `truncated_svd`、
`iterate_residual_decomposition`、`rtn_qdq_residual` 原语（秩 32、20 次残差迭代），
在 FLUX.1-dev 的六种不同线性层形状上测量 `Q(R) + U @ V` 相对 BF16 的**层输出**
NMSE。

筛查规则：MXFP4 group 32 是唯一在 FLUX.1-dev 上有公开模型级数据的 4 bit SVDQuant
配置，并且即使其最差变体（no smooth + RTN）也满足验收门限。当某个 INT4 配置的层
NMSE **不劣于对应的 MXFP4 group-32 对照组**时即视为筛查通过（W4A16 候选对比
W4A16 对照组，W4A4 候选对比 W4A4 对照组）。

```bash
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/proxy_block_sweep.py \
    --shrink 4 --residual-iters 20 --scan-activations --output records/proxy.json
```

结果（六种形状上的几何平均，相对对应 MXFP4 group-32 对照组的比值，越低越好）：

| 权重缩放布局 | 每个缩放覆盖元素数 | 相对 MXFP4 g32 的 NMSE 比值 | 筛查结论 |
|---|---:|---:|---|
| K 轴 16 | 16 | 0.43 | 通过 |
| K 轴 32 | 32 | 0.55 | 通过 |
| K 轴 64 | 64 | 0.68 | 通过 |
| K 轴 128 | 128 | 0.80 | 通过 |
| K 轴 256 | 256 | 0.93 | 通过 |
| K 轴 per-channel（3072+） | >= 3072 | 1.22 | 失败 |
| 块 16x16 | 256 | 1.02 | 失败（临界） |
| 块 16x32 | 512 | 1.18 | 失败 |
| 块 32x32 | 1024 | 1.34 | 失败 |
| 块 32x64 | 2048 | 1.50 | 失败 |
| 块 64x64 | 4096 | 1.66 | 失败 |
| 块 128x128 | 16384 | 2.00 | 失败 |

由此得到两个决定后续工作的结论：

1. **在相同元数据成本下，一维 K 轴分组优于二维块缩放。** `K 轴 256` 与 `16x16`
   都是每 256 个权重共享一个缩放，但 K 轴布局的 NMSE 约好 10%，并落在对照组的
   通过一侧。DiT 投影层各行的动态范围差异很大，跨 16 个输出通道的块必须为块内
   最宽的那一行买单。因此不建议为 INT4 SVDQuant 采用二维块契约。
2. **权重边界位于 K = 256 与 per-output-channel 之间。** K = 256 仍略优于 MXFP4
   对照组，而 per-channel 明显更差。

激活扫描（权重块固定为 K = 64，W4A4，对比 W4A4 的 MXFP4 group-32 对照组）表明，
主导误差的是动态 INT4 激活而非权重：所扫描的每个激活分组（16、32、64、128、
逐 token）都劣于同分组下的 MXFP4 W4A4 对照组，且差距随分组增大而单调扩大。若必须
支持 W4A4 产品路径，激活分组应保持在 MXFP4 对照组的 32 及以下，并在模型级重新
验证；W4A16 的 INT4 路径则宽裕得多。

## 步骤 3 与 5：对照组与模型级扫描

这两步需要具备 FLUX 运行能力的加速器（Intel XPU 是已验证的 SVDQuant 路径，CUDA
同样可用），以及 `diffusers`、`torchmetrics` 和 `image-reward`，并且**不会**在本
仓库的 CI 中运行。

```bash
# 步骤 3：在目标环境中生成全新的 BF16 + MXFP4 对照组
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/run_reference_sweep.py \
    --stage controls --model /models/FLUX.1-dev \
    --prompt-file coco2017_captions.tsv --output-dir records --include-rtn-controls

# 步骤 5：在筛查得到的边界附近做模型级扫描
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/run_reference_sweep.py \
    --stage sweep --weight-blocks 128 256 512 --terminals signround \
    --model /models/FLUX.1-dev --prompt-file coco2017_captions.tsv --output-dir records
```

固定设置（`contract.py::FIXED_CONTRACT`）：BF16 模型与低秩分支 dtype、秩 32、
128 条 COCO2017 caption、50 步推理、200 次 SignRound 迭代、20 次残差迭代、
1024x1024、guidance 3.5、batch size 1，并在 BF16、MXFP4 与所有 INT4 块大小之间
共享同一组提示词与逐提示词随机种子。每次运行都会保存指标、差值、墙钟时间、峰值
显存、解析后的配置以及软件版本。

主要可行性配置是 smooth SVDQuant + SignRound。如果产品路径是 RTN，还需要在最终
边界上运行 no-smooth + RTN；不得由 SignRound 的结果推断 RTN 的可行性。

## 步骤 6：验收门限

门限预先登记在 `contract.py` 中，看到扫描结果之后不得修改。

绝对下限（已公开的最差 MXFP4 SVDQuant RTN 结果，在专用 INT4 门限获批前作为保守的
临时包络）：

```
CLIP        >= 25.9624
CLIP-IQA    >= 0.946939
ImageReward >= 0.934579
```

相对于**全新** BF16 对照组的预算（一旦对照组可复现则优先使用）：CLIP `>= -0.10`、
CLIP-IQA `>= -0.010`、ImageReward `>= -0.09`。

只有当三项指标在聚合结果上全部通过，**并且**最差的重复运行也满足绝对下限时，候选
才算通过。绝不允许用某个更强的无关指标把失败指标平均掉。

```bash
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/aggregate_results.py \
    records/model_level_records.jsonl --output records/summary.json
```

该脚本会打印块大小对比表、检查更小块大小的单调性，并输出选定的最大通过块大小。

## 当前建议

基于筛查证据（步骤 4），有待步骤 5 的模型级确认：

* **推荐的 INT4 权重契约：一维 K 轴分组，`group_size = 128`。** 其误差为 MXFP4
  group-32 对照组的 0.80 倍，相对筛查边界 K = 256 留有充足余量，且 128 是 INT4
  GEMM 内核常用的、对 tile 友好的 K 粒度。若 128 的模型级结果处于临界状态，则以
  K = 64（0.68 倍）作为保守回退。
* **筛查得到的上界为 K = 256**；首个明确失败点是每个输出通道一个缩放。步骤 5 应
  评估 128 与 256（并以 64 作为更小的对照），如果仍在考虑二维契约，还应评估
  16x16。
* **不推荐二维 M×K 块缩放**：在同等元数据成本下它严格劣于 K 轴分组，且即便是
  16x16 也已经处于临界失败。
* **风险在 W4A4 而非 W4A16**：在所扫描的每一种配置中，激活量化都是误差的主导
  因素。

## 步骤 7：go/no-go 输入

对每个通过的块大小，决策需要：INT4 权重字节数与缩放元数据字节数、动态激活缩放的
带宽开销、残差 INT4 GEMM 成本、BF16 低秩 down/up GEMM 与中间张量成本、硬件 tile 与
内存布局兼容性、预期的带宽/占用率/时延收益，以及导出、加载器、后端与一致性测试的
实现成本。

在 K = 128 时，缩放元数据为每 128 个权重一个 FP16 值（在 4 bit 之上约
0.125 bit/权重），而 MXFP4 group 32 为每 32 个权重一个 E8M0 指数
（约 0.25 bit/权重）。因此精度通过的 INT4 工作点在元数据上比现有 MXFP4 路径**更
便宜**，并且其 K 粒度可直接映射到标准 INT4 GEMM tile。这对 W4A16 的 INT4 SVDQuant
内核是有利的 go 信号，但在 FLUX.1-dev 模型级扫描于真实硬件上确认该边界之前仍属
临时结论：如果只有很小的块能通过，从而抹掉了显存或吞吐收益，那就是精度上的成功、
产品上的 no-go。
