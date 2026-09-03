# ARK MXFP4 SVDQuant 双 Kernel 设计

## 1. 目标与范围

本文档以 SVDQuant 推理的两个核心 fused kernel 为主线：

1. **Kernel A：平滑、动态 MXFP4 激活量化与低秩 down projection。**
2. **Kernel B：MXFP4 W4A4 GEMM 与低秩 up projection。**

当前开发机器为 B60，无法执行目标 MXFP4 SIMD/matrix 指令。因此：

- 第一阶段完成并验收 Kernel A。
- 第二阶段尽可能完成 Kernel B 中不依赖目标指令的部分，并准备 native mainloop 的接口、测试和 benchmark；不要求在 B60 上打通完整 fused Kernel B。
- 不在 B60 上冻结目标机器相关的 tile、subgroup、swizzle 或 native packed layout。

离线 SVD 分解、平滑参数搜索、模型校准、checkpoint export 和整图 rewrite 不属于本文的主要开发范围；它们只需为两个 kernel 提供稳定输入。

## 2. 计算分解

对输入 `X`、平滑向量 `S`、量化残差权重 `Wq` 和低秩权重 `Ld/Lu`：

```text
Xh = X * S
QX, SX = MXFP4_RCEIL_QUANT(Xh)
L  = Xh @ Ld.T
Y  = MXFP4_GEMM(QX, SX, Wq, SW) + L @ Lu.T + bias
```

对应两个 kernel：

```text
Kernel A: X, S, Ld
       -> QX, SX, L

Kernel B: QX, SX, Wq, SW, L, Lu, bias
       -> Y
```

该边界与 SVDQuant/Nunchaku 的实现思路一致：量化激活和 low-rank down 共享同一次平滑后输入读取；W4A4 主计算和 low-rank up 在输出侧融合。

## 3. Kernel A：quantize + low-rank down

### 3.1 融合内容

Kernel A 在一次 launch 内完成：

1. 从全局内存读取 `X[M,K]`。
2. 逐元素计算 `Xh = X * smooth`；无 smooth 时等价于乘 1。
3. 每 32 个连续 K 元素计算动态 MXFP4 scale。
4. 按 AutoRound `rceil` contract 量化为 E2M1，并打包两个 4-bit code 到一个 byte。
5. 写出 UE8M0 activation scale。
6. 使用同一份 `Xh` 计算 `L = Xh @ lora_down.T`。

核心收益不是 MXFP4 矩阵指令，而是避免为量化分支和 low-rank 分支重复读取、平滑和存储输入。因此 Kernel A 可以在 B60 上完整开发。

### 3.2 逻辑接口

```cpp
void svdquant_mxfp4_quant_down(
    Tensor x,              // [M,K], BF16/FP16
    Tensor smooth,         // [K], BF16/FP16/FP32, optional
    Tensor lora_down,      // [R,K], BF16/FP16
    Tensor qact,           // [M,K/2], uint8
    Tensor ascales,        // [M,K/32], uint8 UE8M0
    Tensor lora_act,       // [M,R], 与 x 同 dtype
    Queue current_queue);
```

V1 约束：

- `K > 0` 且 `K % 32 == 0`。
- 首发支持 BF16 和 FP16 输入；两者分别验收。
- rank 优先支持 32，同时保留 16 的倍数扩展能力。
- `qact` 使用 low-nibble-first 的 logical layout；是否转为 Kernel B 的 native layout留到目标机器决定。
- `lora_act` **跟随 `x` 的 dtype**（BF16 或 FP16），累加仍然是 FP32，只有写回时收窄。

  这一条在 2026-09 修订过。原先的契约是 FP32，理由写的是"避免在两个 kernel 之间引入额外低精度误差"，
  但这个理由站不住：§5.1 第 4 步里 Kernel B 要算 `lora_act @ lora_up.T`，那是一个 16-bit DPAS GEMM，
  FP32 存进去之后 Kernel B 读出来第一件事就是 round 成 16-bit。多出的精度在跨 kernel 边界处直接被丢弃，
  只剩下带宽开销。输入侧 `lora_down` 本来就已经是 16-bit（与论文的 16-bit 低秩分支一致），
  输出侧却是 FP32，本身也不自洽。跟随 `x` 的 dtype 同时和"`lora_down` dtype 必须等于 `x` dtype"
  这条既有约定保持一致，并且正好是 Kernel B GEMM 想要的操作数类型。
- 使用 PyTorch 当前 XPU queue，不允许隐式 host sync。

### 3.2.1 与 ARK 现有工程约定对齐

Kernel A 的 Python/C++ 边界不需要新发明约定，应直接复用 `auto_round_extension/ark` 中已有的模式，
降低 review 成本并避免和其他 kernel 的调用风格不一致：

- **queue 获取**：复用 `auto_round_kernel/__init__.py` 中的 `get_stream(tensor)`（按输入 tensor 的
  device 取 `torch.xpu.current_stream().sycl_queue`），不要为 Kernel A 单独定义一套 queue 传参方式。
- **dtype 枚举**：复用 `ARK_DT`（已包含 `float8_e8m0`，可直接作为 `ascales` 的枚举值）。新增
  E2M1/MXFP4 codes 时在 `ARK_DT` 里追加常量，不要引入平行的第二套 dtype 编码。
- **Python 侧形状/dtype 校验**：复用 `qlinear.py`/`__init__.py` 中 `_validate_packed_blob` 一类的
  前置校验写法，在调用原生 kernel 前把 ABI 契约（第 4.2 节）在 Python 层再检查一遍，而不是只依赖
  C++ 侧断言。
- **扩展加载**：复用 `xpu_loader.py` 的 `ensure_xpu_lib`/`load_xpu_lib`，新 kernel 的 `.so` 通过同一套
  `required_symbols` 机制暴露给 Python，不新增加载器。

### 3.3 实现建议

以二维 work-group 覆盖 `[M,K]` tile：

- 每个 K-group 的线程协作计算 `amax`。
- exponent 按 3.3.1 冻结契约计算，clamp 到 `[-127, 127]`，因此 UE8M0 code 落在 `[0, 254]`，不生成保留码 255。
- scale 和 code 的 rounding 必须与 AutoRound `quant_mx_rceil` 逐位一致。
- 平滑后的 tile 保留在 register/SLM 中，同时用于 pack 和 down GEMM。
- down projection 使用 FP32 accumulator；不同 reduction 顺序允许正常浮点误差，但不能影响 qact/scale。
- M/K tail 必须显式 mask，padding 不参与 scale reduction。

不要为了模拟目标机器而提前固化：

- Kernel B 专用 weight/activation swizzle。
- 目标 MXFP4 matrix instruction 要求的 tile。
- 仅凭 B60 推测的 subgroup 或 SLM 配置。

### 3.3.1 冻结的数值契约（A0）

本节是 Kernel A 的**唯一**数值权威，已于实现开始前冻结，等价于
`quant_mx_rceil(..., bits=4, group_size=32, data_type="mx_fp4e2m1")` 加
`svdquant_mxfp4.py` 的 codec。任何实现（SYCL / PyTorch reference）都必须逐位复现下述步骤。

**Step 1 — smooth**：`xh = float32(x) * float32(smooth[k])`；`smooth` 为 None 时等价于乘 1。
所有后续计算在 FP32 中进行。

**Step 2 — 组内 amax**：`amax = max(|xh|)` over 每 32 个连续 K 元素。

**Step 3 — shared exponent**（注意零组与退化特例）：

```text
shared_exp = (amax == 0) ? 1.0 : ceil( fp32( log2(amax / 6.0) ) )
shared_exp = clamp(shared_exp, -127, 127)
scale      = 2^shared_exp
ue8m0_code = shared_exp + 127          // 落在 [0, 254]
```

> **零组特例是刻意保留的。** `amax == 0` 时 `quant_mx_rceil` 输出 `shared_exp = 1`（scale `2.0`，
> UE8M0 code `128`），而不是 0/code 127。这源于 mxfp.py 中 `torch.where(max_val == 0,
> torch.ones_like(max_val), ...)` 用 1 作为占位符，且现有 exporter
> `svdquant_mxfp4.py::pack_residual` 已经继承了该行为并落盘。ARK 选择**严格对齐 oracle 与 exporter**
> （方案 A），使 §4.1 的 "bit-exact" gate 不需要任何豁免口。零组的所有 E2M1 code 均为 0，因此反量化
> 结果与 exponent 取值无关，此选择不影响端到端数值，只影响 `ascales` 字节。实现成本两种方案相同。

> **退化下溢特例（对 oracle 的唯一刻意偏离）。** 当整组都是极小 subnormal，使 `amax / 6.0` 在 FP32 下
> 下溢为 0（阈值约 `amax <= 4.9e-45`）时，`log2` 得到 `-inf`，clamp 后应为 `-127`（code 0）。而
> oracle 在此处返回 **NaN**：`ceil_ste(x) = (x.ceil() - x).detach() + x` 在 `x = -inf` 时算出
> `-inf - (-inf) = NaN`，属于 STE 实现泄漏进前向值；整组 qdq 随之全变 NaN。现有 exporter 在该区间同样
> 没有可用契约（`encode_ue8m0(NaN)` 静默退化成 code 127，而 `encode_e2m1` 直接抛 ValueError）。
> ARK 取 clamp 本来的意图 `-127`：该组所有 code 本就是 0，反量化结果为 0，数值上完全合理，且绝不向
> 下游传播 NaN。此区间在真实激活中几乎不可达（需一组 32 个值全部 `< 5e-45`）。

> **`ceil(log2(·))` 必须按"先把实数对数舍入到 FP32，再取 ceil"求值。** 不能用
> "从 exponent bit 取 `floor(log2)`，非 2 的幂再 +1" 的整数捷径：对刚好略大于 `2^e` 的 `v`，真实对数
> `e + 1.7e-7` 在 `ulp(e)` 大于该偏移时（即 `|e| >= 4`，几乎总是成立）会被舍回**恰好 `e`**，于是
> `ceil` 得 `e` 而非 `e+1`。整数捷径在 254 个 `nextafter(2^e)` 采样点里有 246 个与 oracle 不符。
> 工程上可靠的做法是**在 double 下算 log2 再窄化为 FP32**：该写法在 1600 万个围绕 2 的幂构造的
> 对抗性样本上与 torch 逐位一致，且 torch 自身的 `log2` 在 CPU 与 XPU 上结果相同（同一批样本 0 分歧），
> 因此该契约是设备无关且可复现的。

**Step 4 — 归一化与 clamp**：`t = clamp(xh / scale, -6.0, 6.0)`。

**Step 5 — E2M1 量化（round-half-to-even，作用在 code index 上）**：
E2M1 magnitude codebook 为 `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`（code 0..7）。
**必须逐位复现 oracle 的 FP32 运算序列**，而不是用中点阈值比较：

```text
pe   = max(floor(log2(a)), 0)          // a = |t|；pe ∈ {0,1,2}，由 FP32 exponent bit 直接取出
s    = a * 2^(1 - pe)
q    = floor(s + 0.5) - ((s - 0.5) mod 2 == 0)      // round-half-to-even
code = q + 2 * pe
```

`code = q + 2*pe` 成立是因为 oracle 的 magnitude 为 `q * 2^(pe-1)`，枚举可达的 `(pe, q)` 组合后与
codebook index 一一对应。`pe` 用 exponent bit 提取而非 `log2`，既更快也更可靠（GPU `log2` 只有几 ULP
精度，在 2 的整数幂处可能误分类）；零和 subnormal 的 exponent 字段为 0，得 `-127`，被 `max(·,0)` 归为
0，与 oracle 的 `clip(min=min_exp)`（E2M1 下 `min_exp == 0`）结果一致。

其数学意图等价于下表（**仅供理解，不可作为实现**）：

| `a` 区间 | code | magnitude |
|---|---:|---:|
| `[0, 0.25]` | 0 | 0.0 |
| `(0.25, 0.75)` | 1 | 0.5 |
| `[0.75, 1.25]` | 2 | 1.0 |
| `(1.25, 1.75)` | 3 | 1.5 |
| `[1.75, 2.5]` | 4 | 2.0 |
| `(2.5, 3.5)` | 5 | 3.0 |
| `[3.5, 5.0]` | 6 | 4.0 |
| `(5.0, 6.0]` | 7 | 6.0 |

> **为什么中点阈值表不能直接用于实现。** `floor(s + 0.5)` 本身在 FP32 下会舍入，因此比中点小 1 ULP 的
> 值仍可能进位。例如 `a = 0.24999998509`（比 0.25 这个 tie 小 1 ULP）会使 `s + 0.5` 恰好落在 FP32 的
> 中间值上，round-half-to-even 将其取整为 `1.0`，于是 oracle 给出 code 1，而按上表比较则得 code 0。
> 这类点在真实数据中可达（随机数据约每 1e7 个元素出现一次，FLUX 单层 1400 万元素即会命中若干个），
> 所以 kernel 必须复现该行为，§4.1 的 bit-exact gate 才无需开豁免口。上述运算序列已在 60 万个结构化
> 采样点（含每个 tie 的 ±4 ULP 邻域）和 300 万个随机点上与 `quant_element` 零分歧。

**Step 6 — 符号位**：`code |= (signbit(t) ? 8 : 0)`。符号取自归一化后的值，`-0.0` 的符号位为 1
（与 `torch.signbit` 及 `encode_e2m1` 一致）。

**Step 7 — 打包**：low-nibble-first，`qact[m, j] = code[m, 2j] | (code[m, 2j+1] << 4)`
（与 `svdquant_mxfp4.py::pack_nibbles` 一致）。

**Step 8 — low-rank down**：`lora_act[m, r] = sum_k xh[m, k] * float32(lora_down[r, k])`，FP32 累加，
按 `x` 的 dtype 写回。注意乘的是**平滑后未量化**的 `xh`，不是反量化后的值。

收窄**只发生在写回这一步**：累加器全程是 FP32，DPAS 路径的 tail epilogue 也走同一条收窄语句，
因此整行 tile 和尾部行携带完全相同的舍入。

**非有限输入**：`x`/`smooth`/`lora_down` 含 NaN/Inf 时行为未定义；debug 构建必须显式报错，
release 路径不做检测（见 §4.1）。

### 3.3.2 实现落地记录（Kernel A，已实现并验证）

本节记录实际实现过程中发现的、无法从契约本身推导出来的工程约束。三条都属于
"不知道就会静默产生错误结果"的类型，任何重写 Kernel A 的人都必须先读这一节。

**文件布局**

| 文件 | 角色 |
|---|---|
| `auto_round_kernel/svdquant_mxfp4.py` | A0 的 PyTorch 参考实现 + 公共入口 `svdquant_mxfp4_quant_down(backend=auto/ark/reference)`，同时充当测试 oracle 与 emulated fallback |
| `auto_round_kernel/wrapper/include/sycl_svdquant_mxfp4.hpp` | 设备侧数值原语（`e2m1_magnitude_code` / `shared_exponent` / `divide_by_e2m1_max_norm` / `encode_group`）与 host 入口声明；Kernel B 与独立 SYCL 测试复用同一套编码路径 |
| `auto_round_kernel/svdquant_mxfp4_kernel.cpp` | SYCL 并行分解与 launcher |
| `ark.cpp` | `svdquant_mxfp4_quant_down` pybind 绑定（薄封装，只做指针转换） |
| `test/test_svdquant_mxfp4.py` | §4.1 / §4.2 的正确性与 ABI 测试套件（见 §4.1） |
| `test/bench_svdquant_mxfp4.py` | §4.3 对比非融合三步基线的 benchmark（见 §4.3） |

**发现 1：Intel GPU 的 FP32 除法默认不是 IEEE 正确舍入的。**
offload backend 把 `fdiv` 展开成"倒数近似 + 修正"，结果可能比正确商大 1 ULP。
在 `amax / 6` 落在 `6 * 2^e` 附近时，这 1 ULP 会把 `ceil(log2(·))` 顶高一级，
产生 oracle 永远不会输出的 UE8M0 code（B60 实测：`amax = 0x3cc00002` 时 code 120 vs 期望 119）。

- `-fp-model=precise` **不能**修复它：除法展开发生在 link 期的 offload backend，而不是前端。
- `-foffload-fp32-prec-div` 可以修复，但它只在最终 link 生效，会作用于模块内**所有** kernel。
- 采用的方案：在 kernel 内用 Markstein 修正把除法做成正确舍入，不依赖任何全局 flag：
  `q0 = a * y; r = fma(-b, q0, a); q = fma(r, y, q0)`，其中 `y = 1/6` 在 host 端常量折叠因而正确舍入。
  已在 B60 上对 4.2M 个正规数逐位对齐 IEEE FP32 除法。
- 同理，step 4 的 `xh / scale` 改为乘以 `ldexp(1.0f, -exp)`：对 2 的幂而言两者数学等价，
  但后者完全绕开近似除法。

**发现 2：该 TU 必须用 `-fp-model=precise` 编译。** icpx 默认 `-fp-model=fast`，
其重结合会把上面的 Markstein 修正直接折叠回一次普通（近似）除法，等于白做。
已在 `CMakeLists.txt` 中用 `set_source_files_properties` 单文件设置，不影响其他 kernel 的性能。

**发现 3：设备端 `log2` 本身是可信的。** B60 上 `sycl::log2(float)` 与 torch/numpy 的
FP32 `log2` 在所有测试输入上逐位一致（含 `2^e` 与 `6*2^e` 邻域 ±6 ULP、全部 bf16/fp16 正值的穷举、
以及跨全指数范围的 120 万随机值）。因此默认走 FP32 路径；
`ARK_SVDQ_LOG2_FP64=1` 保留 FP64 逃生舱，供 FP32 `log2` 精度更差的部件使用。
早期"必须用 double 再窄化"的判断是把除法误差归因到了 log2 上，已更正。

**并行分解**：分成两条专用路径，按是否请求 low-rank down 投影来选择。

*纯量化路径*（`launch_quant_only`，无 `lora_down`）：扁平 1-D range，每个 work-item 负责一个
32 元素 group，`kQuantWorkGroupSize = 256`。没有 low-rank 投影时不存在跨行复用可利用，
任何按行分块的形状都只会白白丢掉并行度。该路径已是访存受限（见下方数据）。

*融合路径*（`launch_fused`）：**零 shared local memory**，完全构建在 sub-group collective 之上。
一个 sub-group（`kWorkGroupSize = 16` lane）负责 `kRowsPerSubGroup` 行、跨完整 K；
`kFusedWorkGroupSize = 256` 即一个 work-group 内放 16 个 sub-group。
让 sub-group 走完整个 K 是 fuse low-rank down 的前提——`lora_down` 因此每个 sub-group 重读一遍，
而不是每行一遍。sub-group 内各 lane **协同**处理同一个 32 元素 group，lane `l` 负责元素对
`(2l, 2l+1)`；这让 `x` 与 `lora_down` 的访存都完全合并，并使 `amax` 可以由单次
`reduce_over_group(sycl::maximum)` 得到。rank 累加器 `acc[MaxRank]` 常驻寄存器，
最后统一用 `reduce_over_group(sycl::plus)` 归约一次，且每个 rank 的写回分散到不同 lane，
而非全部经由 lane 0。主循环按 `kGroupsPerStep` 个 group 展开；这一步是必需的，
否则逐 group 的 `amax` `reduce_over_group` 会形成串行延迟链并主导整个 kernel。

由于归约顺序固定，`lora_act` 是 run-to-run 确定性的，但**不要求**与 PyTorch 逐位一致
（归约顺序不同，§4.1 用容差把关）。

**调优常量**（`svdquant_mxfp4_kernel.cpp`，均在 B60 上实测——换硬件请重新扫）：
`kGroupsPerStep = 16`、`kRowsPerSubGroup = 1`。这两个参数强非单调且互相耦合，
请成对修改并重新测量。`4608x3072 R=32` 上实测：`(4,1)` 0.83 ms、`(4,2)` 0.91 ms、
`(8,2)` 0.61 ms、`(16,1)` **0.53 ms**、`(16,2)` 1.09 ms、`(32,1)` 0.77 ms。
`kRowsPerSubGroup = 2` 只在极大 K 上有明显收益（它把 `lora_down` 重读减半：
`4608x12288` 用 `(8,2)` 从 2.98 ms 降到 2.62 ms），但在其他形状上要付出 15%–100% 的代价，
因为 `acc[kRowsPerSubGroup][MaxRank]` 在 `R = 32` 时会撞上寄存器压力。
按 K 分派是一个合理的后续选项，只是目前还不值得多一份模板实例化。

**已验证**：`qact` / `ascales` 在 §4.1 全部用例上与 oracle 逐位一致（bf16/fp16、
M ∈ {1,2,7,8,16,64,255,256,4608}、K ∈ {32,64,96,128,256,512,1024,3072,12288}、
R ∈ {1,8,16,32,64}、零组 / 次正规 / 全 6.0 / 负零 / 2 的幂 / 稀疏等）。

`lora_act` 在 `4608x3072 R=32` 上的相对 L2（对 FP64 oracle）：bf16 输入 1.658e-3（DPAS 与 scalar 相同，
均正好压在 1.661e-3 的 bf16 存储下限上）；fp16 输入 scalar 2.078e-4（同样贴住 2.071e-4 的下限）、
DPAS 2.927e-4。

DPAS 在 fp16 下比下限高 1.4x 是**可解释的**：split 的残差 `lo = value - float(hi)` 必须存回操作数类型 T，
而 fp16 的指数范围很窄——`lo` 的量级约为 `value * 2^-11`，当 `value` 的指数低于 -3 时 `lo` 就掉进
fp16 的次正规区并开始丢位。bf16 没有这个问题，因为它保留了 fp32 的指数范围。这个差距远在 §4.1
的 5e-4 门槛之内，因此接受；如果将来需要消掉，办法是把 split 的两个平面存成 bf16 而非 T
（代价是 A/B 必须同型，需要额外一次 A 的转换）。

**实测性能（B60，bf16，P50，对比 PyTorch 参考路径）**：

| shape | R | ARK | 参考路径 | 加速比 |
|---|---|---|---|---|
| `4608x3072` | 32 | 0.232 ms | 7.366 ms | 32.3x |
| `4608x3072` | 16 | 0.226 ms | 7.395 ms | 32.8x |
| `4608x3072` | 64 | 0.254 ms | 7.584 ms | 29.7x |
| `1024x3072` | 32 | 0.088 ms | 1.565 ms | 17.4x |
| `4608x12288` | 32 | 1.130 ms | 29.333 ms | 25.9x |
| `256x3072` | 32 | 0.068 ms | 0.421 ms | 6.0x |
| `1x3072` | 32 | 0.061 ms | 0.415 ms | 6.8x |
| `4608x3072` 纯量化 | - | 0.134 ms | - | 267 GB/s |
| `4608x12288` 纯量化 | - | 0.430 ms | - | 334 GB/s |

（geomean 23.3x，最差 6.0x，均已过 §4.3 门槛。`4608x3072 R=32` 上的演进：
scalar 0.530 ms / geomean 12.5x，joint_matrix DPAS 0.248 ms / geomean 18.3x，
sycl-tla 0.232 ms / geomean 23.3x。最大的单项跃升并不在这个 shape 上，而在 R=64 与 M=1——见 §3.3.4。）

纯量化路径基本已贴到访存屋顶：`4608x3072` 下仅在 device 上读一遍 `x` 就要 0.092 ms（314 GB/s）。

**剩余空间**：低秩分支的诚实上限是**访存屋顶而不是算力屋顶**。`4608x3072 R=32` 的算术强度是
`2*M*K*R / (M*K*2)` ≈ 32 FLOP/byte，而 B60 的 machine balance 是 88e12/386e9 ≈ 228 FLOP/byte，
因此严重 memory bound：28.5 MB 的操作数在实测 386 GB/s 下就是 ~0.074 ms。
不要再用 TFLOP/s 给这条分支定目标。

尚未尝试的手段，按预期收益大致排序：**把低秩投影融进量化 kernel，让 `x` 只读一遍而不是两遍**——
这已是当前最大的单项缺口，`4608x3072` 下融合 kernel 是 0.232 ms、纯量化是 0.134 ms，
差值几乎正好是再过一遍 `x`；针对小 M 的 K 切分（§3.3.4 的行块调整已经找回大部分，优先级下降）；
以及通过编译器诊断检查实际的寄存器 spill，这一项仍然从未直接做过。
原先列在这里的 sycl-tla 2D block load 已经**完成**——见 §3.3.4，
而且它生效的原因与当初的预测并不相同。
性能优化是独立于正确性的后续阶段，不改变 §3.3.1 契约。

**关于"融合是否必要"**：在本契约下非融合写法**不具竞争力**，因为 §3.3.1 要求 smoothing 在 FP32 上完成。
仅仅为了让 torch 能消费而把这个 FP32 中间量物化出来——`(x.float() * smooth).bfloat16() @ lora_down.T`
——在 `4608x3072` 上就已经要 0.986 ms，比整个融合 kernel 还慢。

### 3.3.3 DPAS 低秩投影（已实现并验证）

低秩分支在 scalar 形式下只跑到 ~2.5 TFLOP/s，根因是**取操作数**而不是算力：内层循环每次 FMA 要读
2 字节 `lora_down`（bytes/FMA = 2.00，零寄存器复用），于是 `lora_down` 被逐行重读——`R=32` 时 906 MB，
而 `x` 本身只有 28 MB。DPAS 从结构上解决这个问题：脉动阵列免费提供操作数复用，且累加器搬到专用
累加寄存器，顺带解开了原先卡住行分块的寄存器压力。

**关键的代数改写（fold `smooth` into B）。** 直觉映射（A = 平滑后的激活）是错的：它会强迫把 FP32 的
`x * smooth` 物化、舍入到 16-bit、再经 SLM 中转才能给 `joint_matrix_load` 使用，而 SLM 占用正是早期
版本变慢的原因。注意 `smooth` 是 per-column（per-K）的，因此可以折进**另一个**操作数：

```
lora_act = (x * smooth) @ lora_down^T  ==  x @ (smooth * lora_down)^T
```

好处有三：A 变成内存中原封不动的 `x`，无中转、无 SLM、**A 上零舍入**；被折叠的操作数 `smooth * lora_down`
是 `[K, R<=64]`，极小且常驻 cache；所有精度损失集中到这一个小矩阵上，便于廉价修正。

**精度：split（双趟）。** DPAS 没有 FP32×FP32 模式，被折叠的操作数必须落到 16-bit。只舍入一次的代价是
相对 L2 1.65e-3；把它拆成高位与残差两个平面、跑两趟 DPAS 累进同一个 FP32 累加器，可恢复到 2.44e-6。
由于本 kernel 是 memory bound 且该矩阵已常驻 cache，第二趟几乎免费。§4.1 已记录：在 16-bit 输出契约下，
split 双趟正好压在存储下限上，而单趟比下限差 1.4x~8x——**16-bit 输出并没有让 split 变得多余**。

**VNNI layout。** B 由 prologue 预先 tile 并 VNNI 交织；packer 与 consumer 共用
`packed_b_offset()`，因此两者不可能对不上。B 的两个平面放在调用方提供的 workspace 里，
大小由 `svdquant_workspace_elements()` 查询——Python 侧不复制这个布局知识。

**M 门槛是结构性的，不是调参。** `dpas_lora_supported` 要求 `m >= kTileM (= 8)`。低于一个 tile 高度时
根本没有 tiled 行，整个投影落到 `launch_lora_dpas_tail` 这个 scalar epilogue 上，而它只有 `m * r` 个
work item（M=1、R=32 时是 32 个，跑在 160 EU 上）。K=3072 实测：M=4 只有 scalar 融合路径的 0.24x，
M=8 已经是 2.05x，K=12288 上的断崖位置完全相同。

**`ARK_SVDQUANT_DISABLE_DPAS=1`** 强制走 scalar 路径，测试用它把两条路径互相对拍。
这个 env **刻意不缓存在 function-local static 里**：测试要在同一进程内的两次 launch 之间翻转它，
缓存会让该对拍静默失效（这确实发生过一次）。

**`ARK_SVDQUANT_DISABLE_CUTE=1`** 从 §3.3.4 的 sycl-tla 路径回退到本路径，
不缓存的规则与理由同上。

**状态**：本路径作为默认实现已被 §3.3.4 取代——后者在所有实测 shape 上都更快。
保留它的原因有二：`ARK_SYCL_TLA` 关闭时它是回退路径；以及两条路径在
`test_cute_matches_joint_matrix_path` 中互相对拍。

### 3.3.4 sycl-tla（CuTe）低秩投影（已实现并验证；当前默认）

§3.3.3 把 A 的加载卡在 161 GB/s（屋顶 386），原因是 `joint_matrix_load` 固定取 8 × 16 的 tile，
即每行 32 字节、只有半个 cache line。当时的假设是：Xe 的 2D block load atom 暴露了
`XE_LOAD_2D<Bits, Height, Width, BlockWidth>`（`Height <= 32`、`Bits * Width <= 512`），
用 32 行 × 64 字节的取数就能补上这个缺口。

实现选在 **CuTe 的 arch 层**（直接用 `XE_LOAD_2D` / `XE_DPAS_TT` 加
`__builtin_IB_subgroup_createBlock2DAddressPayload` intrinsic），而不是 device 层。
拒绝 `GemmUniversalAdapter` 的理由与 §3.3.3 中已说明的一致：它的 `<_256,_256,_32>` tile 在
`N = R <= 64` 时 MMA 利用率只有 12.5%，而且它是独立一次 launch，必须重读 `x`。

**三条硬件契约，全部靠 probe 实测而非阅读文档确定。** 这三条只要猜错，产出的数字看起来都很合理
但其实是错的，因此在动手写 kernel **之前**，先在 B60 上对着 host double 参考逐条测过：

1. **`Count=2` 加载的寄存器排布。** `XE_LOAD_2D<16, 32, 32, 16>` 以两个 16 列 block 的形式取
   32 行 × 32 列，目标寄存器按 **block-major** 排列：A tile `(kt, mt)` 落在
   `a[kt * MTiles + mt]`。猜成 row-major 同样"合理"，但会静默地把 K 方向的贡献转置。
2. **`XE_LOAD_2D_VNNI` 由硬件完成 VNNI 交织**，读入的是普通 row-major 的 `[K, N]` 面。
   因此 §3.3.3 里 host 侧的 packer 在本路径上是被**删除**而不是重写：B 就是一个普通的
   row-major `[K, R]` 矩阵。
3. **越界行读回 0。** 该加载会对 payload 中记录的 surface 高度做边界检查。
   这正是本路径**不需要 scalar 尾巴 kernel、也没有 M 下限**的原因：不完整的行 tile 由同一份代码
   处理，只需要对 store 加保护。

**假设被实测证伪。** 把每 sub-group 的行数在 8 / 16 / 32 之间扫描：

| 每 sub-group 行数 | 8 | 16 | 32 |
| --- | --- | --- | --- |
| 几何平均加速比 | **23.33x** | 22.07x | 20.35x |
| flux mlp 4608x3072 R32 | **0.231 ms** | 0.238 | 0.261 |
| flux ffn 4608x12288 R32 | **1.130 ms** | 1.236 | 1.266 |
| mid batch 1024x3072 R32 | **0.087 ms** | 0.100 | 0.126 |

行块越宽反而单调**更慢**。有两个效应盖过了取数宽度：

* **占用率**：每 sub-group 的行数会去除 sub-group 总数，行块过宽会把机器饿死——M = 1024、
  32 行时整个 grid 只有 2 个 work-group，而设备有 160 个 EU。
* **寄存器压力**：累加器是每 lane `Rows/8 * NTiles` 个 8 float 的向量，在 `Rows = 32, R = 64`
  时直接 spill：0.763 ms，而行数减半后是 0.266 ms。

所以行块就取 8，并且**本路径取胜的原因并不是当初立项的那个理由**。`Rows = 8` 时它的行块高度与
`joint_matrix` 完全相同；真正起作用的是加载在 *K* 方向的宽度（每条指令两个 K-tile，行长 64 字节
而非 32）、硬件 VNNI 变换，以及不需要独立的尾巴 kernel。

`ARK_SVDQUANT_CUTE_ROWS`（8/16/32）保留了更宽的实例化，便于在寄存器堆或 EU 数不同的硬件上
免重编译地重跑这次扫描。会重新引入 spill 的取值会被 clamp 而不是照单执行，
`test_cute_row_blocking_override_is_equivalent` 则把三种取值的结果钉在一起。

**与 §3.3.3 的实测对比**（P50，bf16，越低越好）：

| shape | sycl-tla | joint_matrix | 收益 |
| --- | --- | --- | --- |
| flux mlp 4608x3072 R32 | 0.232 ms | 0.249 | 1.07x |
| flux mlp R16 | 0.226 ms | 0.230 | 1.02x |
| flux mlp R64 | 0.254 ms | 0.390 | **1.54x** |
| flux ffn 4608x12288 R32 | 1.130 ms | 1.240 | 1.10x |
| mid batch 1024x3072 | 0.088 ms | 0.111 | 1.26x |
| small batch 256x3072 | 0.068 ms | 0.093 | 1.37x |
| single token M=1 | 0.061 ms | 0.156 | **2.56x** |
| 相对非融合基线的几何平均 | **23.31x** | 18.34x | |
| 单 shape 最差 | **5.99x** | 2.55x | |

收益最大的两处正好对应两个结构性差异：R=64 是 joint_matrix 累加器耗尽寄存器的地方，
M=1 是它交给尾巴 kernel 的地方。

**构建注意事项**：kernel 主体必须放在 `#ifdef __SYCL_DEVICE_ONLY__` 里，因为 2D block load 的
builtin 没有 host 声明。这会导致隐式 `[=]` 捕获在 host 编译时捕获**零个**变量、device 编译时捕获
八个，从而触发 SYCL 的 lambda 尺寸 static assert。因此捕获列表必须显式写出——今后任何使用这些
intrinsic 的 kernel 都同理。

### 3.3.5 将投影融合进量化内核（已实现；**当前为性能回退**）

在此之前，矩阵路径对 `x` 发起两次 launch：`launch_fused_cute` 做投影，`launch_quant_only` 做编码。
因此第 4.4 节验收标准 2（"对 `x` 只有一次 launch"）一直未满足，而两次 launch 的总耗时
（flux mlp 上 0.232 ms）与单独 quant-only（0.134 ms）之间的差距，几乎正好是对 `x` 多走一遍 DRAM
的代价。消除它是显而易见的下一步。

**融合在结构上为什么是干净的。** `kGroupSize == kCuteKStep == 32`。一个 DPAS 的 K 步恰好等于每行
一个 micro-scaling group，因此循环的每次迭代都可以把刚载入的那些行的量化做完——没有残留状态，
不需要第二次遍历。`kFusedCuteRows` 固定为 `kDpasSubGroup == 16`，使量化阶段 lane -> 行 1:1 映射，
从而可以**逐字**复用 `load_smoothed`、`encode_group` 和 `store_group`。这种逐字复用是刻意的：
这三个函数正是 bit-exact PyTorch oracle 测试所钉住的对象，A0 契约不允许移动。结果是成功的——
`test_fused_cute_matches_scalar_path` 在 M ∈ {1, 7, 8, 16, 17, 255, 1024} 与两种 dtype 上对
`qact` 和 `ascales` 断言 `torch.equal`，全部 138 个测试通过。

两个阶段需要的 `x` 每-lane 布局不同，因此量化阶段重新读一次 `x`，而不是去解码 DPAS 的 A 寄存器。
这次重读命中 L1（`XE_LOAD_2D` 刚刚取过同一块 16x64B tile）；要省掉的是 DRAM 往返。之所以放弃从 A
寄存器解码，是因为那需要跨 lane shuffle 来为 nibble 打包重新配对元素，会拿被冻结的 A0 契约去冒险，
却换不到任何带宽收益。

**但它仍然更慢。** 几何平均从 23.31x 掉到 11.09x；flux mlp 从 0.232 -> 0.402 ms，M=1 从
0.061 -> 0.309 ms。墙钟时间几乎不随 M 变化——M=16 是 0.29 ms，M=1024 是 0.30 ms，工作量增加 64 倍
而耗时只多 4%——这是"根本没有被吞吐限制"的典型特征。

**实测原因是并行度饥饿，而不是寄存器溢出。** 溢出是第一个假设（在 3.3.4 的 32 行分块上它是对的），
但在这里是错的。把总元素数固定在 4608x3072，在行与列之间交换——这会把工作在**并行**的 M 维和内核内
**串行**的 K 循环之间搬移，而既不改变 FLOPs 也不改变字节数——得到：

| M | K | subgroup 数 (M/16) | K 迭代数 | fused P50 |
| --- | --- | --- | --- | --- |
| 4608 | 3072 | 288 | 96 | 0.611 ms |
| 9216 | 1536 | 576 | 48 | 0.212 ms |
| 18432 | 768 | 1152 | 24 | **0.149 ms** |
| 36864 | 384 | 2304 | 12 | 0.162 ms |
| 73728 | 192 | 4608 | 6 | 0.166 ms |

工作量完全相同，差距 4.1 倍，并且随 subgroup 数单调改善，直到约 1000 个 subgroup 附近饱和。
融合内核只能暴露 `M / 16` 个 subgroup，因为 K 维被一个串行循环吃掉了；而 `launch_quant_only`
暴露的是 `M * K / 32` 个独立 work-item。在 4608x3072 上这是 288 个 subgroup（4608 个 work-item），
而这台设备大约可以同时容纳 2 万个 work-item——占用率约 22%，且没有别的 warp 来掩盖
load -> DPAS -> `load_smoothed` -> `encode_group` 这条依赖链的延迟。投影本身能忍受这一点，因为它
只占运行时间的一小部分；占主导的编码阶段则不能。

所以这次融合省掉了一次 DRAM 遍历，代价是损失了约 4 倍的延迟掩盖能力。在这台机器上，
那第二次遍历是两者中更便宜的一个。

**修复方案：切分 K（已实现）。** 每个行块分配 `S` 个 subgroup，各自负责 K 区间的一段连续切片。
量化完全不需要协作——micro-scaling group 在构造上就是独立的，且每个切片写入 `qact` 和 `ascales`
中互不相交的字节——因此切分它的代价恰好为零。只有投影需要沿 K 累加，所以只有它需要归约：
每个切片把一份 FP32 `[M, R]` 部分和平面写入 workspace，再由一个很小的
`launch_reduce_partials_cute` 按切片顺序求和。这里采用确定性的部分和缓冲而非 FP32 atomics，
以保证 `lora_act` 逐次运行可复现。

`S` 按固定的目标 subgroup 数来选取（`cute_k_slices`，`kCuteTargetSubGroups = 1024`），而不是取常数，
因为合适的取值完全取决于 M 本身已经提供了多少并行度。该数值背后的扫描，以及曲线在其之上为何重新
上翘的原因，记录在 `sycl_svdquant_mxfp4_cute.hpp` 中该常量的注释里；简而言之，部分和缓冲自身的
流量最终会超过它所换来的占用率收益。`ARK_SVDQUANT_CUTE_TARGET_SUBGROUPS` 可在新硬件上无需重编译
即重新扫描。

**结果。** 回退不仅被弥补，而且被超越——并且第 4.4 节的验收标准 2 现已满足，因为投影不再需要
自己单独遍历一遍 `x`。

| shape | 两次 launch | 融合，未切分 | 融合 + K 切分 |
| --- | --- | --- | --- |
| flux mlp 4608x3072 R32 | 0.232 ms | 0.402 | **0.195** |
| flux mlp R16 | 0.226 ms | 0.369 | **0.179** |
| flux mlp R64 | **0.254 ms** | 0.507 | 0.281 |
| flux ffn 4608x12288 R32 | 1.130 ms | 1.818 | **0.877** |
| mid batch 1024x3072 | 0.088 ms | 0.329 | **0.070** |
| small batch 256x3072 | 0.068 ms | 0.325 | **0.062** |
| single token M=1 | **0.061 ms** | 0.309 | 0.062 |
| 相对未融合基线的几何平均 | 23.31x | 11.09x | **25.79x** |
| R=32 相对 quant-only 的额外开销 | -- | 197.7% | **45.1%** |

R=64 是唯一仍落后于两次 launch 配置的 shape。它是 NTiles=4 的实例化，仅累加器就是每 lane 64 个
float，再加上 A 向量和 32 个 float 的 `smoothed` 暂存，因此它最先出现寄存器紧张——正是 3.3.4 中
决定行分块的同一种压力，这次通过 rank 维度体现出来。把 *N* 维也切分到多个 subgroup 可以把累加器
减半，代价是要么重复做量化，要么把量化限制在某一半 N 上；尚未尝试。

## 4. Kernel A 验收标准

### 4.1 数值正确性

以 §3.3.1 的 A0 契约为准，等价于 AutoRound PyTorch
`quant_mx_rceil(..., bits=4, group_size=32, data_type="mx_fp4e2m1")`，**包括其零组 exponent=1 的行为**。

| 项目 | 验收标准 |
|---|---|
| activation scale | 所有 finite 输入逐元素 bit-exact（含零组 code 128） |
| packed E2M1 code | unpack 后逐元素 bit-exact；packed byte 也必须一致 |
| zero group | exponent 为 1、code 全 0，不产生 NaN/Inf 或 code 255 |
| tie/边界值 | 覆盖上表全部 7 个中点、`amax=6`、略大于 6、最小/最大 exponent |
| non-finite | debug 路径明确报错；不得静默生成成功结果 |
| `lora_act` FP16 输入 | 相对 L2 `<= 5e-4`，cosine `>= 0.9999` |
| `lora_act` BF16 输入 | 相对 L2 `<= 3e-3`，cosine `>= 0.999` |

低秩阈值在 2026-09 随 16-bit 契约一起收紧过（原为 5e-3 / 1e-2）。新阈值不是拍脑袋定的，
而是压在**存储舍入下限**之上：在 `4608x3072 R=32` 上对 FP64 oracle 实测，
把一个数学上精确的结果单纯写成 BF16 就已经产生 1.661e-3 的相对 L2，写成 FP16 是 2.071e-4。
kernel 里任何实现都不可能低于这两个数，因为那是"把结果写出来"本身的代价。
既然下限如此之高，继续沿用 1e-2 就意味着一个真正坏掉的投影也能通过测试。

实测对照（相对 L2，对 FP64）：

| 方案 | FP32 输出 | 存 BF16 | 存 FP16 |
|---|---|---|---|
| scalar FP32 | 5.0e-7 | 1.661e-3 | 2.07e-4 |
| 单趟 bf16 B | 1.65e-3 | 2.34e-3 | 1.66e-3 |
| split-bf16 双趟 | 2.44e-6 | **1.661e-3** | **2.07e-4** |
| 纯存储舍入下限 | — | 1.661e-3 | 2.07e-4 |

两个非显然的结论：(1) split-bf16 双趟正好压在存储下限上，kernel 自身的贡献归零，
而单趟 bf16 比下限差 1.4×（bf16 输出）到 8×（fp16 输出）——**改成 16-bit 输出并没有让 split 双趟变得多余，
反而使它成为唯一能榨干 16-bit 格式的方案**；(2) FP16 存储比 BF16 准 8 倍，
但 fp16 的溢出 headroom 只有 100~200×（K=12288 时 `|lora_act|max` 已到 618），
所以跟随 `x` 的 dtype 而不是一律用 fp16。

这些阈值仍是 ARK 首发 gate，不是论文声明值；若真实 FLUX 层的 reference 分布证明门槛过松，应收紧，不应为通过测试而放宽。

**测试脚本**：`test/test_svdquant_mxfp4.py`。在 `auto_round_extension/ark/test` 下运行：

```bash
PYTHONPATH=<repo>/auto_round_extension/ark:<repo> python -m pytest test_svdquant_mxfp4.py -q
```

该套件对**两组相邻层**都做校验，以便定位任何偏差：
`quant_mx_rceil` oracle <-> `quant_down_reference` <-> 融合 SYCL kernel。
它还把与 oracle 之间那唯一一处有意的偏差（次正规下溢 NaN，见 §3.3.1）固化为显式测试，
这样将来 oracle 若被修复会被立刻发现，而不是被悄悄吸收。整体耗时约 5 秒。

除上述覆盖范围外还包括：当初捕获 §3.3.2 那个 bug 的 FP32 除法 / `log2` 边界压力测试
（在整个指数范围内对 `6*2^e` 与 `2^e` 各取 ±6 ULP，外加对**所有**有限正 bf16 / fp16 幅值的穷举扫描）；
§4.2 运行时 gate（输出 ABI、无 host sync、并发调用无共享 workspace 竞争、尾部行块）；
输入校验与 backend 分派行为（`backend="ark"` 绝不允许静默回退）；
以及与 `auto_round.export.svdquant_mxfp4` 的打包布局一致性。

必须覆盖：

- 随机、全零、常量、极小值、极大值、正负不对称输入。
- M：`1, 2, 7, 16, 64, 256, 4608`。
- K：最小合法值、非 tile 倍数但为 32 倍数，以及 FLUX 的 3072/12288。
- R：16/32，至少对 32 做完整性能验收。
- contiguous 与允许的非 contiguous 输入。
- BF16/FP16、不同 seed、重复运行确定性。

### 4.2 结构与运行时正确性

- qact、ascales、lora_act 的 shape、dtype、device 全部符合 ABI。
- 不读取 padding，不越界写出。
- 同一 queue 上 Kernel A 输出可直接被后续 consumer 使用。
- 不调用 `queue.wait()`、`torch.xpu.synchronize()` 或 CPU round-trip。
- 多 stream、多次并发调用无共享 workspace 数据竞争。
- 64-bit size arithmetic 通过 overflow 和超大 shape 拒绝测试。

### 4.3 性能验收

基线定义为同一 XPU 上三个独立步骤：

```text
smooth kernel
MXFP4 reference quant/pack kernel
BF16/FP16 lora_down GEMM
```

计时使用 XPU event，预热后至少 100 次迭代，报告 P50/P95；输入、输出和 workspace 预分配，禁止把分配时间混入 kernel 时间。

建议 gate：

| 项目 | 验收标准 |
|---|---|
| launch 数 | 主路径 1 个 Kernel A launch |
| global input traffic | profiler 证明 `X` 主 tile 只从 global memory 读取一次 |
| 同步 | 无 host sync |
| 代表 shape 性能 | 在 FLUX 代表 shape 的几何平均上比三步基线快至少 1.20x |
| 单 shape 回退 | 任一发布 shape 不低于基线的 0.95x；否则增加 dispatch/fallback |
| low-rank 增量 | 相比 quant-only Kernel A，rank-32 增量建议不超过 20% |

性能数字属于项目验收目标，不是论文保证。若 B60 的软件 E2M1 pack 限制了收益，仍要求 profiler 证明融合消除了重复流量；最终目标机器需重新测量并冻结正式阈值。

**测量脚本**：`test/bench_svdquant_mxfp4.py`。它把基线的三个步骤**分别**计时再把中位数相加，
这一做法刻意对基线有利（完全不计三个 kernel 之间的 launch 间隙），因此报告的加速比是真实收益的下界。
在 `auto_round_extension/ark` 下运行：

```bash
PYTHONPATH=. python test/bench_svdquant_mxfp4.py            # bf16，100 次迭代
PYTHONPATH=. python test/bench_svdquant_mxfp4.py --dtype fp16 --iters 200
```

**实测（B60，bf16，P50，预热 20 次后计时 100 次）**：

| shape | M | K | R | fused | smooth | quant+pack | GEMM | 合计 | 加速比 | GB/s | TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flux mlp | 4608 | 3072 | 32 | 0.535 ms | 0.595 | 6.575 | 0.305 | 7.474 | 13.96x | 68.4 | 1.69 |
| flux mlp r16 | 4608 | 3072 | 16 | 0.370 ms | 0.595 | 6.579 | 0.212 | 7.385 | 19.94x | 97.9 | 1.22 |
| flux mlp r64 | 4608 | 3072 | 64 | 0.968 ms | 0.594 | 6.578 | 0.360 | 7.533 | 7.78x | 38.6 | 1.87 |
| flux ffn | 4608 | 12288 | 32 | 3.066 ms | 2.325 | 26.064 | 0.912 | 29.302 | 9.56x | 47.2 | 1.18 |
| mid batch | 1024 | 3072 | 32 | 0.146 ms | 0.119 | 1.371 | 0.049 | 1.540 | 10.54x | 56.8 | 1.38 |
| small batch | 256 | 3072 | 32 | 0.138 ms | 0.043 | 0.320 | 0.047 | 0.410 | 2.97x | 16.2 | 0.36 |
| single token | 1 | 3072 | 32 | 0.126 ms | 0.043 | 0.315 | 0.045 | 0.403 | 3.20x | 1.7 | - |
| quant only | 4608 | 3072 | - | 0.134 ms | 0.595 | 6.593 | - | 7.188 | 53.52x | 266.9 | - |
| quant only ffn | 4608 | 12288 | - | 0.430 ms | 2.333 | 26.117 | - | 28.450 | 66.19x | 333.6 | - |

几何平均加速比 **12.48x**（gate 1.20x），最差单 shape **2.97x**（gate 0.95x），两个 gate 均大幅通过。

#### 实测设备屋顶（B60）

下面所有性能判断都以这组**实测**（而非估计）数据为基准：

| 屋顶 | 实测 |
|---|---|
| bf16 矩阵引擎（XMX），4096^3 `torch.matmul` | **88 TFLOP/s** |
| fp16 矩阵引擎 | 93 TFLOP/s |
| FP32 向量，4096^3 `torch.matmul` | 11.6 TFLOP/s（理论 ~12.3） |
| 全局内存 读 / 拷贝 | **386 / 391 GB/s** |

**不要用 `has_subgroup_matrix_multiply_accumulate` 给 Kernel B 做能力门控。**
在当前驱动（`1.15.38308+4`）下该 SYCL 设备属性上报为 **`False`**，但设备显然有可用的矩阵引擎：
bf16 GEMM 达到 88 TFLOP/s，而 FP32 只有 11.6 TFLOP/s，7.6 倍的差距是向量单元不可能产生的。
这是运行时的上报问题，不是硬件缺失。若照该属性写门控，会错误地判定本机没有矩阵引擎。

#### 如何解读 TFLOP/s 这一列

该列**只**统计低秩投影（`2*M*K*R`），因为这是 Kernel A 中唯一有实质算力的部分；
smooth、amax 规约、`log2`、E2M1 编码与 nibble 打包几乎不产生 FLOP，却占据了绝大部分运行时间。
在 `4608x3072 R=32` 下它是 0.906 GFLOP，仅为 Kernel B 将要完成的对应主 GEMM
（`4608x3072x3072`，87 GFLOP）的 **1.04%**。因此这个数字**不能**与已发表的 W4A4 GEMM 吞吐
（例如 Nunchaku 的 600~1000 TFLOP/s）相比：后者描述的是 NVIDIA FP4 tensor core 上的*主 GEMM*，
而前者描述的是一个访存受限的量化 kernel 内部的细秩-32 旁路。
即使 Kernel B 在 B60 上写到完美，也被上面那 88 TFLOP/s 封顶。

Kernel A 真正有意义的指标是**带宽**，按此衡量纯量化路径基本已经收工：
`4608x12288` 下 334 GB/s，为 386 GB/s 屋顶的 87%。

#### "rank-32 增量不超过 20%" 不成立，已降级为诊断项

同 shape 下实测增量约 296%，而非 20%。原因现已实测查明，
且本文档早先给出的解释（"compute bound"）是**错误的**：

| R | 总时间 | lora 增量 | ms/rank | bytes/FMA |
|---|---|---|---|---|
| 8 | 0.286 ms | 0.113 ms | 0.0141 | 2.00 |
| 16 | 0.370 ms | 0.197 ms | 0.0123 | 2.00 |
| 32 | 0.528 ms | 0.355 ms | 0.0111 | 2.00 |
| 64 | 0.934 ms | 0.761 ms | 0.0119 | 2.00 |

该分支受限于**取操作数，而非 FMA 单元**。两个特征可以证明：`ms/rank` 恒定，
说明时间随 R 严格线性增长；`bytes/FMA` 恰好等于 2.00，
意味着每一次 FMA 都要加载 2 字节 `lora_down`，**零寄存器复用**。
其后果是 `lora_down` 每行都被重读一遍——**R=32 时 906 MB，而 `x` 仅 28 MB**，相差 32 倍。
它之所以还能接受，仅仅因为 `lora_down` 只有 0.2 MB，可以常驻 cache。
由此得到的 ~2.5 TFLOP/s 约为 12.3 TFLOP/s FP32 向量屋顶的 20%。

解法是引入操作数复用，有两条路径：

1. **按行做寄存器分块**（`kRowsPerSubGroup > 1`）可把 `bytes/FMA` 减半到 1.00。
   在大 K 上确有收益（`4608x12288`：2.98 -> 2.62 ms），但目前在 `R = 32` 时被
   `acc[kRowsPerSubGroup][MaxRank]` 的寄存器压力卡住；值得在 large-GRF 模式下重试。
2. **把低秩投影改写为 `joint_matrix` / DPAS 形式**，脉动阵列天然提供操作数复用。
   ARK 中已有大量可参照的 `joint_matrix` 基础设施
   （`sycl_tla_dense_gemm.hpp`、`sycl_tla_moe_prefill_s4_dpas.hpp` 等约 10 个头文件）。

**因此 low-rank 累加是 Kernel A 目前最大的单项优化空间**，优先级高于访存侧工作，后者已接近屋顶。

### 4.4 Kernel A milestone

Kernel A 完成需同时满足：

1. golden/random/FLUX shape 数值 gate 全部通过。
2. 单 launch、current queue、无 host sync。
3. profiler 证明输入复用成立。
4. benchmark 结果可复现，并分别列出 quant-only、down-only、unfused 和 fused。
5. API 不依赖未验证的 Kernel B physical layout。

**对照上述标准的当前状态**：1、4、5 已满足（138 个 pytest 用例；§4.3 harness 并列给出 quant-only、
unfused 与 fused；ABI 独立于 Kernel B 冻结）。3 在实质意义上已满足——smoothed 输入根本不会被
物化，因为 §3.3.4 把 `smooth` 折进了低秩算子。

**标准 2 现已满足**：§3.3.5 已把投影折进量化内核，因此对 `x` 恰好只有一次遍历。还剩两次辅助
launch，且都不读 `x`：B 的 prologue 只碰一个 `[K, R<=64]` 的矩阵；当 K 被切分时，部分和归约只碰
`[S, M, R]`。它们全部提交在调用方的 queue 上，无 host 同步。

## 5. Kernel B：W4A4 GEMM + low-rank up

### 5.1 最终融合内容

目标机器上的 Kernel B 应在一次 launch 内完成：

1. 读取 Kernel A 的 qact/ascales。
2. 读取 native packed qweight/wscales。
3. 执行 MXFP4 W4A4 GEMM，并以 FP32 累积 residual output tile。
4. 计算 `lora_act @ lora_up.T`，累加到相同 output tile。
5. 加 bias。
6. cast 为输入 dtype 并写回 `[M,N]`。

不允许 materialize 完整 BF16/FP16 residual activation 或 residual weight。

### 5.2 B60 上完成的内容

第二阶段在 B60 上尽可能完成：

- 稳定的 C++/pybind/SYCL API、shape/dtype/device/queue/error contract。
- workspace query、overflow、alignment 和 bounds check。
- canonical weight/scale 与 native opaque blob 的 header/version contract。**直接沿用**
  `auto_round_kernel/ark.cpp` 里 `packed_weight_size` / `repack_quantized_weight` / `unpack_weight`
  这一组既有 API 的形状与职责划分（查询打包后大小 → 从 canonical 权重打包 → 从 blob 还原），不要为
  Kernel B 另造一套命名或调用顺序。
- reference/emulated Kernel B：decode + FP32 GEMM + low-rank up + bias + cast，仅用于 correctness。
- low-rank up/bias/output epilogue 的独立 SYCL 实现或可插拔组件。
- residual-only、low-rank-only、combined、bias/no-bias、M/N tail 测试。
- mock mainloop 接口，证明 native W4A4 accumulator tile 能无额外中间 tensor 接入 epilogue。
- FLUX shape corpus benchmark harness 和目标机器 bring-up 命令。
- capability gate 的代码骨架：仿照 `wrapper/include/sycl_tla_moe_prefill_s4_dpas.hpp` 中
  `moe_prefill_dpas_s4_enabled()`（env flag，默认可关闭/开启）与
  `moe_prefill_dpas_s4_pergroup_shape_ok()`（shape precondition）的写法，为 MXFP4 W4A4 mainloop 预留
  同构的 `svdquant_mxfp4_dpas_enabled()` / `svdquant_mxfp4_shape_ok()` 占位函数。在 B60 上这两个函数
  可以直接恒定返回 `false`（因为指令不可用），但函数签名、env flag 命名习惯（`ARK_SVDQUANT_MXFP4_DPAS`）
  和调用位置必须在 B60 阶段冻结，Phase 3 只替换函数体，不改调用点。

### 5.3 B60 上只做准备的内容

以下工作必须等目标机器，不在 B60 上宣称完成：

- MXFP4 matrix instruction 路线选择和正确性。
- native qweight/qact/scale swizzle 与 tile layout。
- W4A4 mainloop、subgroup、prefetch、SLM/register 配置。
- native blob payload layout 的最终版本。
- Kernel B 性能门槛和产品化结论。
- `svdquant_mxfp4_dpas_enabled()` / `svdquant_mxfp4_shape_ok()` 的真实实现（B60 阶段恒定返回
  `false`/reference-shape-only）。

不要写返回伪结果的 placeholder kernel。未接入 native mainloop 时，explicit fused 路径必须返回 unsupported；reference/emulated 路径必须在名称和日志中明确标识。

### 5.4 Kernel B 的 B60 milestone

B60 阶段只要求：

1. Kernel B ABI 与 Kernel A 输出兼容。
2. emulated B 对 PyTorch reference 通过数值测试。
3. low-rank up/bias/cast 组件可被 mock accumulator 驱动。
4. native mainloop 的插入点不泄漏到 portable checkpoint。
5. capability gate 不会在 B60 上选择 fused B。

不要求：

- 完整 native Kernel B launch 成功。
- 相对 BF16 GEMM 的性能提升。
- 冻结 target-specific layout 或 tile。

### 5.5 与现有 int4/int8 DPAS kernel 的关系（重要澄清）

`auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_moe_prefill_s4_dpas.hpp` 与
`sycl_tla_moe_prefill_int_dpas.hpp` 已经在本仓库里实现了一条“packed nibble 权重 → 按 K-group 反量化到
workspace → DPAS INT8/BF16 mainloop”的可运行路径，并带有 persistent scheduler、policy 模板类和 env-flag
capability gate。这不是 MXFP4 矩阵指令，但它是本仓库里**唯一**已经在真实 Xe DPAS 硬件上跑通的低比特
GEMM mainloop，因此：

- Phase 2/3 设计 Kernel B mainloop 的 dispatcher、launcher、workspace 生命周期时，应把这两个文件当作
  结构范本（调度器形状、policy 特化方式、env flag 命名），而不是从零设计。
- 如果目标机器上 native MXFP4 matrix instruction 的 bring-up 时间不可控，"S4 → workspace 反量化 →
  INT8/BF16 DPAS" 这条已验证路径可以作为 Kernel B 的**过渡 fallback**（先用 MXFP4 decode 得到 workspace
  再喂给现有 INT8/BF16 DPAS mainloop），以更快拿到一个在真实硬件上可测的性能基线，再决定是否值得投入
  原生 MXFP4 指令路线。是否采用此 fallback 应在 Phase 3 立项时明确记录为一个决策点，而不是默认路径。
- `NunchakuMXFP4Packer`（见 `auto_round/export/svdquant_mxfp4.py`）里的 `comp_n/comp_k/mem_k/num_n_lanes`
  等常量是 CUDA warp-level MMA 的物理布局，只服务于 Nunchaku CUDA runtime 的 checkpoint 互操作，**不是**
  ARK/Xe 的 native layout，不能被当作 Kernel B 的目标 tile/swizzle 参考；ARK 的 native opaque blob 必须
  由 Phase 3 针对 Xe DPAS 重新设计，并通过 `repack_quantized_weight` 风格的 API 从 canonical 权重生成。

## 6. 开发阶段

### Phase 1：Kernel A

1. 冻结 A 的 logical ABI 和 PyTorch oracle。
2. 实现 scale reduction 与 E2M1 pack。
3. 融合 smooth 和 low-rank down。
4. 完成 golden、random、tail、stream 和并发测试。
5. 完成 XPU profiler 与 benchmark gate。

**退出条件**：第 4 节全部满足。

### Phase 2：Kernel B preparation

1. 冻结 B 的 logical ABI 和 opaque native-pack boundary。
2. 完成 emulated B 和独立 epilogue。
3. 完成 mock mainloop、错误路径和 benchmark harness。
4. 输出目标机器 bring-up checklist。

**退出条件**：第 5.4 节全部满足；native W4A4 mainloop 明确标记为 deferred。

### Phase 3：目标机器 bring-up

1. 选择并验证 MXFP4 instruction route。
2. 冻结 native layout/tile。
3. 实现并接入 W4A4 mainloop。
4. 对两个 kernel 做 end-to-end parity、性能和 FLUX 集成验收。

## 7. 参考实现与复用点

### 7.1 论文与跨仓库数值/融合语义

- SVDQuant 论文（arXiv:2411.05007）：low-rank branch 与 low-bit branch 的融合动机及 Nunchaku 性能背景。
- Nunchaku：
  - `src/Linear.cpp`：`GEMM_W4A4::quantize()` 与 `forward_quant()` 的两阶段调用关系。
  - `src/kernels/zgemm/gemm_w4a4_launch_impl.cuh`：`quantize_w4a4_act_fuse_lora` 和 W4A4 epilogue 组合。
  - `src/kernels/zgemm/gemm_w4a4.cuh`：E2M1 quantization、W4A4 mainloop 和 LoRA epilogue。
- AutoRound：
  - `auto_round/data_type/mxfp.py`：`quant_mx_rceil` 数值 oracle，Kernel A 的唯一数值 ground truth。
  - `auto_round/export/svdquant_mxfp4.py`：`encode_e2m1`/`decode_e2m1`/`encode_ue8m0`/`decode_ue8m0`/
    `pack_nibbles`/`unpack_nibbles` 等纯逻辑（HW-independent）codec，Kernel A 的 `qact`/`ascales` 输出
    contract 应逐位对齐这些函数；`NunchakuMXFP4Packer` 则是 CUDA-only 物理布局（见 5.5 节说明，不可
    当作 ARK native layout）。
  - `test/unit/common/export/test_svdquant_mxfp4.py`：pack/unpack 与 AutoRound QDQ 一致性，可直接复用其
    随机/边界向量作为 Kernel A golden test 的种子。
  - `auto_round/algorithms/transforms/svdquant/`（`residual.py`/`apply.py`/`wrapper.py`）：SVD 分解、
    分组、以及 `Q(R) + U@V` 的 PyTorch 前向参考实现，是 Kernel A+B 端到端语义唯一权威来源。

### 7.2 ARK 仓库内可直接复用的工程模式

这些不是 SVDQuant 专属代码，但定义了本仓库现有 kernel 的工程约定，Kernel A/B 应该复用而不是另建一套：

| 关注点 | 复用对象 | 说明 |
|---|---|---|
| queue 获取 | `auto_round_kernel/__init__.py::get_stream` | 从输入 tensor 推导 `sycl_queue`，避免额外的 queue 传参约定 |
| dtype 枚举 | `auto_round_kernel/__init__.py::ARK_DT`（含 `float8_e8m0`） | 新增 E2M1/MXFP4 code 时在此枚举追加常量 |
| canonical→native blob 边界 | `ark.cpp::packed_weight_size` / `repack_quantized_weight` / `unpack_weight` | Kernel B native blob 的查询/打包/还原三段式 API 范本 |
| Python 侧 ABI 前置校验 | `qlinear.py`/`__init__.py` 的 `_validate_packed_blob` 一类写法 | 在调用原生 kernel 前重复校验第 4.2 节的 ABI 契约 |
| 扩展加载 | `xpu_loader.py::ensure_xpu_lib`/`load_xpu_lib` | 新 kernel `.so` 通过同一套 `required_symbols` 机制暴露 |
| 低比特 DPAS mainloop 结构范本 | `wrapper/include/sycl_tla_moe_prefill_s4_dpas.hpp`、`sycl_tla_moe_prefill_int_dpas.hpp` | persistent scheduler、policy 模板类、env-flag capability gate 的现成范本（细节见 5.5 节） |
| pybind 符号注册 | `ark.cpp` 中 `PYBIND11_MODULE(PY_NAME, m)` 内的 `m.def(...)` 列表 | 新 kernel 的 C++ 入口点在此登记，不新建 pybind module |
| 构建注册 | `auto_round_kernel/CMakeLists.txt`（`file(GLOB SRCS *.cpp)` 自动收集，但 pybind 符号仍需手动登记） | 新增 `.cpp` 放在该目录下会被自动编译，但必须同步在 `ark.cpp` 里 `m.def` |

### 7.3 复用边界

复用时只继承数值语义和融合边界，不复制 CUDA-specific warp、layout 或 launch 参数；ARK 仓库内的工程模式
（第 7.2 节）可以复用实现细节和调用约定，但 Kernel B 的 native tile/swizzle 仍必须在目标机器上重新设计
（见 5.5 节）。

## 8. Kernel 开发之外的落地 Gap

只做好 Kernel A/B 本身不足以让这个功能"落地"。以下 gap 独立于 B60/目标机器的限制，需要在两个 Phase
中各自安排负责人和退出条件，不能等到 Kernel 完成后才发现：

### 8.1 Python 集成与 dispatch

目前仓库里没有任何 ARK SVDQuant 的 inference-time 入口：`auto_round/algorithms/transforms/svdquant/
wrapper.py` 的运行时前向只有 PyTorch 参考实现和面向 Nunchaku/CUDA 的导出路径
（`auto_round/export/svdquant_nunchaku.py`），vLLM 侧的 `auto_round_extension/vllm_ext/linear_impl_mxfp4.py`
也只覆盖 vLLM 后端。需要新增（建议放在 `auto_round_extension/ark/auto_round_kernel/`，与 `qlinear.py`
同级）一个 `svdquant_mxfp4_linear.py`（或类似命名）：

- 提供一个可被 `SVDQuantLinear` 或独立测试直接调用的 Python 包装函数/`nn.Module`，形态参照
  `qlinear.py::QuantLinear`。
- 内部按 5.4 节的 capability gate 结果选择「fused Kernel A + native Kernel B」或「fused Kernel A +
  emulated Kernel B」或「显式报错」三条路径之一，不允许静默退化为纯 PyTorch。
- 这一集成点的接口冻结属于 Phase 1 出口条件的一部分（Kernel A 的 ABI 必须先被这一层消费一次，
  才能证明 ABI 在真实调用场景下可用），不要推迟到 Phase 3。

### 8.2 构建、符号注册与打包

- 新增的 `.cpp`/`.hpp` 需要放进 `auto_round_kernel/` 对应目录；`CMakeLists.txt` 的
  `file(GLOB SRCS *.cpp)` 会自动纳入编译，但 **pybind 符号必须手动加入 `ark.cpp` 的
  `PYBIND11_MODULE` 列表**，否则 Python 侧拿不到入口。
- `auto_round_kernel/__init__.py` 的 `ARK_DT` 需要追加 MXFP4/E2M1 相关常量；`xpu_loader.py` 的
  `required_symbols` 列表（由调用方在 `ensure_xpu_lib` 时传入）需要包含新符号名，否则加载器会静默
  找到旧版本 `.so` 并在运行时才报 `AttributeError`。
- `auto_round_kernel/version.py` 按仓库惯例需要在新增能力时同步提示版本兼容性。
- 这些是纯工程性 checklist 项，建议在 Phase 1 PR 描述里逐条勾选，不需要单独设计，但必须显式列出，
  否则容易出现"kernel 编译通过但 Python 侧调用不到"的返工。

### 8.3 Checkpoint / 导出格式边界

- 当前 `svdquant_nunchaku` 导出格式（`auto_round/export/svdquant_nunchaku.py`、
  `auto_round/export/formats/backends/svdquant_nunchaku.py`）产出的是 Nunchaku CUDA runtime 可加载的
  物理布局，和 ARK 的 native blob 是两回事。
- 需要显式决定：ARK 的 checkpoint 输入是（a）复用 `svdquant_mxfp4.py` 里的 **logical/canonical**
  qact/qweight/scale（简单、可移植，但需要 load 时在线 `repack_quantized_weight` 转成 native），还是
  （b）新增一个 `svdquant_ark` 导出格式直接落盘 native blob（load 快，但导出格式和硬件强绑定，未来
  target layout 变化需要重新导出）。
- 这个决定不阻塞 Kernel A/B 本身的开发，但阻塞"能不能跑一个真实 FLUX checkpoint"，建议在 Phase 2
  末期、Phase 3 开始前拍板，避免 native layout 冻结后才发现导出链路缺失。

### 8.4 测试与 benchmark harness 复用

- `auto_round_extension/ark/test/` 已有 `conftest.py`、`ut_utils.py` 等公共 fixture，以及
  `test_moe_prefill_accuracy.py`/`test_moe_prefill_perf.py` 这类"同一 kernel 既有 accuracy 测试又有
  perf 测试"的目录结构；Kernel A/B 的测试文件应放在同一目录并复用同一套 fixture/timing 工具
  （对应第 4.3 节的 XPU event 计时方法应直接抽取自现有 perf 测试的公共代码，不要重写一套计时逻辑）。
- `benchmarks/` 目录下已有 `bench_sparse_topk.py` 这类独立 benchmark 脚本风格，Kernel A 的 FLUX
  shape benchmark 应遵循同一 CLI/输出格式，方便和其他 kernel 的结果放在一起比较。

### 8.5 文档中英同步

- 本设计文档的中英文版本（`ark_SVDQuant_mxfp4_design_doc.md` / `_eng.md`）必须保持结构和结论同步；
  按仓库规则，任何 `.md` 变更都需要同步更新对应 `_CN`/英文版本。后续如果本功能的用户可见文档（例如
  `ark/README.md`/`README_CN.md`，或顶层 `docs/svdquant_details.md`/`_CN.md`）发生变化，同样需要
  中英文同步更新，并在 review 中显式检查。

### 8.6 硬件能力探测的占位约定

- 在拿不到目标机器之前，5.2 节要求的 `svdquant_mxfp4_dpas_enabled()` 只能恒定返回 `false`。这是**唯一**
  允许"先写死"的判断点；除此之外的所有 shape/dtype/ABI 校验都必须是真实校验，不能用"反正拿不到目标
  机器"作为借口简化。
- Phase 3 拿到目标机器后，第一件事是把这个函数替换成真实探测（例如查询 SYCL device aspect 或已知
  device id 白名单），而不是重新设计调用点。

### 8.7 模型级验收基线的来源（可选，供后续参考）

- 如果本功能未来需要模型级（而非仅 kernel 数值）验收，仓库里已有一份面向 SVDQuant INT4 参考精度研究
  的方法论文档（`.github/instructions/svdquant-int4-reference.instructions.md`，位于
  `copilot/svdquant-int4-reference-accuracy` 分支，尚未合并主干），其中记录了 FLUX.1-dev 上 BF16 与
  MXFP4 SVDQuant 的 CLIP/CLIP-IQA/ImageReward 基线数值，以及一套"先验证 controls 可复现、再扫描、再
  gate"的方法论。ARK 项目如果走到模型级验收阶段，应直接移植/引用这份方法论和基线数值，而不是重新
  发明门槛或标准；但这属于 Phase 3 之后的可选后续工作，不是本设计文档 Phase 1/2 的出口条件。
