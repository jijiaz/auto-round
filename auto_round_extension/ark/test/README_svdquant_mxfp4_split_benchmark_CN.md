# Kernel A Split Baseline Benchmark 实现说明

目标：把 `bench_svdquant_mxfp4.py` 中当前的 PyTorch 三步 baseline 替换为 ARK split baseline：

```text
ARK smooth + quant
+ ARK LoRA operand preparation
+ ARK standalone LoRA projection
```

然后将三步耗时之和与当前 fused Kernel A 比较。新增接口只服务 benchmark，不需要作为完整生产 API 设计。

## 1. 复用现有实现

不要在 Python benchmark 中重新实现 kernel。

### Smooth + quant

参考：

```text
auto_round_kernel/svdquant_mxfp4_kernel.cpp
```

复用现有：

```cpp
launch_quant_only<T, UseDoubleLog>()
```

该路径已经完成 smooth、amax、scale、E2M1 encode、pack 和 `qact/ascales` 写回。新增一个薄 wrapper，例如：

```cpp
svdquant_mxfp4_smooth_quant(...)
```

它只调用 `launch_quant_only`，不需要新增设备端 quant 逻辑。

### LoRA operand preparation

参考：

```text
auto_round_kernel/svdquant_mxfp4_cute_kernel.cpp
```

复用：

```cpp
launch_pack_lora_b_cute()
```

它生成当前 CUTE/DPAS 路径使用的：

```text
B = smooth * lora_down
B ~= B_hi + B_lo
```

新增一个薄 wrapper，例如：

```cpp
svdquant_mxfp4_prepare_lora(...)
```

输出 `hi` 和 `lo`，layout、padding 和 dtype 必须直接沿用 `cute_b_cols()` 与现有 workspace 规则。

### LoRA projection

参考：

```text
auto_round_kernel/svdquant_mxfp4_cute_kernel.cpp
```

复用：

```cpp
launch_lora_cute()
```

该函数使用与 fused path 相同的 CUTE/DPAS projection。新增一个薄 wrapper，例如：

```cpp
svdquant_mxfp4_lora_down(...)
```

不要在 split baseline 中使用 `torch.matmul()` 或重新实现 FP32 vector FMA，否则比较的不是同一个 ARK projection。

## 2. 注册 C++ 接口

依次修改：

1. 在 `sycl_svdquant_mxfp4.hpp` 或对应 CUTE header 中添加 host 函数声明。
2. 在 `svdquant_mxfp4_kernel.cpp` 中添加三个 host wrapper。
3. 在 `ark.cpp` 的 `PYBIND11_MODULE` 中注册：

```cpp
m.def("svdquant_mxfp4_smooth_quant", ...);
m.def("svdquant_mxfp4_prepare_lora", ...);
m.def("svdquant_mxfp4_lora_down", ...);
```

4. 在 Python `svdquant_mxfp4.py` 中通过 `ensure_xpu_lib(required_symbols=...)` 获取这三个符号。
5. Python wrapper 负责分配输出和 workspace，并把 tensor 转为 contiguous；benchmark 不直接传 raw pointer。

如果使用现有的 fused CUTE workspace layout，先复用：

```cpp
svdquant_workspace_elements()
cute_b_plane_elements()
cute_b_cols()
```

## 3. 修改 benchmark

文件：

```text
auto_round_extension/ark/test/bench_svdquant_mxfp4.py
```

在 `run_shape()` 中预分配：

```text
qact
ascales
hi
lo
lora_act
```

分别计时三个 ARK 操作：

```text
smooth_quant_p50
prepare_lora_p50
lora_down_p50
```

使用现有 `time_ms()`，保持相同的 warmup、iteration 和 XPU event 计时方式。

split baseline 定义为：

```python
split_sum = smooth_quant_p50 + prepare_lora_p50 + lora_down_p50
```

fused 路径仍使用：

```python
svdquant_mxfp4_quant_down(
    x, smooth, lora_down, backend="ark"
)
```

speedup 改为：

```python
speedup = split_sum / fused_p50
```

表头中的旧列：

```text
smooth quant gemm sum
```

改为：

```text
smooth+quant prepare_lora projection split_sum
```

其中 `projection` 指 ARK standalone CUTE/DPAS projection，不是 PyTorch GEMM。

## 4. quant-only 行

`R=0` 时没有 LoRA operand preparation 和 projection，只保留当前 ARK quant-only kernel 的耗时。

不要把 quant-only 行伪造为三个步骤之和；它应单独显示：

```text
quant-only fused
```

## 5. 正确性检查

新增 split wrapper 后，在 benchmark 或测试中确认：

```python
split_qact == fused_qact
split_ascales == fused_ascales
```

`lora_act` 使用现有 tolerance 检查，不要求不同 launch 之间 bit-exact。

## 6. 编译和运行

在已配置 oneAPI 和 `ark` conda 环境的 shell 中执行：

```bash
cd /home/jijiaz/auto-round/auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh
python -m pip install --no-build-isolation --no-deps -e .
```

运行 BF16 和 FP16 benchmark：

```bash
python test/bench_svdquant_mxfp4.py --dtype bf16 --iters 200
python test/bench_svdquant_mxfp4.py --dtype fp16 --iters 200
```

完成标准：

- benchmark 不再调用 `baseline_smooth`、`baseline_quant_pack` 或 `baseline_lora`；
- `smooth+quant`、`prepare_lora`、`projection` 都来自 ARK wrapper；
- `split_sum / fused_p50` 正常输出；
- `qact`/`ascales` 与 fused path 一致；
- BF16 和 FP16 均能运行。
