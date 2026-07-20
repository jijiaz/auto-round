# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional
import torch
import sys


class ARK_DT:
    float64 = 64
    float32 = 32
    float16 = 16
    bfloat16 = 65552
    int2 = 258
    int3 = 259
    int4 = 260
    int5 = 261
    int6 = 262
    int7 = 263
    int8 = 264
    int32 = 288
    float8_e4m3 = 8
    float8_e5m2 = 65544
    float8_e8m0 = 196616
    undef = 0


def cvt_dtype(dtype):
    if dtype == torch.float32:
        return ARK_DT.float32
    if dtype == torch.float16:
        return ARK_DT.float16
    if dtype == torch.bfloat16:
        return ARK_DT.bfloat16
    if dtype == torch.float8_e4m3fn:
        return ARK_DT.float8_e4m3
    if dtype == torch.float8_e5m2:
        return ARK_DT.float8_e5m2
    if dtype == torch.int8:
        return ARK_DT.int8
    if dtype == torch.int32:
        return ARK_DT.int32
    return ARK_DT.undef


def cvtstr_dtype(dtype):
    if dtype == "fp32":
        return ARK_DT.float32
    if dtype == "fp16":
        return ARK_DT.float16
    if dtype == "bf16":
        return ARK_DT.bfloat16
    if dtype == "fp8_e4m3":
        return ARK_DT.float8_e4m3
    if dtype == "fp8_e5m2":
        return ARK_DT.float8_e5m2
    if dtype == "fp8_e8m0":
        return ARK_DT.float8_e8m0
    if dtype == "int8":
        return ARK_DT.int8
    if dtype == "int4":
        return ARK_DT.int4
    if dtype == "int2":
        return ARK_DT.int2
    if dtype == "int3":
        return ARK_DT.int3
    if dtype == "int5":
        return ARK_DT.int5
    if dtype == "int6":
        return ARK_DT.int6
    if dtype == "int7":
        return ARK_DT.int7
    if dtype == "int32":
        return ARK_DT.int32
    return ARK_DT.undef


def get_stream(A: torch.Tensor) -> int:
    if A.device.type == "cpu":
        return 0
    if A.device.type == "xpu":
        return torch.xpu.current_stream().sycl_queue


def _normalize_tensor_layout(tensor_layout: str) -> str:
    layout = tensor_layout.upper()
    if layout not in ("HND", "NHD"):
        raise ValueError(f"tensor_layout must be either 'HND' or 'NHD', got {tensor_layout!r}")
    return layout


def _attention_shape(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, int, int, int]:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        batch, num_heads, seq_len, head_dim = tensor.shape
    else:
        batch, seq_len, num_heads, head_dim = tensor.shape
    return batch, num_heads, seq_len, head_dim


def _attention_strides_qko(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, int, int, int]:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        batch_stride, head_stride, seq_stride, dim_stride = tensor.stride()
    else:
        batch_stride, seq_stride, head_stride, dim_stride = tensor.stride()
    return seq_stride, dim_stride, head_stride, batch_stride


def _attention_strides_v(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, int, int, int]:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        batch_stride, head_stride, seq_stride, dim_stride = tensor.stride()
    else:
        batch_stride, seq_stride, head_stride, dim_stride = tensor.stride()
    return dim_stride, seq_stride, head_stride, batch_stride


def _validate_attention_tensor(
    tensor: torch.Tensor,
    name: str,
    tensor_layout: str,
    *,
    expected_dtype: torch.dtype | None = None,
) -> tuple[int, int, int, int]:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be a 4D tensor")
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}")

    qko_strides = _attention_strides_qko(tensor, tensor_layout)
    if qko_strides[1] != 1:
        raise ValueError(f"{name} must be contiguous along the head-dim axis; got stride {qko_strides[1]}")
    if any(stride <= 0 for stride in qko_strides):
        raise ValueError(f"{name} must have positive non-zero strides, got {tensor.stride()}")

    return _attention_shape(tensor, tensor_layout)


def _empty_attention_output(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    tensor_layout: str,
) -> torch.Tensor:
    layout = _normalize_tensor_layout(tensor_layout)
    shape = (batch, num_heads, seq_len, head_dim) if layout == "HND" else (batch, seq_len, num_heads, head_dim)
    return torch.empty(shape, device=device, dtype=dtype)


@dataclass
class _CpuPackedKVCacheEntry:
    descriptor: object
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    seq_len: int
    key_version: int
    value_version: int


_CPU_PUBLIC_PACKED_KV_CACHE_MAX = 8
_CPU_PUBLIC_PACKED_KV_CACHE: "OrderedDict[tuple, _CpuPackedKVCacheEntry]" = OrderedDict()


def _cpu_public_packed_kv_available() -> bool:
    return (
        cpu_lib is not None
        and hasattr(cpu_lib, "ark_cpu_bestla_sdpa_packed_desc")
        and hasattr(cpu_lib, "ark_cpu_update_packed_k_desc")
        and hasattr(cpu_lib, "ark_cpu_update_packed_v_desc")
    )


def _cpu_public_packed_kv_cache_key(key: torch.Tensor, value: torch.Tensor, tensor_layout: str) -> tuple:
    batch, num_heads_kv, _, head_dim = _attention_shape(key, tensor_layout)
    return (
        key.device.type,
        key.device.index,
        key.dtype,
        value.dtype,
        key.data_ptr(),
        value.data_ptr(),
        batch,
        num_heads_kv,
        head_dim,
        _normalize_tensor_layout(tensor_layout),
    )


def _attention_seq_slice(tensor: torch.Tensor, tensor_layout: str, start: int, end: int) -> torch.Tensor:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        return tensor[:, :, start:end, :]
    return tensor[:, start:end, :, :]


def _cpu_public_get_packed_kv_entry(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tensor_layout: str,
) -> tuple[_CpuPackedKVCacheEntry, int]:
    layout = _normalize_tensor_layout(tensor_layout)
    batch, num_heads_kv, seq_len_kv, head_dim = _attention_shape(key, layout)
    cache_key = _cpu_public_packed_kv_cache_key(key, value, layout)
    key_version = int(key._version)
    value_version = int(value._version)
    entry = _CPU_PUBLIC_PACKED_KV_CACHE.get(cache_key)

    if entry is None or seq_len_kv > int(entry.descriptor.logical_capacity):
        descriptor = ark_cpu_packed_kv_descriptor(batch, num_heads_kv, seq_len_kv, head_dim, dtype=key.dtype)
        cache_k, cache_v = ark_cpu_packed_kv_alloc_from_descriptor(descriptor, dtype=key.dtype, device=key.device)
        entry = _CpuPackedKVCacheEntry(descriptor, cache_k, cache_v, 0, -1, -1)
        _CPU_PUBLIC_PACKED_KV_CACHE[cache_key] = entry
    else:
        _CPU_PUBLIC_PACKED_KV_CACHE.move_to_end(cache_key)

    if len(_CPU_PUBLIC_PACKED_KV_CACHE) > _CPU_PUBLIC_PACKED_KV_CACHE_MAX:
        _CPU_PUBLIC_PACKED_KV_CACHE.popitem(last=False)

    if entry.key_version == key_version and entry.value_version == value_version:
        if seq_len_kv > entry.seq_len:
            tail_k = _attention_seq_slice(key, layout, entry.seq_len, seq_len_kv)
            tail_v = _attention_seq_slice(value, layout, entry.seq_len, seq_len_kv)
            ark_cpu_update_packed_kv_from_descriptor(
                entry.descriptor,
                entry.cache_k,
                entry.cache_v,
                tail_k,
                tail_v,
                entry.seq_len,
                tensor_layout=layout,
                no_zeroing=False,
            )
            entry.seq_len = seq_len_kv
        return entry, seq_len_kv

    ark_cpu_update_packed_kv_from_descriptor(
        entry.descriptor,
        entry.cache_k,
        entry.cache_v,
        key,
        value,
        0,
        tensor_layout=layout,
        no_zeroing=False,
    )
    entry.seq_len = seq_len_kv
    entry.key_version = key_version
    entry.value_version = value_version
    return entry, seq_len_kv


def _cpu_public_mixed_sdpa_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    scale: float | None,
    tensor_layout: str,
) -> torch.Tensor:
    entry, seq_len_kv = _cpu_public_get_packed_kv_entry(key, value, tensor_layout=tensor_layout)
    return ark_cpu_bestla_sdpa_packed_from_descriptor(
        entry.descriptor,
        query,
        entry.cache_k,
        entry.cache_v,
        seq_len_kv,
        is_causal=is_causal,
        scale=scale,
        tensor_layout=tensor_layout,
    )


def _validate_attention_mask(
    attn_mask: torch.Tensor | None,
    *,
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
) -> None:
    if attn_mask is None:
        return
    if attn_mask.device != device:
        raise ValueError("attn_mask must be on the same device as Q")
    if not attn_mask.is_contiguous():
        raise ValueError("attn_mask must be contiguous")
    if attn_mask.dtype != torch.float32:
        raise ValueError(f"attn_mask must be float32 (additive bias), got {attn_mask.dtype}")
    expected_mask_shape = (batch, 1, seq_len_q, seq_len_kv)
    if attn_mask.shape != expected_mask_shape:
        raise ValueError(f"attn_mask shape must be {expected_mask_shape}, got {tuple(attn_mask.shape)}")


def _validate_no_dropout(dropout_p: float, api_name: str) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError(f"{api_name}: dropout_p must be 0.0 (got {dropout_p}); dropout is not supported")


def _validate_head_ratio(num_heads_q: int, num_heads_kv: int) -> None:
    if num_heads_kv <= 0:
        raise ValueError("num_heads_kv must be greater than 0")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv}) for MQA/GQA attention"
        )


def _validate_attention_geometry(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tensor_layout: str,
    *,
    key_dtype: torch.dtype | None = None,
    value_dtype: torch.dtype | None = None,
) -> tuple[int, int, int, int, int, int]:
    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=key_dtype)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout, expected_dtype=value_dtype)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    _validate_head_ratio(Hq, Hkv)
    return B, Hq, Hkv, Sq, Skv, D


def _contiguous_hnd_qko_strides(num_heads: int, seq_len: int, head_dim: int) -> tuple[int, int, int, int]:
    return head_dim, 1, seq_len * head_dim, num_heads * seq_len * head_dim


def _contiguous_hnd_v_strides(num_heads: int, seq_len: int, head_dim: int) -> tuple[int, int, int, int]:
    return 1, head_dim, seq_len * head_dim, num_heads * seq_len * head_dim


def _torch_dtype_from_ark_dtype(dtype: int) -> torch.dtype:
    if dtype == ARK_DT.float16:
        return torch.float16
    if dtype == ARK_DT.bfloat16:
        return torch.bfloat16
    if dtype == ARK_DT.float32:
        return torch.float32
    raise ValueError(f"Unsupported ARK dtype code: {dtype}")


def _normalize_batch_padding(n_padding, batch: int):
    if n_padding is None:
        return None
    if isinstance(n_padding, int):
        return int(n_padding) if n_padding > 0 else None
    if isinstance(n_padding, torch.Tensor):
        if n_padding.ndim != 1 or n_padding.numel() != batch:
            raise ValueError(f"n_padding tensor must be 1D with {batch} elements, got shape {tuple(n_padding.shape)}")
        return [int(v) for v in n_padding.to(device="cpu", dtype=torch.int64).tolist()]
    if isinstance(n_padding, Sequence) and not isinstance(n_padding, (str, bytes)):
        values = [int(v) for v in n_padding]
        if len(values) == 0:
            return None
        if len(values) != batch:
            raise ValueError(f"n_padding sequence must have length {batch}, got {len(values)}")
        return values
    raise TypeError("n_padding must be None, an int, a 1D tensor, or a length-batch sequence of ints")


@dataclass(frozen=True)
class _XPUKVCacheMeta:
    batch: int
    num_heads_kv: int
    capacity: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    storage_layout: str = "HND"
    storage_format: str = "contiguous"

    @classmethod
    def from_tensors(cls, key_cache: torch.Tensor, value_cache: torch.Tensor) -> "_XPUKVCacheMeta":
        if key_cache.device.type != "xpu" or value_cache.device.type != "xpu":
            raise ValueError("XPU KV cache tensors must live on an XPU device")
        if key_cache.dtype != value_cache.dtype:
            raise ValueError("K/V cache tensors must have identical dtype")
        if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
            raise ValueError("K/V cache tensors must be 4D tensors with identical shape")
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            raise ValueError("K/V cache tensors must be contiguous")
        batch, num_heads_kv, capacity, head_dim = key_cache.shape
        if key_cache.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported XPU KV cache dtype: {key_cache.dtype}")
        return cls(batch, num_heads_kv, capacity, head_dim, key_cache.dtype, key_cache.device)


# -----------------------------------------------------------------------------
# Module-level lib loading (replaces the previous singleton ``ARK`` class).
# -----------------------------------------------------------------------------

cpu_lib = None
xpu_lib = None

try:
    from . import auto_round_kernel_cpu as _cpu_lib_mod

    cpu_lib = _cpu_lib_mod
except ImportError as _e:
    print(f"ARK is unable to load CPU lib: {_e}")
    cpu_lib = None

if torch.xpu.is_available():
    try:
        from . import auto_round_kernel_xpu as _xpu_lib_mod

        xpu_lib = _xpu_lib_mod
    except ImportError as _e:
        print(f"ARK is unable to load XPU lib: {_e}")
        xpu_lib = None


def get_lib(A: torch.Tensor):
    lib = None
    if A.device.type == "xpu":
        lib = xpu_lib
    if A.device.type == "cpu":
        lib = cpu_lib
    if lib is None:
        raise NotImplementedError(f"Current device {A.device} is not supported")
    return lib


# A: mxk,  B: nxk, bias: n
def matmul(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    m = A.shape[0]
    n = B.shape[0]
    k = B.shape[1]
    lib = get_lib(A)
    ctype = A.dtype
    if A.device.type == "cpu":
        ctype = torch.float32
    C = torch.zeros(m, n, dtype=ctype, device=A.device)
    stream = get_stream(A)
    lib.matmul(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        cvt_dtype(B.dtype),
        C.contiguous().data_ptr(),
        cvt_dtype(C.dtype),
        bias.to(C.dtype).contiguous().data_ptr(),
        True,
    )
    return C


# A: mxk:s8,  B: nxk:s8, return: mxn:s32
def igemm_s8s8s32(A: torch.Tensor, B: torch.Tensor):
    m = A.shape[0]
    n = B.shape[0]
    k = B.shape[1]
    lib = get_lib(A)
    if lib is None:
        raise NotImplementedError(f"Current device {A.device} is not supported")
    C = torch.zeros(m, n, dtype=torch.int32, device=A.device)
    stream = get_stream(A)
    lib.matmul(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        cvt_dtype(B.dtype),
        C.contiguous().data_ptr(),
        cvt_dtype(C.dtype),
        0,
        True,
    )
    return C


# A: mxk:DT,  B: nxk:s8, scaleB: n:DT
# return: mxn:DT
def woqgemm_s8(A: torch.Tensor, B: torch.Tensor, scaleB: torch.Tensor, bias: torch.Tensor):
    m = A.shape[0]
    n = B.shape[0]
    k = B.shape[1]
    lib = get_lib(A)

    C = torch.zeros(m, n, dtype=A.dtype, device=A.device)
    stream = get_stream(A)
    lib.woqgemm_s8(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        C.contiguous().data_ptr(),
        bias.contiguous().data_ptr(),
        True,
        scaleB.contiguous().data_ptr(),
    )
    return C


# A: mxk:DT,  B: BS:s8, bias: n:DT
# return: C: mxn:DT
def woqgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    n,
    k,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    m = A.shape[0]
    lib = get_lib(A)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)
    st = cvtstr_dtype(scale_type)
    C = torch.zeros(m, n, dtype=A.dtype, device=A.device)
    stream = get_stream(A)
    lib.woqgemm(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        C.contiguous().data_ptr(),
        bias.contiguous().data_ptr(),
        groupsize,
        ct,
        wt,
        st,
        asym,
    )
    return C


# QB: k*n:int8,  scaleB: k/blocksize*n:DT
# return: blob:BS:int8
def _repack_quantized_weight_core(
    QB: torch.Tensor,
    scaleB: torch.Tensor,
    zp: torch.Tensor,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    k = QB.shape[0]
    n = QB.shape[1]
    lib = get_lib(QB)
    stream = get_stream(QB)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)
    st = cvtstr_dtype(scale_type)
    BS = lib.packed_weight_size(stream, n, k, groupsize, ct, wt, st, asym)
    blob = torch.zeros(BS, dtype=torch.int8, device=QB.device)
    lib.repack_quantized_weight(
        stream,
        QB.contiguous().data_ptr(),
        zp.contiguous().data_ptr(),
        scaleB.contiguous().data_ptr(),
        blob.data_ptr(),
        n,
        k,
        groupsize,
        ct,
        wt,
        st,
        asym,
    )
    return blob


# QB: blob:BS:int8
# return: out:nxk:out_dtype
def _unpack_weight_core(
    blob: torch.Tensor,
    out_dtype: torch.dtype,
    n,
    k,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    lib = get_lib(blob)
    stream = get_stream(blob)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)

    st = cvtstr_dtype(scale_type)
    oshape = (n, k) if blob.device.type == "xpu" else (k, n)
    out = torch.zeros(oshape, dtype=out_dtype, device=blob.device)
    lib.unpack_weight(
        stream,
        blob.data_ptr(),
        out.data_ptr(),
        cvt_dtype(out_dtype),
        n,
        k,
        groupsize,
        ct,
        wt,
        st,
        asym,
    )
    if blob.device.type == "cpu":
        return out.T
    return out


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    tensor_layout: str = "HND",
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - attn_mask: Additive float32 attention bias with shape [B, 1, Sq, Skv].
    - dropout_p: Must be 0.0; dropout is not supported.
    - is_causal: Apply the standard causal mask.
    - scale: Softmax scale. Uses 1 / sqrt(D) when None.
    - tensor_layout: Layout of Q/K/V/O tensors.

    Returns:
    - O: same layout as the input tensors.
    - (O, LSE): if return_lse is True on XPU.
    """
    if query.device.type not in ("cpu", "xpu"):
        raise NotImplementedError(f"sdpa is not supported on {query.device.type}")
    if query.device.type == "cpu" and return_lse:
        raise NotImplementedError("return_lse is not supported on CPU")

    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16) if query.device.type == "cpu" else (
        torch.float16,
        torch.bfloat16,
    )
    if query.dtype not in supported_dtypes:
        raise ValueError(f"Q dtype {query.dtype} is unsupported on {query.device.type}")

    # CPU BestLA mixed precision: F32 query with F16/BF16 K/V produces an F32
    # output. These are the only two cross-dtype combinations wired today; every
    # other combination still requires K/V to match Q. Homogeneous fp16/bf16 is
    # NOT a mixed combination and is unaffected by this branch.
    mixed_kv = (
        query.device.type == "cpu"
        and query.dtype == torch.float32
        and key.dtype == value.dtype
        and key.dtype in (torch.float16, torch.bfloat16)
    )
    if not mixed_kv and (key.dtype != query.dtype or value.dtype != query.dtype):
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Hkv, Sq, Skv, D = _validate_attention_geometry(
        query, key, value, tensor_layout, key_dtype=key.dtype, value_dtype=value.dtype
    )
    # The SYCL-TLA (XPU) flash-attention kernels are only compiled for a fixed
    # set of head dimensions. The CPU kernel supports arbitrary head_dim.
    if query.device.type == "xpu" and D not in (64, 128, 96, 192):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128, 96, 192")

    _validate_no_dropout(dropout_p, "sdpa")
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=Skv, device=query.device)

    lib = get_lib(query)
    stream = get_stream(query)
    # Mixed precision (F32 Q + F16/BF16 K/V) accumulates in and emits F32; the
    # homogeneous path keeps the operand dtype.
    out_dtype = torch.float32 if mixed_kv else value.dtype
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=out_dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    if query.device.type == "cpu":
        if mixed_kv and attn_mask is None and _cpu_public_packed_kv_available():
            return _cpu_public_mixed_sdpa_packed(
                query,
                key,
                value,
                is_causal=bool(is_causal),
                scale=scale,
                tensor_layout=tensor_layout,
            )
        q_strides = _attention_strides_qko(query, tensor_layout)
        k_strides = _attention_strides_qko(key, tensor_layout)
        v_strides = _attention_strides_v(value, tensor_layout)
        o_strides = _attention_strides_qko(O, tensor_layout)
        lib.sdpa(
            stream,
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            O.data_ptr(),
            attn_mask.data_ptr() if attn_mask is not None else 0,
            *q_strides,
            *k_strides,
            *v_strides,
            *o_strides,
            cvt_dtype(query.dtype),
            cvt_dtype(key.dtype),
            cvt_dtype(O.dtype),
            B,
            Hq,
            Hkv,
            Sq,
            Skv,
            D,
            float(scale) if scale is not None else 1.0 / (D**0.5),
            bool(is_causal),
            False,
            False,
            False,
            None,
        )
        return O

    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)

    LSE = torch.empty(B, Hq, Sq, dtype=torch.float32, device=query.device) if return_lse else None
    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sdpa(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        layout_code,
        LSE.data_ptr() if LSE is not None else 0,
    )
    if return_lse:
        return O, LSE
    return O


def debug_cpu_sdpa_route(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    tensor_layout: str = "HND",
    use_alibi: bool = False,
    use_tanh: bool = False,
    prefer_fp32: bool = False,
    n_padding=None,
) -> int:
    """Return the resolved internal CPU SDPA route for tests/debugging."""
    if query.device.type != "cpu":
        raise NotImplementedError("debug_cpu_sdpa_route is only supported on CPU")
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_debug_resolve_sdpa_route"):
        raise NotImplementedError("ARK CPU debug route resolver is not available")

    mixed_kv = (
        query.dtype == torch.float32
        and key.dtype == value.dtype
        and key.dtype in (torch.float16, torch.bfloat16)
    )
    if not mixed_kv and (key.dtype != query.dtype or value.dtype != query.dtype):
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")
    B, Hq, Hkv, Sq, Skv, D = _validate_attention_geometry(
        query, key, value, tensor_layout, key_dtype=key.dtype, value_dtype=value.dtype
    )
    normalized_n_padding = _normalize_batch_padding(n_padding, B)
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=Skv, device=query.device)

    out_dtype = torch.float32 if mixed_kv else value.dtype
    O = _empty_attention_output(B, Hq, Sq, D, dtype=out_dtype, device=query.device, tensor_layout=tensor_layout)
    q_strides = _attention_strides_qko(query, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    o_strides = _attention_strides_qko(O, tensor_layout)
    return cpu_lib.ark_cpu_debug_resolve_sdpa_route(
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        bool(use_alibi),
        bool(use_tanh),
        bool(prefer_fp32),
        normalized_n_padding,
    )


def sage(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    qscale: torch.Tensor = None,
    kscale: torch.Tensor = None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """SAGE attention prefill+decode.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - scale: Attention scale. Uses 1 / sqrt(D) when None.
    - quant_block_size: Block size for qscale and kscale.
    - tensor_layout: Layout of Q/K/V/O tensors.

    Returns:
    - O: same layout as the input tensors.
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sage is only supported on XPU")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError(f"sage expects int8 Q/K tensors, got Q={query.dtype}, K={key.dtype}")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"sage expects fp16/bf16 V tensors, got V={value.dtype}")
    if qscale is None or kscale is None:
        raise ValueError("qscale and kscale must be provided for sage")

    B, Hq, Hkv, Sq, Skv, D = _validate_attention_geometry(
        query, key, value, tensor_layout, key_dtype=torch.int8, value_dtype=value.dtype
    )
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")
    _validate_no_dropout(dropout_p, "sage")
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=Skv, device=query.device)

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)
    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sage(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        qscale.data_ptr() if qscale is not None else 0,
        kscale.data_ptr() if kscale is not None else 0,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        layout_code,
    )
    return O


def sage_pvi8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    qscale: torch.Tensor = None,
    kscale: torch.Tensor = None,
    vscale: torch.Tensor = None,
    out_dtype: torch.dtype = torch.float16,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Low-level SAGE attention with pre-quantized INT8 Q/K/V and PV int8.

    Expects contiguous layouts:
    - query: [B, Hq, Sq, D] int8
    - key: [B, Hkv, Skv, D] int8
    - value: [B, Hkv, Skv, D] int8
    - qscale: [B, Hq, ceil(Sq / quant_block_size), 1] float32
    - kscale: [B, Hkv, ceil(Skv / quant_block_size), 1] float32
    - vscale: [B, Hkv, ceil(Skv / quant_block_size), D] float32

    Returns:
    - O: [B, Hq, Sq, D] float16
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sage_pvi8 is only supported on XPU")
    if query.dtype != torch.int8 or key.dtype != torch.int8 or value.dtype != torch.int8:
        raise ValueError(f"Q/K/V must be int8, got Q={query.dtype}, K={key.dtype}, V={value.dtype}")
    if out_dtype != torch.float16:
        raise ValueError(f"sage_pvi8 output must be float16, got {out_dtype}")
    if qscale is None or kscale is None or vscale is None:
        raise ValueError("qscale, kscale and vscale must be provided for sage_pvi8")

    B, Hq, Hkv, Sq, Skv, D = _validate_attention_geometry(
        query, key, value, tensor_layout, key_dtype=torch.int8, value_dtype=torch.int8
    )
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")
    _validate_no_dropout(dropout_p, "sage_pvi8")
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=Skv, device=query.device)

    q_blocks = (Sq + quant_block_size - 1) // quant_block_size
    kv_blocks = (Skv + quant_block_size - 1) // quant_block_size
    if qscale.numel() != B * Hq * q_blocks:
        raise ValueError(
            f"qscale must have {B * Hq * q_blocks} elements for shape [B, Hq, ceil(Sq/block), 1], got {qscale.numel()}"
        )
    if kscale.numel() != B * Hkv * kv_blocks:
        raise ValueError(
            f"kscale must have {B * Hkv * kv_blocks} elements for shape [B, Hkv, ceil(Skv/block), 1], got {kscale.numel()}"
        )
    if vscale.numel() != B * Hkv * kv_blocks * D:
        raise ValueError(
            f"vscale must have {B * Hkv * kv_blocks * D} elements for shape [B, Hkv, ceil(Skv/block), D], got {vscale.numel()}"
        )

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=out_dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)
    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sage_pvi8(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        qscale.data_ptr(),
        kscale.data_ptr(),
        vscale.data_ptr(),
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        layout_code,
    )
    return O


def sagev1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
    return_lse: bool = False,
    smooth_k: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SAGE v1 attention prefill+decode.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - scale: Attention scale. Uses 1 / sqrt(D) when None.
    - quant_block_size: Quantization block size used by the kernel.
    - tensor_layout: Layout of Q/K/V/O tensors.

    Returns:
    - O: same layout as the input tensors.
    - (O, LSE): if return_lse is True.
    """
    if quant_block_size <= 0:
        return sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            tensor_layout=tensor_layout,
            return_lse=return_lse,
        )
    if query.device.type != "xpu":
        raise NotImplementedError("sagev1 is only supported on XPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Hkv, Sq, Skv, D = _validate_attention_geometry(
        query, key, value, tensor_layout, key_dtype=query.dtype, value_dtype=query.dtype
    )
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")
    _validate_no_dropout(dropout_p, "sagev1")
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=Skv, device=query.device)

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    LSE = torch.empty(B, Hq, Sq, dtype=torch.float32, device=query.device) if return_lse else None
    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)
    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sagev1(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(value.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        layout_code,
        bool(smooth_k),
        LSE.data_ptr() if LSE is not None else 0,
    )
    if return_lse:
        return O, LSE
    return O


def sagev1_pvi8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
    return_lse: bool = False,
    smooth_k: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SAGE v1 attention with PV int8 path.

    Expects FP16 Q/K/V input and quantizes Q/K/V internally before calling
    the PV int8 kernel.
    """
    if quant_block_size <= 0:
        return sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            tensor_layout=tensor_layout,
            return_lse=return_lse,
        )
    if query.device.type != "xpu":
        raise NotImplementedError("sagev1_pvi8 is only supported on XPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Hkv, Sq, Skv, D = _validate_attention_geometry(
        query, key, value, tensor_layout, key_dtype=query.dtype, value_dtype=query.dtype
    )
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")
    _validate_no_dropout(dropout_p, "sagev1_pvi8")
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=Skv, device=query.device)

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    LSE = torch.empty(B, Hq, Sq, dtype=torch.float32, device=query.device) if return_lse else None
    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)
    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sagev1_pvi8(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(value.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        layout_code,
        bool(smooth_k),
        LSE.data_ptr() if LSE is not None else 0,
    )
    if return_lse:
        return O, LSE
    return O


def ark_cpu_kv_cache_alloc(
    batch: int,
    num_heads_kv: int,
    capacity: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate an ARK CPU KV cache in internal HND layout: [B, Hkv, capacity, D]."""
    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError("ark_cpu_kv_cache_alloc only supports CPU tensors")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported KV cache dtype: {dtype}")
    shape = (batch, num_heads_kv, capacity, head_dim)
    return torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype)


def ark_cpu_kv_update(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    start_pos: int,
    *,
    tensor_layout: str = "HND",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append K/V tensors to an ARK CPU KV cache allocated by ``ark_cpu_kv_cache_alloc``."""
    if (
        key_cache.device.type != "cpu"
        or value_cache.device.type != "cpu"
        or key.device.type != "cpu"
        or value.device.type != "cpu"
    ):
        raise ValueError("ark_cpu_kv_update only supports CPU tensors")
    if key_cache.dtype != value_cache.dtype or key.dtype != key_cache.dtype or value.dtype != key_cache.dtype:
        raise ValueError("K/V cache and source tensors must have the same dtype")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("K/V caches must be 4D tensors with identical shape")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise ValueError("K/V caches must be contiguous")

    batch, num_heads_kv, capacity, head_dim = key_cache.shape
    Bk, Hkv, append_len, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=key_cache.dtype)
    Bv, Hkv2, append_len_v, Dv = _validate_attention_tensor(value, "V", tensor_layout, expected_dtype=key_cache.dtype)
    if (Bk, Bv) != (batch, batch) or Hkv != num_heads_kv or Hkv2 != num_heads_kv:
        raise ValueError("K/V source batch or head count does not match cache")
    if append_len_v != append_len or Dk != head_dim or Dv != head_dim:
        raise ValueError("K/V source shape does not match cache")
    if start_pos < 0 or start_pos + append_len > capacity:
        raise ValueError("KV append range exceeds cache capacity")
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_kv_update"):
        raise NotImplementedError("ARK CPU KV cache update kernel is not available")

    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    cpu_lib.ark_cpu_kv_update(
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        *k_strides,
        *v_strides,
        cvt_dtype(key_cache.dtype),
        batch,
        num_heads_kv,
        append_len,
        head_dim,
        capacity,
        int(start_pos),
    )
    return key_cache, value_cache


# -----------------------------------------------------------------------------
# Internal/experimental CPU mixed-route lifecycle helpers.
#
# These APIs exist to manage backend state (packed descriptors/caches/rope/packed
# forwards). They are intentionally outside the public sdpa() contract, which
# remains the standard SDPA surface.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ArkCpuPackedKVHandle:
    """Internal/experimental handle for the packed BestLA CPU KV-cache path."""
    descriptor: object
    dtype: torch.dtype

    @classmethod
    def create(
        cls,
        batch: int,
        num_heads_kv: int,
        capacity: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> "ArkCpuPackedKVHandle":
        return cls(ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, dtype=dtype), dtype)

    def info(self) -> dict:
        return ark_cpu_packed_kv_info(descriptor=self.descriptor)

    def alloc(self, *, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        return ark_cpu_packed_kv_alloc_from_descriptor(self.descriptor, dtype=self.dtype, device=device)

    def update(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
        *,
        tensor_layout: str = "HND",
        no_zeroing: bool = False,
    ) -> None:
        return ark_cpu_update_packed_kv_from_descriptor(
            self.descriptor, cache_k, cache_v, key, value, start_pos, tensor_layout=tensor_layout, no_zeroing=no_zeroing
        )

    def copy(
        self,
        dst_cache_k: torch.Tensor,
        dst_cache_v: torch.Tensor,
        src_cache_k: torch.Tensor,
        src_cache_v: torch.Tensor,
        seq_off: int,
        seq_size: int,
        *,
        no_zeroing: bool = False,
    ) -> None:
        return ark_cpu_copy_packed_kv_from_descriptor(
            self.descriptor, dst_cache_k, dst_cache_v, src_cache_k, src_cache_v, seq_off, seq_size, no_zeroing=no_zeroing
        )

    def shift_k(self, cache_k: torch.Tensor, cossin: torch.Tensor, *, seq_keep: int) -> None:
        return ark_cpu_shift_packed_k_from_descriptor(self.descriptor, cache_k, cossin, seq_keep=seq_keep)

    def forward(
        self,
        query: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        seq_len_kv: int,
        num_heads_kv: int | None = None,
        *,
        is_causal: bool = False,
        scale: Optional[float] = None,
        use_alibi: bool = False,
        use_tanh: bool = False,
        prefer_fp32: bool = False,
        n_padding=None,
        tensor_layout: str = "HND",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del num_heads_kv
        return ark_cpu_bestla_sdpa_packed_from_descriptor(
            self.descriptor,
            query,
            cache_k,
            cache_v,
            seq_len_kv,
            is_causal=is_causal,
            scale=scale,
            use_alibi=use_alibi,
            use_tanh=use_tanh,
            prefer_fp32=prefer_fp32,
            n_padding=n_padding,
            tensor_layout=tensor_layout,
        )


def ark_cpu_packed_kv_descriptor(
    batch: int,
    num_heads_kv: int,
    capacity: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float16,
):
    """Create an internal/experimental packed-KV descriptor for repeated CPU BestLA cache operations."""
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_packed_kv_descriptor"):
        raise NotImplementedError("ARK CPU packed KV descriptor is not available (requires BestLA CPU extension build)")
    return cpu_lib.ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, cvt_dtype(dtype))


def ark_cpu_packed_kv_alloc_from_descriptor(
    descriptor,
    *,
    dtype: Optional[torch.dtype] = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_packed_kv_elems_desc"):
        raise NotImplementedError("ARK CPU packed KV descriptor allocation is not available (requires BestLA CPU extension build)")
    desc_info = ark_cpu_packed_kv_info(descriptor=descriptor)
    desc_dtype = _torch_dtype_from_ark_dtype(int(desc_info["dtype"]))
    alloc_dtype = dtype if dtype is not None else desc_dtype
    if alloc_dtype != desc_dtype:
        raise ValueError(f"Descriptor dtype {desc_dtype} does not match requested allocation dtype {alloc_dtype}")
    k_elems, v_elems = cpu_lib.ark_cpu_packed_kv_elems_desc(descriptor)
    return (
        torch.zeros(k_elems, dtype=alloc_dtype, device=device),
        torch.zeros(v_elems, dtype=alloc_dtype, device=device),
    )


def ark_cpu_packed_kv_alloc(
    batch: int,
    num_heads_kv: int,
    capacity: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> tuple:
    """Allocate internal/experimental 1-D packed K/V tensors for the BestLA decode path.

    Returns (cache_k, cache_v) as 1-D tensors of the requested dtype.  The packed
    geometry is NTILE24_ROWPACK1 for fp16, NTILE48_ROWPACK2 for bf16, matching the
    layout expected by ark_cpu_update_packed_k/v and ark_cpu_bestla_sdpa_packed.

    Both tensors are zero-initialized (unwritten packed slots read as zero).
    """
    descriptor = ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, dtype=dtype)
    return ark_cpu_packed_kv_alloc_from_descriptor(descriptor, dtype=dtype, device=device)


def ark_cpu_packed_kv_info(
    batch: Optional[int] = None,
    num_heads_kv: Optional[int] = None,
    capacity: Optional[int] = None,
    head_dim: Optional[int] = None,
    *,
    dtype: torch.dtype = torch.float16,
    descriptor=None,
) -> dict:
    """Return the internal/experimental packed-KV descriptor used by the CPU BestLA path."""
    if descriptor is not None:
        if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_packed_kv_info_desc"):
            raise NotImplementedError("ARK CPU packed KV descriptor query is not available (requires BestLA CPU extension build)")
        return dict(cpu_lib.ark_cpu_packed_kv_info_desc(descriptor))
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_packed_kv_info"):
        raise NotImplementedError("ARK CPU packed KV info query is not available (requires BestLA CPU extension build)")
    if batch is None or num_heads_kv is None or capacity is None or head_dim is None:
        raise ValueError("batch, num_heads_kv, capacity, and head_dim are required when descriptor is not provided")
    return dict(cpu_lib.ark_cpu_packed_kv_info(batch, num_heads_kv, capacity, head_dim, cvt_dtype(dtype)))


def ark_cpu_update_packed_kv_from_descriptor(
    descriptor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    start_pos: int,
    *,
    tensor_layout: str = "HND",
    no_zeroing: bool = False,
) -> None:
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_update_packed_k_desc"):
        raise NotImplementedError("ARK CPU packed KV descriptor update is not available (requires BestLA CPU extension build)")
    batch, num_heads_kv, append_len, head_dim = _attention_shape(key, tensor_layout)
    if batch != int(descriptor.batch_size) or num_heads_kv != int(descriptor.heads_kv) or head_dim != int(descriptor.head_dim):
        raise ValueError("K descriptor shape does not match the key/value tensors")
    if start_pos < 0 or start_pos + append_len > int(descriptor.logical_capacity):
        raise ValueError("KV append range exceeds packed descriptor capacity")
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    cpu_lib.ark_cpu_update_packed_k_desc(
        cache_k.data_ptr(), key.data_ptr(), *k_strides, descriptor, append_len, int(start_pos), bool(no_zeroing)
    )
    cpu_lib.ark_cpu_update_packed_v_desc(
        cache_v.data_ptr(), value.data_ptr(), *v_strides, descriptor, append_len, int(start_pos), bool(no_zeroing)
    )


def ark_cpu_update_packed_kv(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    start_pos: int,
    capacity: int,
    *,
    tensor_layout: str = "HND",
    no_zeroing: bool = False,
) -> None:
    """Append raw K/V tokens at [start_pos, start_pos+append_len) into packed caches.

    cache_k and cache_v must have been allocated by ark_cpu_packed_kv_alloc with
    the same (batch, num_heads_kv, capacity, head_dim, dtype).  key and value are
    raw HND/NHD tensors; tensor_layout selects the stride convention.  capacity
    must match the value passed to ark_cpu_packed_kv_alloc.  The update is in-place.
    """
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_update_packed_k"):
        raise NotImplementedError("ARK CPU packed KV update is not available (requires BestLA CPU extension build)")
    kv_dtype = cvt_dtype(key.dtype)
    batch, num_heads_kv, append_len, head_dim = _attention_shape(key, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    cpu_lib.ark_cpu_update_packed_k(
        cache_k.data_ptr(), key.data_ptr(),
        *k_strides,
        kv_dtype, batch, num_heads_kv, append_len, head_dim, capacity, int(start_pos), bool(no_zeroing),
    )
    cpu_lib.ark_cpu_update_packed_v(
        cache_v.data_ptr(), value.data_ptr(),
        *v_strides,
        kv_dtype, batch, num_heads_kv, append_len, head_dim, capacity, int(start_pos), bool(no_zeroing),
    )


def ark_cpu_copy_packed_kv(
    dst_cache_k: torch.Tensor,
    dst_cache_v: torch.Tensor,
    src_cache_k: torch.Tensor,
    src_cache_v: torch.Tensor,
    seq_off: int,
    seq_size: int,
    *,
    batch: int,
    num_heads_kv: int,
    capacity: int,
    head_dim: int,
    dtype: torch.dtype,
    no_zeroing: bool = False,
) -> None:
    """Copy a logical window from one packed KV cache to another."""
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_copy_packed_k"):
        raise NotImplementedError("ARK CPU packed KV copy is not available (requires BestLA CPU extension build)")
    kv_dtype = cvt_dtype(dtype)
    cpu_lib.ark_cpu_copy_packed_k(
        dst_cache_k.data_ptr(),
        src_cache_k.data_ptr(),
        kv_dtype,
        batch,
        num_heads_kv,
        capacity,
        head_dim,
        int(seq_off),
        int(seq_size),
        bool(no_zeroing),
    )
    cpu_lib.ark_cpu_copy_packed_v(
        dst_cache_v.data_ptr(),
        src_cache_v.data_ptr(),
        kv_dtype,
        batch,
        num_heads_kv,
        capacity,
        head_dim,
        int(seq_off),
        int(seq_size),
        bool(no_zeroing),
    )


def ark_cpu_copy_packed_kv_from_descriptor(
    descriptor,
    dst_cache_k: torch.Tensor,
    dst_cache_v: torch.Tensor,
    src_cache_k: torch.Tensor,
    src_cache_v: torch.Tensor,
    seq_off: int,
    seq_size: int,
    *,
    no_zeroing: bool = False,
) -> None:
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_copy_packed_k_desc"):
        raise NotImplementedError("ARK CPU packed KV descriptor copy is not available (requires BestLA CPU extension build)")
    cpu_lib.ark_cpu_copy_packed_k_desc(
        dst_cache_k.data_ptr(), src_cache_k.data_ptr(), descriptor, int(seq_off), int(seq_size), bool(no_zeroing)
    )
    cpu_lib.ark_cpu_copy_packed_v_desc(
        dst_cache_v.data_ptr(), src_cache_v.data_ptr(), descriptor, int(seq_off), int(seq_size), bool(no_zeroing)
    )


def ark_cpu_shift_packed_k(
    cache_k: torch.Tensor,
    cossin: torch.Tensor,
    *,
    batch: int,
    num_heads_kv: int,
    capacity: int,
    head_dim: int,
    dtype: torch.dtype,
    seq_keep: int,
) -> None:
    """Apply packed-K shift-RoPE in-place on the BF16 packed cache path."""
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_shift_packed_k"):
        raise NotImplementedError("ARK CPU packed K shift-RoPE is not available (requires BestLA CPU extension build)")
    if cossin.dtype != torch.float16:
        raise ValueError(f"cossin must be float16, got {cossin.dtype}")
    cpu_lib.ark_cpu_shift_packed_k(
        cache_k.data_ptr(),
        cossin.data_ptr(),
        cvt_dtype(dtype),
        batch,
        num_heads_kv,
        capacity,
        head_dim,
        int(seq_keep),
    )


def ark_cpu_shift_packed_k_from_descriptor(
    descriptor,
    cache_k: torch.Tensor,
    cossin: torch.Tensor,
    *,
    seq_keep: int,
) -> None:
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_shift_packed_k_desc"):
        raise NotImplementedError("ARK CPU packed K descriptor shift-RoPE is not available (requires BestLA CPU extension build)")
    if cossin.dtype != torch.float16:
        raise ValueError(f"cossin must be float16, got {cossin.dtype}")
    cpu_lib.ark_cpu_shift_packed_k_desc(cache_k.data_ptr(), cossin.data_ptr(), descriptor, int(seq_keep))


def ark_cpu_bestla_sdpa_packed(
    query: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    seq_len_kv: int,
    capacity: int,
    num_heads_kv: int,
    *,
    is_causal: bool = False,
    scale: Optional[float] = None,
    use_alibi: bool = False,
    use_tanh: bool = False,
    prefer_fp32: bool = False,
    n_padding=None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Internal/experimental BestLA mixed-precision SDPA over a packed K/V cache.

    query must be float32; cache_k/cache_v must be float16 or bfloat16 (produced
    by ark_cpu_packed_kv_alloc + ark_cpu_update_packed_kv).  seq_len_kv is the
    current valid sequence length in the cache (<= capacity).  capacity and
    num_heads_kv must match the values used at allocation time.

    This helper is outside the standard public sdpa() contract and exists for the
    internal mixed-route / packed-cache feature surface. n_padding accepts either
    one scalar applied to every batch entry or a length-B vector.
    """
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_bestla_sdpa_packed"):
        raise NotImplementedError("ARK CPU packed BestLA SDPA is not available (requires BestLA CPU extension build)")

    kv_dtype = cvt_dtype(cache_k.dtype)
    batch, num_heads_q, seq_len_q, head_dim = _attention_shape(query, tensor_layout)
    normalized_n_padding = _normalize_batch_padding(n_padding, batch)
    sm_scale = scale if scale is not None else (head_dim ** -0.5)
    output = _empty_attention_output(batch, num_heads_q, seq_len_q, head_dim,
                                     dtype=query.dtype, device=query.device, tensor_layout=tensor_layout)
    q_strides = _attention_strides_qko(query, tensor_layout)
    o_strides = _attention_strides_qko(output, tensor_layout)
    cpu_lib.ark_cpu_bestla_sdpa_packed(
        query.data_ptr(), cache_k.data_ptr(), cache_v.data_ptr(), output.data_ptr(),
        *q_strides, *o_strides,
        cvt_dtype(query.dtype), kv_dtype,
        batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, capacity, head_dim,
        float(sm_scale), is_causal, use_alibi, use_tanh, prefer_fp32, normalized_n_padding,
    )
    return output


def ark_cpu_bestla_sdpa_packed_from_descriptor(
    descriptor,
    query: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    seq_len_kv: int,
    *,
    is_causal: bool = False,
    scale: Optional[float] = None,
    use_alibi: bool = False,
    use_tanh: bool = False,
    prefer_fp32: bool = False,
    n_padding=None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Descriptor-based internal/experimental packed BestLA SDPA forward."""
    if cpu_lib is None or not hasattr(cpu_lib, "ark_cpu_bestla_sdpa_packed_desc"):
        raise NotImplementedError(
            "ARK CPU packed BestLA SDPA descriptor path is not available (requires BestLA CPU extension build)"
        )
    batch, num_heads_q, seq_len_q, head_dim = _attention_shape(query, tensor_layout)
    if batch != int(descriptor.batch_size) or head_dim != int(descriptor.head_dim):
        raise ValueError("Query shape does not match the packed KV descriptor")
    normalized_n_padding = _normalize_batch_padding(n_padding, batch)
    sm_scale = scale if scale is not None else (head_dim ** -0.5)
    output = _empty_attention_output(
        batch, num_heads_q, seq_len_q, head_dim, dtype=query.dtype, device=query.device, tensor_layout=tensor_layout
    )
    q_strides = _attention_strides_qko(query, tensor_layout)
    o_strides = _attention_strides_qko(output, tensor_layout)
    cpu_lib.ark_cpu_bestla_sdpa_packed_desc(
        query.data_ptr(),
        cache_k.data_ptr(),
        cache_v.data_ptr(),
        output.data_ptr(),
        *q_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        descriptor,
        num_heads_q,
        seq_len_q,
        seq_len_kv,
        float(sm_scale),
        bool(is_causal),
        bool(use_alibi),
        bool(use_tanh),
        bool(prefer_fp32),
        normalized_n_padding,
    )
    return output


def ark_xpu_kv_cache_alloc(
    batch: int,
    num_heads_kv: int,
    capacity: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "xpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a contiguous XPU KV cache in internal HND layout: [B, Hkv, capacity, D]."""
    device = torch.device(device)
    if device.type != "xpu":
        raise ValueError("ark_xpu_kv_cache_alloc only supports XPU tensors")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported XPU KV cache dtype: {dtype}")
    if batch <= 0 or num_heads_kv <= 0 or capacity <= 0 or head_dim <= 0:
        raise ValueError("batch, num_heads_kv, capacity, and head_dim must be greater than 0")
    shape = (batch, num_heads_kv, capacity, head_dim)
    return torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype)


def ark_xpu_kv_update(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    start_pos: int,
    *,
    tensor_layout: str = "HND",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append raw HND/NHD K/V tensors into a persistent contiguous XPU KV cache."""
    meta = _XPUKVCacheMeta.from_tensors(key_cache, value_cache)
    if key.device != meta.device or value.device != meta.device:
        raise ValueError("K/V source tensors must be on the same XPU device as the cache")
    if key.dtype != meta.dtype or value.dtype != meta.dtype:
        raise ValueError("K/V cache and source tensors must have the same dtype")
    if xpu_lib is None or not hasattr(xpu_lib, "ark_xpu_kv_update"):
        raise NotImplementedError("ARK XPU KV cache update kernel is not available")

    Bk, Hkv, append_len, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=meta.dtype)
    Bv, Hkv2, append_len_v, Dv = _validate_attention_tensor(value, "V", tensor_layout, expected_dtype=meta.dtype)
    if (Bk, Bv) != (meta.batch, meta.batch) or Hkv != meta.num_heads_kv or Hkv2 != meta.num_heads_kv:
        raise ValueError("K/V source batch or head count does not match cache")
    if append_len_v != append_len or Dk != meta.head_dim or Dv != meta.head_dim:
        raise ValueError("K/V source shape does not match cache")
    if start_pos < 0 or start_pos + append_len > meta.capacity:
        raise ValueError("KV append range exceeds cache capacity")

    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    xpu_lib.ark_xpu_kv_update(
        get_stream(key),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        *k_strides,
        *v_strides,
        cvt_dtype(meta.dtype),
        meta.batch,
        meta.num_heads_kv,
        append_len,
        meta.head_dim,
        meta.capacity,
        int(start_pos),
    )
    return key_cache, value_cache


def sdpa_with_kv_cache(
    query: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    seq_len_kv: int,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Decode-style attention over a persistent contiguous XPU KV cache."""
    if query.device.type != "xpu":
        raise NotImplementedError("sdpa_with_kv_cache is only supported on XPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    meta = _XPUKVCacheMeta.from_tensors(cache_k, cache_v)
    if meta.device != query.device:
        raise ValueError("query and KV cache must be on the same XPU device")
    if meta.dtype != query.dtype:
        raise ValueError(f"query dtype must match KV cache dtype, got Q={query.dtype}, cache={meta.dtype}")
    if seq_len_kv <= 0 or seq_len_kv > meta.capacity:
        raise ValueError(f"seq_len_kv must be in [1, {meta.capacity}], got {seq_len_kv}")
    if xpu_lib is None or not hasattr(xpu_lib, "sdpa_with_kv_cache"):
        raise NotImplementedError("ARK XPU KV-cache decode kernel is not available")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout, expected_dtype=query.dtype)
    if B != meta.batch or D != meta.head_dim:
        raise ValueError("query batch/head_dim must match the KV cache")
    _validate_head_ratio(Hq, meta.num_heads_kv)
    if D not in (64, 128, 96, 192):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128, 96, 192")
    if is_causal and Sq != 1:
        raise NotImplementedError(
            "sdpa_with_kv_cache only supports is_causal=True for single-token decode (seq_len_q == 1)"
        )
    _validate_no_dropout(dropout_p, "sdpa_with_kv_cache")
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=seq_len_kv, device=query.device)

    output = _empty_attention_output(B, Hq, Sq, D, dtype=query.dtype, device=query.device, tensor_layout=tensor_layout)
    q_strides = _attention_strides_qko(query, tensor_layout)
    o_strides = _attention_strides_qko(output, tensor_layout)
    k_strides = _contiguous_hnd_qko_strides(meta.num_heads_kv, seq_len_kv, meta.head_dim)
    v_strides = _contiguous_hnd_v_strides(meta.num_heads_kv, seq_len_kv, meta.head_dim)
    xpu_lib.sdpa_with_kv_cache(
        get_stream(query),
        query.data_ptr(),
        cache_k.data_ptr(),
        cache_v.data_ptr(),
        output.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        B,
        Hq,
        meta.num_heads_kv,
        Sq,
        seq_len_kv,
        meta.capacity,
        meta.head_dim,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return output


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    kernel: str = "v1_pvhalf",
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SAGE attention dispatcher.

    Signature mirrors ``sageattention.sageattn``.

    Args:
    - q, k, v: Query/Key/Value tensors. Layout selected by ``tensor_layout``.
    - tensor_layout: "HND" or "NHD".
    - is_causal: Whether to apply causal mask.
    - sm_scale: Softmax scale. Uses ``1 / sqrt(head_dim)`` when None.
    - return_lse: If True, returns (O, LSE) tuple.
    - kernel: Which SAGE variant to dispatch to.
        - "v1_pvhalf" (default): PV in half precision (calls ``sagev1``).
        - "v1_pvi8": PV in INT8 precision (calls ``sagev1_pvi8``).
    - kwargs: Forwarded to the underlying kernel (e.g. ``attn_mask``,
      ``dropout_p``, ``enable_gqa``, ``quant_block_size``).

    Returns:
    - O: same layout as the input tensors.
    - (O, LSE): if return_lse is True.
    """

    if kernel == "v1_pvhalf":
        impl = sagev1
    elif kernel == "v1_pvi8":
        impl = sagev1_pvi8
    else:
        raise ValueError(f"Unsupported sageattn kernel={kernel!r}; supported: 'v1_pvhalf', 'v1_pvi8'")

    return impl(
        query=q,
        key=k,
        value=v,
        is_causal=is_causal,
        scale=sm_scale,
        tensor_layout=tensor_layout,
        return_lse=return_lse,
        **kwargs,
    )


def sage_dynquant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """SAGE Attention with dynamic INT8 block-wise quantization of Q/K.

    Takes FP16 Q, K, V inputs. Quantizes Q and K to INT8 per-block
    using a fused SYCL kernel, then calls SAGE V1 with INT8 data.
    API is like SDPA but with an extra quant_block_size parameter.

    Args:
        query: [B, Hq, Sq, D] float16
        key:   [B, Hkv, Skv, D] float16
        value: [B, Hkv, Skv, D] float16
        quant_block_size: Number of tokens sharing one INT8 scale.
            E.g. 64 means 64 consecutive tokens share one absmax.
            0 means per-token (block_size=1).

    Returns:
        O: [B, Hq, Sq, D] float16
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sage_dynquant is only supported on XPU")

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Hkv, Sq, Skv, D = _validate_attention_geometry(
        query, key, value, tensor_layout, key_dtype=query.dtype, value_dtype=query.dtype
    )
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")
    _validate_no_dropout(dropout_p, "sage_dynquant")
    _validate_attention_mask(attn_mask, batch=B, seq_len_q=Sq, seq_len_kv=Skv, device=query.device)

    # block_size=0 means per-token
    block_size = quant_block_size if quant_block_size > 0 else 1

    # SAGE V1 kernel uses K-tile size=32; quant_block_size must be 1 or >=32
    if block_size != 1 and block_size < 32:
        raise ValueError(
            f"quant_block_size={block_size} is not supported. "
            f"Must be 1 (per-token) or >= 32 (e.g. 32, 64, 128, 256)."
        )

    lib = get_lib(query)
    stream = get_stream(query)
    q_blocks = (Sq + block_size - 1) // block_size
    kv_blocks = (Skv + block_size - 1) // block_size
    q_i8 = torch.empty_like(query, dtype=torch.int8)
    q_scale = torch.empty((B, Hq, q_blocks, 1), dtype=torch.float32, device=query.device)
    q_strides = _attention_strides_qko(query, tensor_layout)
    lib.sage_dynamic_quant_layout(
        stream,
        query.data_ptr(),
        0,
        q_i8.data_ptr(),
        q_scale.data_ptr(),
        B,
        Hq,
        Sq,
        D,
        block_size,
        *q_strides,
    )

    k_i8 = torch.empty_like(key, dtype=torch.int8)
    k_scale = torch.empty((B, Hkv, kv_blocks, 1), dtype=torch.float32, device=key.device)
    k_strides = _attention_strides_qko(key, tensor_layout)
    lib.sage_dynamic_quant_layout(
        stream,
        key.data_ptr(),
        0,
        k_i8.data_ptr(),
        k_scale.data_ptr(),
        B,
        Hkv,
        Skv,
        D,
        block_size,
        *k_strides,
    )

    # Call SAGE v1 with matching quant_block_size
    return sage(
        q_i8,
        k_i8,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
        tensor_layout=tensor_layout,
    )


def moe_gemm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MOE GEMM (Mixture of Experts Grouped GEMM).

    Computes grouped GEMM for MOE layers where different experts process
    different numbers of tokens.

    Expects contiguous layouts:
    - activations: [total_tokens, K] (BF16/FP16)
    - weights: [num_experts, K, N] (BF16/FP16, Row major)
    - num_tokens_per_expert: [num_experts] (int32)
    - scales (optional): [num_experts, N] or None

    Returns:
    - outputs: [total_tokens, N] (same dtype as activations)
    """
    if activations.device.type != "xpu":
        raise NotImplementedError("moe_gemm is only supported on XPU")

    if activations.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"activations must be fp16/bf16, got {activations.dtype}")
    if weights.dtype != activations.dtype:
        raise ValueError("weights dtype must match activations dtype")

    if activations.ndim != 2 or weights.ndim != 3:
        raise ValueError("activations must be 2D [total_tokens, K], weights must be 3D [num_experts, K, N]")

    if not activations.is_contiguous() or not weights.is_contiguous():
        raise ValueError("activations and weights must be contiguous")

    if num_tokens_per_expert.dtype != torch.int32:
        num_tokens_per_expert = num_tokens_per_expert.to(torch.int32)

    if not num_tokens_per_expert.is_contiguous():
        num_tokens_per_expert = num_tokens_per_expert.contiguous()

    total_tokens, K = activations.shape
    num_experts, K_w, N = weights.shape  # weights are [num_experts, K, N]

    if K != K_w:
        raise ValueError(f"K dimension mismatch: activations K={K}, weights K={K_w}")

    if num_tokens_per_expert.shape[0] != num_experts:
        raise ValueError(f"num_tokens_per_expert length {num_tokens_per_expert.shape[0]} != num_experts {num_experts}")

    # Validate total tokens
    expected_total = int(num_tokens_per_expert.sum().item())
    if expected_total != total_tokens:
        raise ValueError(f"Sum of num_tokens_per_expert ({expected_total}) != total_tokens ({total_tokens})")

    lib = get_lib(activations)
    stream = get_stream(activations)
    outputs = torch.empty((total_tokens, N), device=activations.device, dtype=activations.dtype)

    scales_ptr = scales.data_ptr() if scales is not None else 0

    lib.moe_gemm(
        stream,
        activations.data_ptr(),
        weights.data_ptr(),
        scales_ptr,
        outputs.data_ptr(),
        cvt_dtype(activations.dtype),
        N,
        K,
        num_tokens_per_expert.data_ptr(),
        num_experts,
    )
    return outputs


def patch_torch_sdpa(*args, **kwargs):
    from .torch_sdpa_patch import patch_torch_sdpa_with_ark

    return patch_torch_sdpa_with_ark(*args, **kwargs)


def unpatch_torch_sdpa():
    from .torch_sdpa_patch import unpatch_torch_sdpa_with_ark

    return unpatch_torch_sdpa_with_ark()


class _ArkInternalCpuNamespace:
    """Internal/experimental CPU helpers and backend lifecycle tools."""

    debug_resolve_sdpa_route = staticmethod(debug_cpu_sdpa_route)
    kv_cache_alloc = staticmethod(ark_cpu_kv_cache_alloc)
    kv_update = staticmethod(ark_cpu_kv_update)
    packed_kv_descriptor = staticmethod(ark_cpu_packed_kv_descriptor)
    packed_kv_alloc_from_descriptor = staticmethod(ark_cpu_packed_kv_alloc_from_descriptor)
    packed_kv_alloc = staticmethod(ark_cpu_packed_kv_alloc)
    packed_kv_info = staticmethod(ark_cpu_packed_kv_info)
    update_packed_kv_from_descriptor = staticmethod(ark_cpu_update_packed_kv_from_descriptor)
    update_packed_kv = staticmethod(ark_cpu_update_packed_kv)
    copy_packed_kv = staticmethod(ark_cpu_copy_packed_kv)
    copy_packed_kv_from_descriptor = staticmethod(ark_cpu_copy_packed_kv_from_descriptor)
    shift_packed_k = staticmethod(ark_cpu_shift_packed_k)
    shift_packed_k_from_descriptor = staticmethod(ark_cpu_shift_packed_k_from_descriptor)
    bestla_sdpa_packed = staticmethod(ark_cpu_bestla_sdpa_packed)
    bestla_sdpa_packed_from_descriptor = staticmethod(ark_cpu_bestla_sdpa_packed_from_descriptor)
    PackedKVHandle = ArkCpuPackedKVHandle


class _ArkInternalNamespace:
    """Internal/experimental helper surface."""

    cpu = _ArkInternalCpuNamespace()


internal = _ArkInternalNamespace()


__all__ = ["patch_torch_sdpa", "unpatch_torch_sdpa"]


# -----------------------------------------------------------------------------
# Compatibility layer
#
# Some callers (e.g. auto_round_extension/ark/qlinear.py) historically imported
# this package as a module and expected certain functions to exist at the module
# level (e.g. ark.repack_quantized_weight, ark.woq_linear).
#
# The wrappers below keep backward compatibility for legacy positional call
# styles without changing the compiled extension.
# -----------------------------------------------------------------------------


def check_isa_supported(_isa: str) -> bool:
    # Best-effort: some builds expose ISA checks via native libs; keep safe.
    # Returning False is conservative and avoids misconfiguration.
    return False


def repack_quantized_weight(*args, **kwargs):
    """Repack quantized weights into ARK/BestLA packed format.

    Supports two call styles:

    1) New style (recommended):
       repack_quantized_weight(QB, scaleB, zp, groupsize, compute_type, weight_type, scale_type, asym)

    2) Legacy style used by qlinear.py:
       repack_quantized_weight(QB, scaleB, zp, g_idx, compute_type, weight_type, scale_type, asym, groupsize)
       repack_quantized_weight(QB, scaleB, zp, g_idx, weight_type, compute_type, scale_type, asym, groupsize)
       (g_idx is ignored)
    """

    if kwargs:
        return _repack_quantized_weight_core(**kwargs)

    if len(args) == 8:
        QB, scaleB, zp, groupsize, compute_type, weight_type, scale_type, asym = args
    elif len(args) == 9:
        QB, scaleB, zp, _g_idx, a4, a5, scale_type, asym, groupsize = args
        # Legacy call sites sometimes swap compute_type/weight_type.
        compute_types = {"fp16", "bf16", "fp32", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"}
        if isinstance(a4, str) and a4 in compute_types:
            compute_type, weight_type = a4, a5
        else:
            weight_type, compute_type = a4, a5
    else:
        raise TypeError("repack_quantized_weight() expects 8 or 9 positional arguments; " f"got {len(args)}")

    # Some native paths may still expect a valid zp pointer even when asym=False.
    if (zp is None) or (isinstance(zp, torch.Tensor) and zp.numel() == 0):
        if not bool(asym):
            k = QB.shape[0]
            n = QB.shape[1]
            zp = torch.zeros((k // int(groupsize), n), dtype=torch.int8, device=QB.device)
        else:
            zp = torch.empty(0, dtype=torch.int8, device=QB.device)

    return _repack_quantized_weight_core(
        QB,
        scaleB,
        zp,
        groupsize,
        compute_type,
        weight_type,
        scale_type,
        asym,
    )


def unpack_weight(
    blob: torch.Tensor,
    out_dtype: torch.dtype,
    n,
    k,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    return _unpack_weight_core(blob, out_dtype, n, k, groupsize, compute_type, weight_type, scale_type, asym)


def packed_weight_size(A: torch.Tensor, n, k, groupsize, compute_type, weight_type, scale_type, asym):
    # Keep signature convenient for Python callers; native library needs a stream.
    lib = get_lib(A)
    stream = get_stream(A)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)
    st = cvtstr_dtype(scale_type)
    return lib.packed_weight_size(stream, n, k, groupsize, ct, wt, st, asym)


def woq_linear(
    A: torch.Tensor,
    packed_B: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    compute_type,
    weight_type,
    scale_type,
    asym,
    groupsize=None,
):
    """Linear helper that writes into a preallocated output tensor."""

    if groupsize is None:
        groupsize = A.shape[-1]

    result = woqgemm(
        A,
        packed_B,
        bias,
        out.shape[-1],
        A.shape[-1],
        int(groupsize),
        compute_type,
        weight_type,
        scale_type,
        bool(asym),
    )
    out.copy_(result)
    return out


if __name__ == "__main__":
    print(cpu_lib is None, xpu_lib is None)

    def matmul_test():
        m = n = k = 128
        dt = torch.int8
        device = "cpu"
        has_bias = False
        if dt == torch.int8:
            A = torch.randint(-128, 127, (m, k), dtype=dt, device=device)
            B = torch.randint(-128, 127, (n, k), dtype=dt, device=device)
            C = igemm_s8s8s32(A, B)
            print(C)
        else:
            A = torch.rand(m, k, dtype=dt, device=device) - 0.5
            B = torch.rand(k, n, dtype=dt, device=device) - 0.5
            bias = torch.rand(1, n, dtype=dt, device=device) if has_bias else torch.empty(0)
            C = matmul(A, B, bias)
        ref = torch.matmul(A, B.T)
        if has_bias:
            ref = ref + bias
        dff = abs(C - ref)
        if dt != torch.int8:
            print(dff.max(), dff.mean())
            print(torch.allclose(ref, C, 0.01, 0.1))

    def woq():
        m = n = k = 128
        dt = torch.float32
        device = "cpu"
        A = torch.rand(m, k, dtype=dt, device=device) - 0.5
        bias = torch.rand(1, n, dtype=dt, device=device) + 1000
        B = torch.randint(-128, 127, (n, k), dtype=torch.int8, device=device)
        scaleB = torch.rand(n, 1, dtype=dt, device=device)
        C = woqgemm_s8(A, B, scaleB, bias)
        print(C)
        DB = B.to(dt) * scaleB
        ref = torch.matmul(A, DB.T) + bias
        print(ref)
        dff = abs(C - ref)
        print(dff.max(), dff.mean())

    def pack_unpack():
        m = n = k = 128
        groupsize = 32
        dt = torch.float32
        device = "xpu"
        B = torch.randint(-8, 7, (k, n), dtype=torch.int8, device=device)
        zp = torch.randint(-8, 7, (k // groupsize, n), dtype=torch.int8, device=device)
        scaleB = torch.rand(k // groupsize, n, dtype=dt, device=device) / 100
        blob = repack_quantized_weight(B, scaleB, zp, groupsize, "fp32", "int4", "fp32", False)
        dq = unpack_weight(blob, dt, n, k, groupsize, "fp32", "int4", "fp32", False)
        print(blob, dq)
        scale_re = scaleB.repeat_interleave(repeats=groupsize, dim=0).to(dt)

        DB = B.to(dt) * scale_re
        dff = abs(DB.T - dq)
        print(dff.max(), dff.mean())

    pack_unpack()
