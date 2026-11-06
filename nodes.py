# ComfyUI-Qwen-TTS Node Implementation
# Based on the open-source Qwen3-TTS project by Alibaba Qwen team

import copy as _copy_module
import os
# Force offline mode for HuggingFace Hub to prevent metadata lookups/downloads
# during model loading.  local_files_only=True alone is not sufficient on
# Windows (transformers #43502 — from_pretrained still hits /api/models/.../revision/main).
os.environ.setdefault("HF_HUB_OFFLINE", "true")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "true")
import sys
import warnings
import torch
import numpy as np
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from contextlib import contextmanager
import folder_paths
import types

# Suppress bitsandbytes MatMul8bitLt bf16->fp16 cast warning (harmless, fires every inference)
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*inputs will be cast.*")

from comfy import model_management
from comfy.utils import ProgressBar

try:
    from qwen_tts.core.models.modeling_qwen3_tts import set_progress_hook
except Exception:  # model layer unavailable; progress degrades to coarse phases
    set_progress_hook = None

# bitsandbytes int8 quantization config (optional hard dependency)
try:
    from transformers import BitsAndBytesConfig
    _BITSANDBYTES_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None  # type: ignore[assignment]
    _BITSANDBYTES_AVAILABLE = False

_DICT_KEYS_TYPE = type({}.keys())


def _deepcopy_dict_keys(obj, memo=None):
    """copy.deepcopy handler for dict_keys view objects.

    Registers into copy._deepcopy_dispatch so that deepcopy(model) — called by
    bitsandbytes LLM.int8() preprocessing (transformers/integrations/bitsandbytes.py
    :get_keys_to_not_convert) — never crashes on dict_keys attributes anywhere in
    the Qwen3-TTS model graph.
    """
    return list(obj)


try:
    _copy_module._deepcopy_dispatch[_DICT_KEYS_TYPE] = _deepcopy_dict_keys
    _DICT_KEYS_DISPATCH_PATCHED = True
except Exception:
    _DICT_KEYS_DISPATCH_PATCHED = False


# Modules that must stay in bf16 (sensitive to the bf16->fp16 cast bitsandbytes
# applies during quantized matmul). Kept in sync with the skip list used to
# build BitsAndBytesConfig in load_qwen_model.
_INT8_SKIP_MODULES = [
    "lm_head",
    "output_proj",
    "codec_encoder",
    "codec_decoder",
    "speaker_encoder",
    "vocoder",
    "seanet",
    "audio_codec",
    "decoder",
    "feature_projection",
]


def _get_parent_module(model, target_name):
    """Return (parent_module, child_name_or_index) for a dotted module path."""
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    child = parts[-1]
    return parent, child


def _convert_linear_to_int8(model, skip_modules):
    """Convert eligible Linear modules to bnb.nn.Linear8bitLt so a pre-quantized
    state_dict (weight.int8 + weight.SCB) can be loaded without re-quantizing."""
    import bitsandbytes as bnb

    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(skip in name for skip in skip_modules):
            continue
        parent, child = _get_parent_module(model, name)
        new_module = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
            has_fp16_weights=False,
        )
        if isinstance(parent, torch.nn.ModuleList):
            parent[int(child)] = new_module
        else:
            setattr(parent, child, new_module)


def _load_model_with_int8(source, **kwargs):
    """Load model with bitsandbytes int8 (dict_keys deepcopy handled via dispatch table)."""
    load_kwargs = dict(kwargs)
    load_kwargs.setdefault("local_files_only", True)
    load_kwargs.setdefault("use_safetensors", True)
    return Qwen3TTSModel.from_pretrained(source, **load_kwargs)


def _load_int8_model(source, int8_source, **kwargs):
    """Load model with bitsandbytes int8.

    When int8_source (a saved *_int8/pytorch_model.bin) is provided, load the
    plain bf16 model WITHOUT quantization_config (skipping the outlier scan +
    int8 conversion),     convert eligible Linear modules to Linear8bitLt, and load
    the pre-quantized state_dict from disk. This is the fast path.

    When int8_source is None (first load), quantize on the fly via
    quantization_config (the slow path).
    """
    if int8_source and _BITSANDBYTES_AVAILABLE:
        # Fast path: load pre-quantized weights from disk, skip re-quantization
        load_kwargs = {k: v for k, v in kwargs.items() if k != "quantization_config"}
        try:
            model = _load_model_with_int8(source, **load_kwargs)
            _convert_linear_to_int8(model.model, _INT8_SKIP_MODULES)
            # bitsandbytes Linear8bitLt must be on CUDA (via .cuda(), not .to())
            # before loading a quantized state_dict
            target_device = kwargs.get("device_map", "cuda")
            if target_device != "cpu":
                if target_device == "cuda":
                    model.model = model.model.cuda()
                else:
                    model.model = model.model.to(target_device)
            state_dict = torch.load(os.path.join(int8_source, "pytorch_model.bin"), map_location="cpu")
            model.model.load_state_dict(state_dict, strict=False)
            print("Loaded pre-quantized int8 weights from disk (skipped re-quantization)")
            return model
        except Exception as e:
            print(f"⚠️ [Qwen3-TTS] Fast int8 load from {int8_source} failed ({e}); falling back to quantize-on-load.")
            model = _load_model_with_int8(source, **kwargs)
            return model
    # Slow path (first time): quantize on the fly with quantization_config
    model = _load_model_with_int8(source, **kwargs)
    return model


# ---------------------------------------------------------------------------
# INT8 W8A8 (+ ConvRot) quantization — first-time setup philosophy
#
# Quantize once inside ComfyUI on first run, save to models/qwen-tts/, load
# fast on subsequent runs. Falls back to bf16 when Triton is unavailable.
# ---------------------------------------------------------------------------

_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None

_CONVROT_GROUP_SIZE = 256
_HADAMARD_CACHE = {}


def _build_hadamard(size, device="cpu", dtype=torch.float32):
    """Normalized regular Hadamard matrix (size x size), size a power of two >= 4."""
    cache_key = (size, str(device), str(dtype))
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of two >= 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[cache_key] = h
    return h


def _rotate_groupwise(x, hadamard, group_size):
    """Online activation rotation: rotate each group_size block by H."""
    orig_shape = x.shape
    features = orig_shape[-1]
    n_groups = features // group_size
    grouped = x.reshape(*orig_shape[:-1], n_groups, group_size)
    h = hadamard.to(dtype=x.dtype, device=x.device)
    return torch.matmul(grouped, h).reshape(orig_shape)


if _TRITON_AVAILABLE:
    @triton.jit
    def _w8a8_gemm_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
                          M, N, K,
                          stride_am, stride_ak,
                          stride_bk, stride_bn,
                          stride_cm, stride_cn,
                          BLOCK_SIZE_M: tl.constexpr = 64,
                          BLOCK_SIZE_N: tl.constexpr = 64,
                          BLOCK_SIZE_K: tl.constexpr = 64):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=a_mask, other=0).to(tl.int8)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0).to(tl.int8)
            acc += tl.dot(a, b)
        a_s = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)[:, None]
        b_s = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)[None, :]
        out = acc.to(tl.float32) * a_s * b_s
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, out.to(tl.bfloat16), mask=c_mask)


class W8A8Linear(torch.nn.Module):
    """INT8 weight (+ per-channel fp32 scale) linear layer with optional ConvRot.

    Weight is stored as int8 [out, in] + fp32 scale [out]. Activations are
    quantized dynamically per token at forward. int8 x int8 -> int32 matmul is
    done with a Triton kernel and dequantized to bf16. Falls back to bf16 when
    Triton is unavailable.
    """

    def __init__(self, in_features, out_features, bias=False, convrot=False, group_size=_CONVROT_GROUP_SIZE):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.convrot = bool(convrot) and (self.in_features % self.group_size == 0)
        self.weight = torch.nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.empty(self.out_features, dtype=torch.float32), requires_grad=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_features, dtype=torch.bfloat16), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        if self.convrot:
            self.register_buffer("hadamard", _build_hadamard(self.group_size, device="cpu", dtype=torch.float32), persistent=False)
        else:
            self.hadamard = None

    def quantize_from_bf16(self, weight_bf16):
        w = weight_bf16.to(torch.float32)
        if self.convrot:
            w = _rotate_groupwise(w, self.hadamard, self.group_size)
        w_abs = w.abs()
        scale = w_abs.max(dim=1, keepdim=True)[0].clamp(min=1e-12) / 127.0
        w_q = (w / scale).round().clamp(-128, 127).to(torch.int8)
        self.weight.data = w_q
        self.weight_scale.data = scale.squeeze(1).to(torch.float32)

    def forward(self, x):
        x_in = x
        if self.convrot:
            x_in = _rotate_groupwise(x_in, self.hadamard, self.group_size)
        x_f = x_in.to(torch.float32)
        x_scale = x_f.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12) / 127.0
        x_q = (x_f / x_scale).round().clamp(-128, 127).to(torch.int8)
        out = self._matmul_dequant(x_q, x_scale)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def _matmul_dequant(self, x_q, x_scale):
        flat = x_q.reshape(-1, self.in_features)
        flat_scale = x_scale.reshape(-1, 1)
        M, K = flat.shape
        N = self.out_features
        out = torch.empty(M, N, dtype=torch.bfloat16, device=flat.device)
        if _TRITON_AVAILABLE and flat.is_cuda and self.weight.is_cuda:
            try:
                _w8a8_gemm_kernel[(M + 63) // 64, (N + 63) // 64](
                    flat, self.weight, flat_scale.reshape(-1), self.weight_scale, out,
                    M, N, K,
                    flat.stride(0), flat.stride(1),
                    self.weight.stride(0), self.weight.stride(1),
                    out.stride(0), out.stride(1),
                )
                return out.reshape(*x_q.shape[:-1], N)
            except Exception:
                pass
        # bf16 fallback: dequantize weight then bf16 matmul
        w_bf16 = (self.weight.to(torch.float32) * self.weight_scale.to(torch.float32)[:, None]).to(torch.bfloat16)
        a_bf16 = (flat.to(torch.float32) * flat_scale.to(torch.float32)).to(torch.bfloat16)
        out = torch.nn.functional.linear(a_bf16, w_bf16, None)
        return out.reshape(*x_q.shape[:-1], N)


def _convert_linear_to_w8a8(model, skip_modules, convrot=True,
                            group_size=_CONVROT_GROUP_SIZE, quantize=True):
    """Convert eligible Linear modules to W8A8Linear shells.

    When quantize=True (first-run path), quantize bf16 weight -> int8 + scale.
    When quantize=False (fast-path load from checkpoint), create empty shells;
    load_state_dict fills weight/weight_scale from the saved checkpoint.

    ConvRot is applied only when in_features % group_size == 0.
    """
    converted = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(skip in name for skip in skip_modules):
            skipped += 1
            continue
        use_convrot = bool(convrot) and (module.in_features % group_size == 0)
        parent, child = _get_parent_module(model, name)
        new_module = W8A8Linear(module.in_features, module.out_features,
                                bias=(module.bias is not None),
                                convrot=use_convrot, group_size=group_size)
        if quantize:
            new_module.quantize_from_bf16(module.weight.detach())
            if module.bias is not None:
                new_module.bias.data = module.bias.detach().to(torch.bfloat16)
        if isinstance(parent, torch.nn.ModuleList):
            parent[int(child)] = new_module
        else:
            setattr(parent, child, new_module)
        converted += 1
    print(f"[Qwen3-TTS] w8a8: replaced {converted} Linear layers "
          f"(skipped {skipped}; quantize={quantize}).")
    return converted


def _quantize_model_w8a8(source, model_name, qwen_root, device_map="cuda", dtype=torch.bfloat16,
                         attn_implementation=None, quantization_config=None, convrot=True):
    """First-run: load bf16 model, convert talker Linear layers to W8A8Linear,
    save *_w8a8.safetensors + _comfy_quant.json sidecar. Returns the model."""
    from safetensors.torch import save_file
    import json

    load_kwargs = {"device_map": device_map, "dtype": dtype, "quantization_config": None}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    print(f"🔧 [Qwen3-TTS] w8a8: loading bf16 model for one-time quantization: {source}")
    model = _load_model_with_int8(source, **load_kwargs)
    _convert_linear_to_w8a8(model.model, _INT8_SKIP_MODULES, convrot=convrot, group_size=_CONVROT_GROUP_SIZE)
    if device_map != "cpu" and torch.cuda.is_available():
        model.model = model.model.cuda()
    os.makedirs(qwen_root, exist_ok=True)
    w8a8_path = os.path.join(qwen_root, model_name + "_w8a8.safetensors")
    sidecar_path = os.path.join(qwen_root, model_name + "_w8a8_comfy_quant.json")
    print(f"💾 [Qwen3-TTS] w8a8: saving quantized checkpoint to {w8a8_path}")
    sd = model.model.state_dict()
    cpu_sd = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in sd.items()}
    save_file(cpu_sd, w8a8_path)
    metadata = {
        "format": "int8_w8a8",
        "convrot": bool(convrot),
        "convrot_groupsize": _CONVROT_GROUP_SIZE,
        "dtype": str(dtype),
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"💾 [Qwen3-TTS] w8a8: saved sidecar to {sidecar_path}")
    return model


def _load_w8a8_model(source, w8a8_path, device_map="cuda", dtype=torch.bfloat16,
                     attn_implementation=None, quantization_config=None):
    """Subsequent-run fast path: load bf16 model, convert Linear -> W8A8Linear,
    move to CUDA, load state_dict from *_w8a8.safetensors. Returns model."""
    from safetensors.torch import load_file
    import json

    load_kwargs = {"device_map": device_map, "dtype": dtype, "quantization_config": None}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = _load_model_with_int8(source, **load_kwargs)
    sidecar_path = w8a8_path[:-len(".safetensors")] + "_comfy_quant.json"
    convrot = True
    group_size = _CONVROT_GROUP_SIZE
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            convrot = bool(meta.get("convrot", True))
            group_size = int(meta.get("convrot_groupsize", _CONVROT_GROUP_SIZE))
        except Exception:
            pass
    _convert_linear_to_w8a8(model.model, _INT8_SKIP_MODULES, convrot=convrot,
                            group_size=group_size, quantize=False)
    if device_map != "cpu" and torch.cuda.is_available():
        model.model = model.model.cuda()
    print(f"🚀 [Qwen3-TTS] w8a8: loading fast-path quantized weights from {w8a8_path}")
    state_dict = load_file(w8a8_path, device="cpu")
    model.model.load_state_dict(state_dict, strict=False)
    return model


# Register "qwen-tts" model folder for extra_model_paths.yaml support
try:
    folder_paths.add_model_folder_path("qwen-tts", os.path.join(folder_paths.models_dir, "qwen-tts"))
except Exception:
    pass


def _comfy_attention_available() -> dict:
    """Mirror ComfyUI's native attention-backend detection for the Qwen-TTS node layer."""
    try:
        import comfy.model_management as _mm
    except Exception:
        return {}
    return {
        "xformers": bool(getattr(_mm, "xformers_enabled", lambda: False)()),
        "flash_attn": bool(getattr(_mm, "flash_attention_enabled", lambda: False)()),
        "sage_attn": bool(getattr(_mm, "sage_attention_enabled", lambda: False)()),
        "comfy_kitchen": bool(getattr(_mm, "comfy_kitchen_attention_enabled", lambda: False)()),
        "sdpa": True,
        "eager": True,
    }


# Common languages list for UI
DEMO_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]

# Language mapping dictionary to engine codes
LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

# Inline emotion tag map: tag -> detailed instruct spoken to the model.
# Mirrors ComfyUI-Qwen3-TTS/nodes.py EMOTION_MAP.
EMOTION_MAP = {
    # English
    "happy": "Speak in a happy and cheerful tone, full of energy.",
    "angry": "Speak with an angry and stern tone, aggressive and sharp.",
    "sad": "Speak in a sad, low-spirited voice with a hint of sorrow.",
    "whisper": "Speak in a very soft whisper, as if sharing a secret.",
    "fearful": "Speak with a trembling, terrified voice, sounding panic-stricken.",
    "surprised": "Speak with a shocked and incredulous tone, pitch raised.",
    "disgusted": "Speak with a tone of revulsion and strong dislike.",
    "neutral": "Speak in a natural, calm, and steady tone.",
    # Chinese
    "开心": "用愉快、开心的语气说话，充满活力和阳光。",
    "激动": "语气非常激动，语速稍快，充满兴奋感。",
    "生气": "用愤怒、严厉的语气说话，语调生硬且带有攻击性。",
    "难过": "语气低沉、忧伤，带有明显的哀伤和哭腔感。",
    "温柔": "声音轻柔、温婉，充满爱意和关怀。",
    "恐惧": "声音颤抖，语气惊恐不安，呼吸感加重。",
    "冷酷": "语气冰冷、没有任何情感波动，显得疏离而机械。",
    "低语": "用极小的声音说话，像是在耳边轻声细语，充满神秘感。",
    "惊讶": "语气充满震惊和不可思议，音调上扬，带有明显的意外感。",
    "厌恶": "语气充满嫌弃和反感，带有不屑和排斥的情感。",
    "平静": "语气平稳自然，没有明显的情绪波动，清晰且沉稳。",
}

# Model family options for UI (0.6B / 1.7B)
MODEL_FAMILIES = ["0.6B", "1.7B"]
# Mapping of family to default HuggingFace repo ID
MODEL_FAMILY_TO_HF = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# All required models for batch download
ALL_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
]

_MODELS_CHECKED = False
_MODEL_CACHE = {}

# Handle qwen_tts package import
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_dir = os.path.join(current_dir, "qwen_tts")

# CRITICAL: Add current_dir to sys.path FIRST so Python can resolve 'qwen_tts' as a package
# This allows qwen_tts internal files to use relative imports like 'from ..core.models import ...'
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also ensure the qwen_tts folder itself is importable
if qwen_tts_dir not in sys.path:
    sys.path.insert(0, qwen_tts_dir)

try:
    # 1. Try absolute import first (if user installed via pip)
    import qwen_tts
    Qwen3TTSModel = qwen_tts.Qwen3TTSModel
    VoiceClonePromptItem = qwen_tts.VoiceClonePromptItem
except ImportError:
    try:
        # 2. Fallback to local package import (relative or absolute via sys.path)
        from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    except ImportError as e:
        import traceback
        print(f"\n❌ [Qwen3-TTS] Critical Import Error: {e}")
        if not os.path.exists(qwen_tts_dir):
            print(f"   Missing directory: {qwen_tts_dir}")
            print("   Please clone the repository with submodules or ensure 'qwen_tts' folder exists.")
        else:
            print("   Traceback for debugging:")
            traceback.print_exc()
            print("\n   Common fix: run 'pip install -r requirements.txt' in your ComfyUI environment.")
        
        Qwen3TTSModel = None
        VoiceClonePromptItem = None


# ---------------------------------------------------------------------------
# Path guards
#
# Widget values are stored in the workflow, so whoever shares a workflow also
# decides these strings. Every filesystem path built from a widget is resolved
# against a fixed base directory here, so a shared workflow cannot make this
# plugin read or write files elsewhere on the host.
# ---------------------------------------------------------------------------

def _resolve_within(base_dir, raw, label="path", must_exist=False):
    """Resolve `raw` against `base_dir` and confirm the result stays inside it.

    realpath() collapses symlinks and "..", then commonpath() confirms the
    resolved path really is under the base directory. Both steps are needed:
    realpath alone does not stop a symlink that points outside the base, and
    commonpath alone does not stop an unresolved "..".

    Returns the resolved absolute path, or raises ValueError when the value
    escapes the base directory (or, with must_exist, does not exist).
    """
    base_real = os.path.realpath(base_dir)
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"Empty {label}.")

    # Reject absolute paths and Windows drive/UNC prefixes up front so the
    # error message names the actual problem instead of a generic escape.
    if os.path.isabs(text) or os.path.splitdrive(text)[0]:
        raise ValueError(
            f"{label} must be a relative path inside {base_real}; "
            f"absolute paths are rejected. Got: {text}"
        )

    candidate = os.path.realpath(os.path.join(base_real, text))

    try:
        inside = os.path.commonpath([candidate, base_real]) == base_real
    except ValueError:
        # Raised on Windows when the two paths are on different drives.
        inside = False

    if not inside:
        raise ValueError(
            f"{label} must stay inside {base_real}. Got: {text}"
        )

    if must_exist and not os.path.exists(candidate):
        raise ValueError(f"{label} not found: {candidate}")

    return candidate


def _voices_dir():
    """The single base directory this plugin writes voices into."""
    return os.path.realpath(os.path.join(folder_paths.models_dir, "qwen-tts", "voices"))


def _model_search_roots():
    """Directories this plugin may load models from: ComfyUI model folders only.

    Every entry comes from the local ComfyUI installation (its models
    directory, or a path registered under the "qwen-tts" / "TTS" keys, which is
    how extra_model_paths.yaml contributes a folder). None of them come from a
    workflow, so a shared workflow cannot add a location to this list.
    """
    roots = []
    try:
        roots.append(os.path.join(folder_paths.models_dir, "qwen-tts"))
    except Exception:
        pass
    try:
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        roots.append(os.path.join(comfy_root, "models", "qwen-tts"))
        roots.append(os.path.join(os.path.dirname(comfy_root), "models", "qwen-tts"))
    except Exception:
        pass
    for key in ("qwen-tts", "TTS"):
        try:
            roots.extend(folder_paths.get_folder_paths(key) or [])
        except Exception:
            pass

    unique = []
    for raw in roots:
        try:
            real = os.path.realpath(raw)
        except Exception:
            continue
        if real and real not in unique:
            unique.append(real)
    return unique


def _names_remote_machine(text):
    """True when the string points at another host (Windows UNC share).

    On Windows any filesystem call on such a path - even just a stat - opens an
    SMB session and hands the remote machine the current user's credentials, so
    the shape has to be refused before the path is touched.
    """
    normalized = str(text or "").strip().replace("/", "\\")
    return normalized.startswith("\\\\")


def _custom_model_roots():
    """Folders a workflow may pick a fine-tuned model from.

    Two fixed bases, both decided by the local ComfyUI installation and never by
    the workflow text: the model directories from _model_search_roots(), plus
    output/qwen3tts_finetune where this plugin's own Train node writes
    checkpoints (so "fine-tune then load the checkpoint" keeps working with a
    relative folder name).
    """
    roots = list(_model_search_roots())
    try:
        roots.append(os.path.realpath(os.path.join(folder_paths.output_directory,
                                                   "qwen3tts_finetune")))
    except Exception:
        pass
    unique = []
    for real in roots:
        if real and real not in unique:
            unique.append(real)
    return unique


def _resolve_custom_model_dir(raw):
    """Resolve the ``custom_model_path`` widget inside ComfyUI's model folders.

    The widget is workflow text, so it is treated as a folder name *relative* to
    the folders returned by _custom_model_roots(): network paths and absolute
    paths are refused before any filesystem call, then _resolve_within()
    re-checks the result with realpath + commonpath. Returns None when the
    widget is empty (normal auto-detect).
    """
    text = str(raw or "").strip()
    if not text:
        return None

    if _names_remote_machine(text):
        raise ValueError(
            "custom_model_path must be a folder name inside ComfyUI's model "
            f"directory, not a network path. Got: {text}"
        )
    if os.path.isabs(text) or os.path.splitdrive(text)[0]:
        raise ValueError(
            "custom_model_path must be a folder name inside ComfyUI's model "
            f"directory (models/qwen-tts), not an absolute path. Got: {text}\n"
            "Put the fine-tuned model under models/qwen-tts (or keep it in the "
            "plugin's output/qwen3tts_finetune folder) and give only the folder "
            "path relative to that base here."
        )

    roots = _custom_model_roots()
    if not roots:
        raise ValueError(
            "No ComfyUI model directory is available for custom_model_path "
            "(expected models/qwen-tts)."
        )

    last_error = None
    for root in roots:
        try:
            return _resolve_within(root, text, "custom_model_path", must_exist=True)
        except ValueError as exc:
            last_error = exc

    raise ValueError(
        "custom_model_path must name an existing folder inside one of ComfyUI's "
        f"model directories or output/qwen3tts_finetune. Looked in: "
        f"{', '.join(roots)}\n{last_error}"
    )


def _safe_filename(name, label="filename"):
    """Reduce a user-supplied file name to a bare name inside the base directory.

    Directory separators, drive letters and ".." are rejected outright rather
    than sanitised, so the result is always a plain file name directly inside
    the base directory.
    """
    text = str(name or "").strip()
    if not text:
        raise ValueError(f"Empty {label}.")
    if os.path.isabs(text) or os.path.splitdrive(text)[0]:
        raise ValueError(f"{label} must be a bare file name, not an absolute path. Got: {text}")
    if "/" in text or "\\" in text:
        raise ValueError(f"{label} must not contain directory separators. Got: {text}")
    # Checked as a path segment, so that "..", "../x" and "a/../b" are all caught.
    if os.pardir in text.replace("\\", "/").split("/"):
        raise ValueError(f"{label} must not contain '..'. Got: {text}")
    if text in (os.curdir, ""):
        raise ValueError(f"{label} must name a file. Got: {text}")
    return text


# The .qvp files hold VoiceClonePromptItem objects, which weights_only=True
# refuses to unpickle unless the class is registered as a safe global.
# Registering it keeps torch.load in safe mode while still letting a user's own
# saved voices load; it does not allow arbitrary code from the file to run.
_SAFE_GLOBALS_REGISTERED = False


def _register_safe_globals():
    """Allow VoiceClonePromptItem to be unpickled by torch.load(weights_only=True)."""
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    if VoiceClonePromptItem is None:
        return
    add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add is None:
        return
    try:
        add([VoiceClonePromptItem])
    except Exception as exc:  # pragma: no cover - depends on the torch build
        print(f"[Qwen3-TTS] Could not register VoiceClonePromptItem as a safe global: {exc}")
        return
    _SAFE_GLOBALS_REGISTERED = True


ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager", "xformers", "comfy_kitchen"]

def check_attention_implementation():
    """Check available attention implementations and return in priority order."""
    available = []

    # Prefer ComfyUI's own backend detection (honours --use-ck-attention / --use-sage-attention / etc.)
    try:
        comfy_avail = _comfy_attention_available()
        priority_map = {
            "sage_attn": "sage_attn",
            "flash_attn": "flash_attn",
            "xformers": "xformers",
            "comfy_kitchen": "comfy_kitchen",
        }
        for name, flag in priority_map.items():
            if comfy_avail.get(name):
                available.append(name)
    except Exception:
        pass

    # Fallback: direct package probes (unchanged behavior)
    try:
        from sageattention import sageattn  # noqa: F401
        if "sage_attn" not in available:
            available.append("sage_attn")
    except ImportError:
        pass
    try:
        import flash_attn  # noqa: F401
        if "flash_attn" not in available:
            available.append("flash_attn")
    except ImportError:
        pass
    try:
        import comfy_kitchen  # noqa: F401
        if comfy_kitchen.int8_attention_is_available() and "comfy_kitchen" not in available:
            available.append("comfy_kitchen")
    except ImportError:
        pass

    available.append("sdpa")
    available.append("eager")
    return available

def get_attention_implementation(selection: str) -> str:
    """Get the actual attention implementation based on selection and availability."""
    available = check_attention_implementation()

    if selection == "auto":
        priority = ["sage_attn", "comfy_kitchen", "flash_attn", "xformers", "sdpa", "eager"]
        for attn in priority:
            if attn in available:
                print(f"[Qwen3-TTS] Auto-selected attention: {attn}")
                return attn
        return "eager"
    else:
        if selection in available:
            print(f"[Qwen3-TTS] Using attention: {selection}")
            return selection
        else:
            print(f"[Qwen3-TTS] Requested attention '{selection}' not available, falling back to sdpa")
            if "sdpa" in available:
                return "sdpa"
            return "eager"


def split_text_by_pauses(text: str, config: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Split text into segments based on punctuation and defined pause durations.
    Returns: List of (segment_text, pause_duration_after)
    """
    if not config:
        return [(text, 0.0)]

    pause_linebreak = config.get("pause_linebreak", 0.5)
    period_pause = config.get("period_pause", 0.4)
    comma_pause = config.get("comma_pause", 0.2)
    question_pause = config.get("question_pause", 0.6)
    hyphen_pause = config.get("hyphen_pause", 0.3)

    # Handle linebreaks: replace \n with break tags before punctuation processing
    if pause_linebreak > 0:
        text = re.sub(r'\n+', f' [break={pause_linebreak}] ', text)

    # Inject break tags for both English and Chinese punctuation
    if period_pause > 0:
        text = re.sub(r'[\.。](?!\d)', rf'\g<0> [break={period_pause}]', text)
    if comma_pause > 0:
        text = re.sub(r'[,，](?!\d)', rf'\g<0> [break={comma_pause}]', text)
    if question_pause > 0:
        text = re.sub(r'[\?？](?!\d)', rf'\g<0> [break={question_pause}]', text)
    if hyphen_pause > 0:
        text = re.sub(r'[-—](?!\d)', rf'\g<0> [break={hyphen_pause}]', text)

    # Process explicit break tags
    pause_pattern = r'\[break=([\d\.]+)\]'
    parts = re.split(pause_pattern, text)
    
    segments = []
    current_segment_text = ""
    
    for i in range(len(parts)):
        chunk = parts[i]
        if i % 2 == 0:
            # This is text
            if chunk.strip():
                if current_segment_text:
                    # We had previous text waiting for a pause, but found text instead.
                    # This shouldn't happen with our regex unless there's no punctuation between them.
                    segments.append((current_segment_text.strip(), 0.0))
                current_segment_text = chunk
        else:
            # This is a pause duration
            try:
                pause_val = float(chunk)
            except ValueError:
                pause_val = 0.0
                
            if current_segment_text:
                segments.append((current_segment_text.strip(), pause_val))
                current_segment_text = ""
            elif segments and pause_val > 0:
                # Add this pause to the previous segment (consecutive pauses)
                prev_txt, prev_pause = segments[-1]
                segments[-1] = (prev_txt, prev_pause + pause_val)

    if current_segment_text.strip():
        segments.append((current_segment_text.strip(), 0.0))
        
    return segments


def parse_text_with_inline_tags(text: str) -> List[Tuple[str, Union[str, float]]]:
    """
    Split text into segments based on inline [pause:X] / [p:X] tags.
    Returns: List of (seg_type, content) where seg_type is "text" or "pause".
    Pause tags are removed from the spoken text and turned into silence by the caller.
    """
    if not text:
        return [("text", "")]

    segments: List[Tuple[str, Union[str, float]]] = []
    pause_pattern = re.compile(r"\[(?:pause|p):(\d+(?:\.\d+)?)(?:ms)?\]", re.IGNORECASE)

    for line in str(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        last_idx = 0
        for match in pause_pattern.finditer(line):
            text_part = line[last_idx:match.start()].strip()
            if text_part:
                segments.append(("text", text_part))
            try:
                duration = float(match.group(1))
                segments.append(("pause", duration))
            except ValueError:
                pass
            last_idx = match.end()
        remaining = line[last_idx:].strip()
        if remaining:
            segments.append(("text", remaining))

    if not segments:
        segments.append(("text", text.strip()))
    return segments


def extract_emotion_instruct(text: str, base_instruct: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Extract inline emotion tags like [happy] or [开心] from text.
    Returns (clean_text, combined_instruct_or_None).
    Tags are removed from the spoken text and converted into a detailed instruct.
    """
    tags = re.findall(r"\[([^\]]+)\]", text)
    clean_text = re.sub(r"\[([^\]]+)\]", "", text).strip()
    emotion_instruct = None
    if tags:
        fragments = []
        for t in tags:
            if t in EMOTION_MAP:
                fragments.append(EMOTION_MAP[t])
            else:
                # Pass through custom descriptive tags as-is; the model
                # understands natural-language instructs directly.
                fragments.append(t)
        emotion_instruct = " ".join(fragments)
    combined = None
    parts = [p for p in [base_instruct, emotion_instruct] if p]
    if parts:
        combined = " ".join(parts)
    return clean_text, combined


def _render_segments(segments: List[Tuple[str, Union[str, float]]], render_one, default_sr: int = 24000) -> Dict[str, Any]:
    """
    Render ("text", content) / ("pause", duration) segments into one ComfyUI audio dict.
    render_one(text: str) -> (np.ndarray waveform, int sr)
    Returns {"waveform": torch.Tensor [1, 1, N], "sample_rate": sr}.
    """
    audio_results = []
    sr = default_sr
    for seg_type, content in segments:
        if seg_type == "pause":
            if content and float(content) > 0:
                audio_results.append(np.zeros(int(float(content) * sr), dtype=np.float32))
        else:
            wav, seg_sr = render_one(content)
            sr = seg_sr
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32)
            wav = np.squeeze(wav)
            while wav.ndim > 1:
                wav = np.mean(wav, axis=0)
            audio_results.append(np.ravel(wav))
    if not audio_results:
        raise RuntimeError("No audio generated from inline tag segments.")
    full = np.concatenate(audio_results)
    waveform = torch.from_numpy(full).float().unsqueeze(0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sr)}


def unload_cached_model(cache_key=None):
    """Unload cached model(s) and clear GPU memory.

    Args:
        cache_key: If provided, only unload that specific model.
                   If None, unload all cached models.
    """
    global _MODEL_CACHE

    models_to_unload = []
    if cache_key and cache_key in _MODEL_CACHE:
        print(f"[Qwen3-TTS] Unloading model: {cache_key}...")
        models_to_unload.append(_MODEL_CACHE.pop(cache_key))
    elif _MODEL_CACHE:
        print(f"[Qwen3-TTS] Unloading {len(_MODEL_CACHE)} cached model(s)...")
        models_to_unload = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()

    # Move models to CPU before releasing references, so CUDA tensors are freed
    for model in models_to_unload:
        try:
            if hasattr(model, "model") and hasattr(model.model, "cpu"):
                model.model = model.model.cpu()
        except Exception:
            pass

    model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import gc
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[Qwen3-TTS] Model cache and GPU memory cleared")




def apply_qwen3_patches(model):
    """Apply stability and compatibility patches to the model instance"""
    if model is None:
        return
    
    # 1. Monkeypatch model._normalize_audio_inputs to return mutable lists to avoid upstream tuple assignment bug
    # and to support (waveform, sr) tuple format from ComfyUI
    orig_normalize = getattr(model, "_normalize_audio_inputs", None)
    
    def _safe_normalize(self, audios):
        # Adapted from upstream but returns list entries as lists [waveform, sr]
        # and correctly handles the (waveform, sr) tuple format
        if isinstance(audios, list):
            items = audios
        elif isinstance(audios, tuple) and len(audios) == 2 and isinstance(audios[0], np.ndarray):
            # This is a single audio in tuple format (waveform, sr)
            items = [audios]
        else:
            items = [audios]

        out = []
        for a in items:
            if a is None:
                continue
            if isinstance(a, str):
                wav, sr = self._load_audio_to_np(a)
                out.append([wav.astype(np.float32), int(sr)])
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, list) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                # If we still can't identify it, it might be the cause of our NoneType or other issues
                print(f"⚠️ [Qwen3-TTS] Unknown audio input type: {type(a)}")
                continue

        # ensure mono
        for i in range(len(out)):
            wav, sr = out[i][0], out[i][1]
            if wav.ndim > 1:
                out[i][0] = np.mean(wav, axis=-1).astype(np.float32)
        return out

    try:
        model._normalize_audio_inputs = types.MethodType(_safe_normalize, model)
    except Exception as e:
        print(f"⚠️ [Qwen3-TTS] Failed to apply audio normalization patch: {e}")


def download_model_if_needed(model_id: str, qwen_root: str) -> str:
    """Download a specific model if not found locally"""
    folder_name = model_id.split("/")[-1]
    target_dir = os.path.join(qwen_root, folder_name)
    
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Model already exists
        return target_dir
    
    print(f"\n📥 [Qwen3-TTS] Downloading {model_id}...")
    try:
        from huggingface_hub import snapshot_download
        # Temporarily allow network for first-time download (HF_HUB_OFFLINE is
        # set globally to prevent metadata lookups during from_pretrained).
        prev_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            snapshot_download(repo_id=model_id, local_dir=target_dir)
        finally:
            if prev_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = prev_offline
        print(f"✅ [Qwen3-TTS] {folder_name} downloaded successfully.\n")
        return target_dir
    except ImportError:
        print("⚠️ [Qwen3-TTS] 'huggingface_hub' not found. Please install it to use auto-download.")
        return None
    except Exception as e:
        print(f"❌ [Qwen3-TTS] Failed to download {model_id}: {e}")
        return None


def check_and_download_tokenizer():
    """Check and download tokenizer (shared by all models)"""
    global _MODELS_CHECKED
    if _MODELS_CHECKED:
        return
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # ComfyUI root search
    comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
    qwen_root = os.path.join(comfy_models_path, "qwen-tts")
    os.makedirs(qwen_root, exist_ok=True)

    # Download tokenizer (required by all models)
    tokenizer_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    download_model_if_needed(tokenizer_id, qwen_root)
    
    _MODELS_CHECKED = True


def load_qwen_model(model_type: str, model_choice: str, device: str, precision: str, quantization: str = "none", attention: str = "auto", unload_after: bool = False, previous_attention: str = None, custom_model_path: Optional[str] = None):
    """Shared model loading logic with caching and local path priority"""
    global _MODEL_CACHE
    
    if previous_attention is not None and previous_attention != attention:
        print(f"🔄 [Qwen3-TTS] Attention changed from '{previous_attention}' to '{attention}', clearing cache...")
        unload_cached_model()
    
    attn_impl = get_attention_implementation(attention)

    # custom_model_path is workflow text: resolve it into ComfyUI's model
    # folders before anything is stat'ed, so a shared workflow cannot point at
    # an arbitrary folder or a remote share.
    custom_dir = _resolve_custom_model_dir(custom_model_path)

    # Check and download tokenizer (shared by all models)
    check_and_download_tokenizer()
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # 针对 Mac 的关键修复
        else:
            device = "cpu"
    
    if quantization == "w8a8":
        # W8A8 activations dequantize to bf16; keep everything in bf16 on CUDA.
        dtype = torch.bfloat16
    elif quantization == "int8":
        # bitsandbytes int8 matmul kernels compute in float16; matching dtype
        # avoids the bf16->fp16 cast warning at its source
        dtype = torch.float16
    elif device == "mps" and precision == "bf16":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    # VoiceDesign restriction
    if model_type == "VoiceDesign" and model_choice == "0.6B":
        raise RuntimeError("❌ VoiceDesign only supports 1.7B models!")
        
    # --- int8 quantization config (bitsandbytes LLM.int8()) ---
    quant_config = None
    if quantization == "int8":
        if device != "cuda":
            print("[Qwen3-TTS] int8 quantization requires CUDA; falling back to unquantized bf16/fp32 on device '%s'." % device)
        elif not _BITSANDBYTES_AVAILABLE:
            print("[Qwen3-TTS] quantization='int8' requested but bitsandbytes/transformers BitsAndBytesConfig is not installed. Falling back to unquantized bf16/fp32.")
        else:
            try:
                # Layers that must stay in bf16 to avoid garbled audio output.
                # Qwen3-TTS has Talker + Code Predictor + Speaker Encoder + Codec/Vocoder;
                # the codec/vocoder/speaker-encoder layers are sensitive to the bf16->fp16 cast
                # that bitsandbytes applies during quantized matmul. Keep them in bf16.
                skip_modules = _INT8_SKIP_MODULES
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=skip_modules,
                )
                print("[Qwen3-TTS] int8 quantization enabled (skip_modules=%s)" % skip_modules)
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS] Failed to build BitsAndBytesConfig: {e}. Falling back to unquantized bf16/fp32.")
                quant_config = None
        
    # Cache key includes attention implementation, the resolved model folder,
    # and the quantization setting so int8 vs none do not collide
    cache_key = (model_type, model_choice, device, precision, quantization, attn_impl, custom_dir)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Clear old cache only when adding a new model with different config
    if _MODEL_CACHE:
        _MODEL_CACHE.clear()
    
    # --- 1. Determine search directories (ComfyUI model folders only) ---
    base_paths = _model_search_roots()

    # --- 2. Search for matching models ---
    HF_MODEL_MAP = {
        ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }
    
    final_source = HF_MODEL_MAP.get((model_type, model_choice)) or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    found_local = None

    if custom_dir:
        print(f"🔧 [Qwen3-TTS] Using custom model folder: {custom_dir}")
        found_local = custom_dir
    
    if not found_local:
        for base in base_paths:
            try:
                if not os.path.isdir(base): continue
                subdirs = os.listdir(base)
                for d in subdirs:
                    cand = os.path.join(base, d)
                    if os.path.isdir(cand):
                        # Match logic: contains model size and type keyword
                        if model_choice in d and model_type.lower() in d.lower():
                            found_local = cand
                            break
                if found_local: break
            except Exception: pass
    
    if found_local:
        final_source = found_local
        print(f"✅ [Qwen3-TTS] Loading local model: {os.path.basename(final_source)}")
    else:
        # Try to download the specific model if not found locally
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
        qwen_root = os.path.join(comfy_models_path, "qwen-tts")
        
        downloaded_path = download_model_if_needed(final_source, qwen_root)
        if downloaded_path:
            final_source = downloaded_path
            print(f"✅ [Qwen3-TTS] Loading downloaded model: {os.path.basename(final_source)}")
        else:
            raise RuntimeError(
                f"❌ [Qwen3-TTS] Model '{final_source}' not found locally and auto-download failed. "
                f"Download it manually into {qwen_root} or check network/HF access."
            )

    # --- int8 pre-quantized folder check (quantize once, load from disk) ---
    int8_source = None
    if quantization == "int8":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
        qwen_root = os.path.join(comfy_models_path, "qwen-tts")
        model_name = os.path.basename(final_source)
        int8_folder = os.path.join(qwen_root, model_name + "_int8")
        if os.path.isdir(int8_folder) and os.path.exists(os.path.join(int8_folder, "pytorch_model.bin")):
            int8_source = int8_folder
            print(f"🔧 [Qwen3-TTS] int8: loading pre-quantized weights from disk: {int8_folder}")
        else:
            print(f"🔧 [Qwen3-TTS] int8: no saved weights found at {int8_folder}, will re-quantize on load")

    if quantization == "int8" and int8_source is None:
        print("int8 first load may take ~2-3 minutes (one-time quantization); subsequent generations are cached.")

    if Qwen3TTSModel is None:
        raise RuntimeError(
            "❌ [Qwen3-TTS] Model class is not loaded because the 'qwen_tts' package failed to import. "
            "Please check the ComfyUI console for the detailed 'Critical Import Error' above."
        )

    # Map attention implementation to model loading parameter
    attn_param = None
    use_sage_attn = False
    use_comfy_kitchen = False

    if attn_impl == "flash_attn":
        attn_param = "flash_attention_2"
    elif attn_impl == "xformers":
        attn_param = "sdpa"   # transformers has no 'xformers' value; fall back to sdpa
    elif attn_impl == "comfy_kitchen":
        use_comfy_kitchen = True
        attn_param = "sdpa"   # base path; comfy_kitchen applied via backend hook below
    elif attn_impl == "sage_attn":
        use_sage_attn = True
    elif attn_impl == "sdpa":
        attn_param = "sdpa"
    elif attn_impl == "eager":
        attn_param = "eager"

    # --- w8a8 quantization (quantize once, load fast) ---
    w8a8_loaded = False
    if quantization == "w8a8":
        if device != "cuda" or not torch.cuda.is_available():
            print("[Qwen3-TTS] w8a8 quantization requires CUDA; falling back to unquantized bf16/fp32.")
            quantization = "none"
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
            qwen_root = os.path.join(comfy_models_path, "qwen-tts")
            model_name = os.path.basename(final_source)
            w8a8_path = os.path.join(qwen_root, model_name + "_w8a8.safetensors")
            if os.path.exists(w8a8_path):
                print(f"🔧 [Qwen3-TTS] w8a8: loading pre-quantized weights from disk: {w8a8_path}")
                model = _load_w8a8_model(final_source, w8a8_path, device_map=device, dtype=dtype,
                                         attn_implementation=attn_param, quantization_config=None)
                w8a8_loaded = True
            else:
                print(f"🔧 [Qwen3-TTS] w8a8: no saved weights found at {w8a8_path}; running first-time "
                      f"quantization (one-time, may take a few minutes)...")
                model = _quantize_model_w8a8(final_source, model_name, qwen_root, device_map=device, dtype=dtype,
                                             attn_implementation=attn_param, quantization_config=None, convrot=True)
                w8a8_loaded = True

    if not w8a8_loaded:
        if use_sage_attn:
            try:
                from sageattention import sageattn
                print(f"🔧 [Qwen3-TTS] Loading model with sage_attn (sageattention)")
                
                model = _load_int8_model(final_source, int8_source, device_map=device, dtype=dtype, quantization_config=quant_config)
                
                # Patch attention modules to use sageattention
                patched_count = 0
                for name, module in model.model.named_modules():
                    if hasattr(module, 'forward') and 'Attention' in type(module).__name__ or 'attn' in name.lower():
                        try:
                            original_forward = module.forward
                            def make_sage_forward(orig_forward, mod):
                                def sage_forward(*args, **kwargs):
                                    # Extract q, k, v from attention call
                                    if len(args) >= 3:
                                        q, k, v = args[0], args[1], args[2]
                                    else:
                                        return orig_forward(*args, **kwargs)
                                    
                                    # Handle attention_mask
                                    attn_mask = kwargs.get('attention_mask', None)
                                    
                                    # Call sageattention
                                    out = sageattn(q, k, v, is_causal=False, attn_mask=attn_mask)
                                    return out
                                return sage_forward
                            
                            module.forward = make_sage_forward(original_forward, module)
                            patched_count += 1
                        except Exception:
                            pass
                
                print(f"🔧 [Qwen3-TTS] Patched {patched_count} attention modules with sage_attn")
                
            except (ImportError, Exception) as e:
                print(f"⚠️ [Qwen3-TTS] Failed with sage_attn, falling back to default attention: {e}")
                model = _load_int8_model(final_source, int8_source, device_map=device, dtype=dtype, quantization_config=quant_config)
        else:
            # comfy_kitchen branch (int8_attention kernel via Comfy Kitchen package)
            if use_comfy_kitchen:
                try:
                    import comfy_kitchen  # noqa: F401
                    if not comfy_kitchen.int8_attention_is_available():
                        raise ImportError("comfy_kitchen int8_attention not available on this device")
                    print(f"🔧 [Qwen3-TTS] Loading model with comfy_kitchen attention (base sdpa; kitchen backend active)")
                    model = _load_int8_model(final_source, int8_source, device_map=device, dtype=dtype, attn_implementation="sdpa", quantization_config=quant_config)
                except (ImportError, Exception) as e:
                    print(f"⚠️ [Qwen3-TTS] comfy_kitchen attention unavailable ({e}), falling back to sdpa")
                    model = _load_int8_model(final_source, int8_source, device_map=device, dtype=dtype, attn_implementation="sdpa", quantization_config=quant_config)
            else:
                try:
                    if attn_param:
                        print(f"🔧 [Qwen3-TTS] Loading model with attention: {attn_impl}")
                        model = _load_int8_model(final_source, int8_source, device_map=device, dtype=dtype, attn_implementation=attn_param, quantization_config=quant_config)
                    else:
                        print(f"🔧 [Qwen3-TTS] Loading model with attention: {attn_impl}")
                        model = _load_int8_model(final_source, int8_source, device_map=device, dtype=dtype, quantization_config=quant_config)
                except (ImportError, ValueError, Exception) as e:
                    print(f"⚠️ [Qwen3-TTS] Failed with {attn_impl}, falling back to default attention: {e}")
                    model = _load_int8_model(final_source, int8_source, device_map=device, dtype=dtype, quantization_config=quant_config)
    
    # Apply patches
    apply_qwen3_patches(model)
    
    _MODEL_CACHE[cache_key] = model

    # --- save int8 model to disk for faster future loads ---
    if quantization == "int8":
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
            qwen_root = os.path.join(comfy_models_path, "qwen-tts")
            model_name = os.path.basename(final_source)
            int8_folder = os.path.join(qwen_root, model_name + "_int8")
            if not os.path.exists(int8_folder):
                print(f"💾 [Qwen3-TTS] Saving int8 model to: {int8_folder}")
                os.makedirs(int8_folder, exist_ok=True)
                # Move tensors to CPU before saving (bitsandbytes compatibility)
                sd = model.model.state_dict()
                cpu_sd = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in sd.items()}
                torch.save(cpu_sd, os.path.join(int8_folder, "pytorch_model.bin"))
                if hasattr(model, "processor") and model.processor is not None:
                    model.processor.save_pretrained(int8_folder)
                # Save config.json unconditionally so *_int8/ is a self-contained checkpoint
                import json
                with open(os.path.join(int8_folder, "config.json"), "w", encoding="utf-8") as f:
                    json.dump(model.model.config.to_dict(), f, indent=2, ensure_ascii=False)
                if hasattr(model.model, "generation_config") and model.model.generation_config is not None:
                    with open(os.path.join(int8_folder, "generation_config.json"), "w", encoding="utf-8") as f:
                        json.dump(model.model.generation_config.to_dict(), f, indent=2, ensure_ascii=False)
                print(f"✅ [Qwen3-TTS] int8 model state saved.")
        except Exception as e:
            import traceback
            print(f"⚠️ [Qwen3-TTS] Failed to save int8 model state: {type(e).__name__}: {e}")
            traceback.print_exc()
    
    if unload_after:
        def unload_callback():
            unload_cached_model()
        model._unload_callback = unload_callback
    else:
        model._unload_callback = None
    
    return model

def _current_node_id():
    """Return the executing ComfyUI node id, or None outside execution."""
    try:
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        if ctx is not None:
            node_id = getattr(ctx, "node_id", None)
            if node_id:
                return node_id
    except Exception:
        pass
    return None


# The active TTSProgress console bar. A module-level one-slot holder because
# _emit_progress is a plain function; set/cleared by register()/unregister()
# so sequential nodes never inherit each other's bar.
_CONSOLE_BAR = [None]


def _emit_progress(value, total):
    """Report progress for the executing node, preferring the official API.

    comfy_api.latest.Execution.set_progress() resolves the node id from the
    execution context and calls ProgressRegistry.update_progress() directly.
    That is the documented path and the one ComfyUI's own code uses for the
    node panel bar, so prefer it.

    comfy.utils.ProgressBar is not used as the transport on purpose: its global
    hook forwards node_id straight through, and ProgressBar's default node_id is
    None (ComfyUI's own code passes none either, e.g.
    comfy_api/latest/_input_impl/video_types.py:772). That writes the update to
    a None-keyed registry entry, which the frontend -- matching progress_state
    entries by node id -- never displays. Measured: the executing node's entry
    stayed {value: 0.0, max: 1.0} for an entire run that way.

    ProgressBar is still worth using for its throttling, so the caller keeps it
    and we only decide where the value goes.
    """
    _CONSOLE_BAR[0].update(value, total) if _CONSOLE_BAR[0] else None
    node_id = _current_node_id()
    if node_id is None:
        return False
    try:
        from comfy_execution.progress import get_progress_state

        get_progress_state().update_progress(
            node_id=node_id, value=value, max_value=total
        )
        return True
    except Exception:
        return False

class _ConsoleBar:
    """A self-contained stderr progress bar.

    ComfyUI 0.37 no longer registers CLIProgressHandler, so nothing draws a
    console bar for custom nodes any more. This restores one without touching
    global state.
    """

    MIN_INTERVAL = 0.1          # seconds between redraws
    WIDTH = 28                  # bar width in characters

    def __init__(self, label="Qwen3-TTS"):
        self.label = label
        self.total = 0
        self.value = 0
        self._last = 0.0
        self._closed = False
        # Decide once, and re-validate in _draw: a console-less launcher (or a
        # redirected stream) must never receive a byte, no matter how the object
        # is poked from the outside.
        self.tty = self._detect_tty()

    @staticmethod
    def _detect_tty():
        try:
            import sys

            return bool(getattr(sys.stderr, "isatty", lambda: False)())
        except Exception:
            return False

    def start(self, total):
        self.total = max(1, int(total or 1))
        self.value = 0
        self._last = 0.0
        self._closed = False
        self._draw(force=True)

    def update(self, value, total=None):
        if self._closed or not self.tty:
            return
        if total is not None:
            self.total = max(1, int(total))
        self.value = value
        self._draw()

    def close(self, message="done"):
        if self._closed or not self.tty:
            self._closed = True
            return
        self._closed = True
        self._draw(force=True, message=message)

    def _draw(self, force=False, message=None):
        if not self.tty or not self._detect_tty():
            self.tty = False
            return
        try:
            now = time.perf_counter()
            if not force and now - self._last < self.MIN_INTERVAL:
                return
            self._last = now
            frac = 0.0 if not self.total else max(0.0, min(1.0, self.value / self.total))
            filled = int(round(frac * self.WIDTH))
            bar = "#" * filled + "-" * (self.WIDTH - filled)
            # `{}` not `{bar}`: str.format() treats a bare name as a *named*
            # placeholder and looks it up in kwargs, so the local variable `bar`
            # was invisible and every call raised KeyError: 'bar'. The except
            # branch then disabled the bar, so nothing was ever written.
            text = "\r{} [{}] {}/{} ({:.0f}%) {}".format(
                self.label, bar, self.value, self.total, frac * 100.0, message or ""
            )
            # Use the module-level `sys`. A function-local `import sys` rebinds
            # the name to the object ComfyUI installed in this module's globals
            # at import time, and that object does not see a redirected
            # sys.stderr: measured 0 bytes written with the local import vs 144
            # bytes with the module-level one, on the same object.
            sys.stderr.write(text + ("\n" if message else ""))
            sys.stderr.flush()
        except Exception:
            # never let a progress bar break generation
            self.tty = False

class TTSProgress:
    """Real per-token progress for one generation call.

    The bar is 100 slots wide on purpose. ComfyUI renders ProgressBar as a
    percentage, and a coarse bar (the original 3 slots) cannot express
    per-token progress at all: a 30-step generation spread over 3 slots yields
    1 -> 2 -> 3, which looks exactly like the frozen bar this replaces. At 100
    slots the bar tracks the sample count, and ProgressBar's own throttling
    (min interval + min percent change) keeps websocket traffic low.

    The hook is scoped to a single generate() call via register()/unregister(),
    so sequential or concurrent nodes never cross-talk. Everything is guarded:
    progress reporting must never be able to break generation.
    """

    # Measured on Qwen3-TTS-12Hz-1.7B: 12 chars -> 21 steps, 19 -> 52, 40 -> 84.
    STEPS_PER_CHAR = 2.1
    MIN_BUDGET = 24

    @classmethod
    def estimate_budget(cls, text, max_new_tokens):
        """Realistic step budget for `text`, clamped by the configured ceiling.

        max_new_tokens is a hard cap (2048 by default) that generation never
        approaches, because the talker stops on codec_eos_token_id. Using it as
        the progress denominator keeps the ratio near zero for the whole run.
        """
        try:
            chars = len(text or "")
        except Exception:
            chars = 0
        ceiling = max(1, int(max_new_tokens or 1))
        return max(1, min(int(chars * cls.STEPS_PER_CHAR) + cls.MIN_BUDGET, ceiling))

    def __init__(self, max_new_tokens, phase_total=100, chunk_total=None,
                 text=None):
        if text is None:
            self.max_new_tokens = max(1, int(max_new_tokens or 1))
        else:
            self.max_new_tokens = self.estimate_budget(text, max_new_tokens)
        # batched generation: the bar is scaled so every chunk owns a slice
        self.chunk_total = int(chunk_total) if chunk_total else None
        self.phase_total = int(phase_total)
        self.pbar = ProgressBar(phase_total, node_id=_current_node_id())
        self.cbar = _ConsoleBar()
        self._active = False
        self._base = 0
        self._span = self.phase_total

    def register(self, span=None):
        """Start observing a generate() call.

        `span` is how many bar slots this call may fill. Single-shot nodes leave
        it None and hand the entire bar to the upcoming call; batched nodes pass
        the per-chunk slice width.
        """
        if span is not None:
            self._span = max(1, int(span))
            self._base = int(getattr(self.pbar, "current", 0) or 0)
        else:
            self._span = self.phase_total
            self._base = 0
        _CONSOLE_BAR[0] = self.cbar
        self.cbar.start(self.phase_total)
        if set_progress_hook is not None and not self._active:
            try:
                set_progress_hook(self._on_step)
                self._active = True
            except Exception:
                self._active = False
        return self

    def unregister(self):
        if self._active and set_progress_hook is not None:
            try:
                set_progress_hook(None)
            except Exception:
                pass
        self._active = False
        if _CONSOLE_BAR[0] is self.cbar:
            _CONSOLE_BAR[0] = None

    def finish_generation(self):
        """The current generate() call ended; release its slice of the bar."""
        self.cbar.start(self.phase_total)
        self._active = False
        if set_progress_hook is not None:
            try:
                set_progress_hook(None)
            except Exception:
                pass
        try:
            self._base = int(getattr(self.pbar, "current", 0) or 0)
        except Exception:
            self._base = 0

    def _on_step(self, step, total):
        try:
            # The node's max_new_tokens is the ceiling the user configured and
            # therefore the honest denominator. The hook's `total` is the model's
            # own default (8192), which is unrelated to the node setting and
            # would keep the ratio pinned near zero for the whole run.
            budget = self.max_new_tokens
            if budget <= 1 and total:
                budget = max(1, int(total))
            budget = max(1, int(budget))
            value = self._base + int(round(self._span * min(int(step), budget) / budget))
            value = max(1, min(value, self.phase_total))
            if not _emit_progress(value, self.phase_total):
                self.pbar.update_absolute(value, self.phase_total, None)
        except Exception:
            pass

    def phase_loaded(self):
        """Model ready; enter the generation phase (never rewinds the bar)."""
        try:
            current = int(getattr(self.pbar, "current", 0) or 0)
        except Exception:
            current = 0
        if not _emit_progress(min(1, current), self.phase_total):
            self.pbar.update_absolute(min(1, current), self.phase_total, None)

    def phase_done(self):
        """Generation finished; complete the bar."""
        self.cbar.close("audio ready")
        if not _emit_progress(self.phase_total, self.phase_total):
            self.pbar.update_absolute(self.phase_total, self.phase_total, None)

class VoiceDesignNode:
    """
    VoiceDesign Node: Generate custom voices based on text descriptions.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello [pause:0.5] world. [happy] This is cheerful!", "placeholder": "Enter text to synthesize"}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (required for VoiceDesign)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "quantization": (["none", "w8a8", "int8"], {"default": "none"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceDesign: Generate custom voices from descriptions."

    def generate(self, text: str, instruct: str, model_choice: str, device: str, precision: str, language: str, quantization: str = "none", seed: int = 0, max_new_tokens: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not text or not instruct:
            raise RuntimeError("Text and instruction description are required")

        pbar = TTSProgress(max_new_tokens, text=text)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "VoiceDesign":
                previous_attention = key[5] if len(key) > 5 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("VoiceDesign", model_choice, device, precision, quantization, attention, unload_model_after_generate, previous_attention)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.phase_loaded()

        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        instruct_arg = instruct if instruct and instruct.strip() else None

        segments = parse_text_with_inline_tags(text)

        pbar.register()

        def render_design(text_content: str):
            clean_text, combined_instruct = extract_emotion_instruct(text_content, instruct_arg)
            wavs, seg_sr = model.generate_voice_design(
                text=[clean_text],
                language=mapped_lang,
                instruct=combined_instruct,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            return wavs[0], seg_sr

        audio_data = _render_segments(segments, render_design)

        pbar.phase_done()
        pbar.unregister()

        if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
            model._unload_callback()

        return (audio_data,)


class VoiceCloneNode:
    """
    VoiceClone (Base) Node: Create clones from reference audio and synthesize target text.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "target_text": ("STRING", {"multiline": True, "default": "Good one. [pause:0.5] Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "quantization": (["none", "w8a8", "int8"], {"default": "none"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional)"}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT", {"tooltip": "Reusable voice clone prompt from VoiceClonePromptNode"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
                "custom_model_path": ("STRING", {
                    "default": "",
                    "placeholder": "Folder name inside models/qwen-tts (or output/qwen3tts_finetune)",
                    "tooltip": "Optional fine-tuned model folder, given as a path RELATIVE to "
                               "ComfyUI's models/qwen-tts directory (or to output/qwen3tts_finetune "
                               "for checkpoints produced by the Train node). Absolute paths, drive "
                               "letters and network share paths are refused: leave empty to use the "
                               "standard model lookup.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceClone: Clone voice from reference audio."

    def _audio_tensor_to_tuple(self, audio_tensor: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        # Accept multiple possible ComfyUI audio formats
        waveform = None
        sr = None
        try:
            if isinstance(audio_tensor, dict):
                # Common keys: 'waveform'/'sample_rate' or 'data'/'sampling_rate'
                if "waveform" in audio_tensor:
                    waveform = audio_tensor.get("waveform")
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr") or audio_tensor.get("sampling_rate")
                elif "data" in audio_tensor and "sampling_rate" in audio_tensor:
                    waveform = np.asarray(audio_tensor.get("data"))
                    sr = audio_tensor.get("sampling_rate")
                elif "audio" in audio_tensor and isinstance(audio_tensor["audio"], (tuple, list)):
                    # maybe {'audio': (sr, data)}
                    a = audio_tensor["audio"]
                    if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], (int, float)):
                        sr, waveform = int(a[0]), np.asarray(a[1])
                else:
                    # fallback: try common numeric fields
                    for k in ("samples", "y", "wave"):
                        if k in audio_tensor:
                            waveform = np.asarray(audio_tensor.get(k))
                            break
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr") or audio_tensor.get("sampling_rate") or audio_tensor.get("rate")
            elif isinstance(audio_tensor, tuple) and len(audio_tensor) == 2:
                # (waveform, sr) or (sr, waveform)
                a0, a1 = audio_tensor
                if isinstance(a0, (int, float)) and isinstance(a1, (list, np.ndarray, torch.Tensor)):
                    sr = int(a0)
                    waveform = np.asarray(a1)
                elif isinstance(a1, (int, float)) and isinstance(a0, (list, np.ndarray, torch.Tensor)):
                    sr = int(a1)
                    waveform = np.asarray(a0)
            elif isinstance(audio_tensor, list):
                # maybe [waveform, sr]
                if len(audio_tensor) == 2 and isinstance(audio_tensor[0], (list, np.ndarray)) and isinstance(audio_tensor[1], (int, float)):
                    waveform = np.asarray(audio_tensor[0])
                    sr = int(audio_tensor[1])
        except Exception:
            pass
        # Normalize to 1-D numpy float32 array (model expects 1-D waveforms)
        if isinstance(waveform, torch.Tensor):
            # ComfyUI audio is often [batch, channels, samples] or [channels, samples]
            if waveform.dim() > 1:
                # Squeeze out any unit dimensions (like batch=1, channel=1)
                waveform = waveform.squeeze()
                # If still multi-dimensional (e.g. stereo), average to mono
                if waveform.dim() > 1:
                    waveform = torch.mean(waveform, dim=0)
            waveform = waveform.cpu().numpy()

        if isinstance(waveform, np.ndarray):
            # Double check for numpy version of same logic
            if waveform.ndim > 1:
                waveform = np.squeeze(waveform)
                if waveform.ndim > 1:
                    # Heuristic: the smaller dimension is likely channels
                    if waveform.shape[0] < waveform.shape[1]:
                        waveform = np.mean(waveform, axis=0)
                    else:
                        waveform = np.mean(waveform, axis=1)
            waveform = waveform.astype(np.float32)

        # Final safety flatten to ensure it's 1-D
        if waveform is not None and waveform.ndim > 1:
            waveform = waveform.flatten()

        # Basic validation
        if waveform is None or not isinstance(waveform, np.ndarray) or waveform.size == 0:
            raise RuntimeError("Failed to parse reference audio waveform")
        
        min_samples = 1024
        if waveform.size < min_samples:
            # Pad with zeros to avoid upstream padding errors
            pad_to = min_samples
            pad_amount = pad_to - waveform.size
            waveform = np.concatenate([waveform, np.zeros(pad_amount, dtype=np.float32)])

        # Return as tuple (waveform, sr) with 1-D numpy waveform as expected by the tokenizer
        return (waveform, int(sr))

    def generate(self, target_text: str, model_choice: str, device: str, precision: str, language: str, quantization: str = "none", 
                 ref_audio: Optional[Dict[str, Any]] = None, ref_text: str = "", 
                 voice_clone_prompt: Optional[Any] = None, seed: int = 0, 
                 max_new_tokens: int = 2048,
                 top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05,
                 x_vector_only: bool = False, attention: str = "auto",
                 unload_model_after_generate: bool = False, custom_model_path: str = "") -> Tuple[Dict[str, Any]]:
        if ref_audio is None and voice_clone_prompt is None:
            raise RuntimeError("Either reference audio or voice clone prompt is required")

        pbar = TTSProgress(max_new_tokens, text=target_text)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[5] if len(key) > 5 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("Base", model_choice, device, precision, quantization, attention, unload_model_after_generate, previous_attention, custom_model_path)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        pbar.phase_loaded()

        audio_tuple = None
        if ref_audio is not None:
            if isinstance(ref_audio, dict):
                audio_tuple = self._audio_tensor_to_tuple(ref_audio)
            else:
                raise RuntimeError("Unsupported reference audio format")

            try:
                mapped_lang = LANGUAGE_MAP.get(language, "auto")

                voice_clone_prompt_param = None
                ref_audio_param = None
                if voice_clone_prompt is not None:
                    voice_clone_prompt_param = voice_clone_prompt
                elif ref_audio is not None:
                    ref_audio_param = audio_tuple
                else:
                    raise RuntimeError("Either 'ref_audio' or 'voice_clone_prompt' must be provided")

                segments = parse_text_with_inline_tags(target_text)

                pbar.register()

                def render_clone(text_content: str):
                    clean_text, combined_instruct = extract_emotion_instruct(text_content, None)
                    wavs, seg_sr = model.generate_voice_clone(
                        text=[clean_text],
                        language=mapped_lang,
                        ref_audio=ref_audio_param,
                        ref_text=ref_text if ref_text and ref_text.strip() else None,
                        voice_clone_prompt=voice_clone_prompt_param,
                        x_vector_only_mode=x_vector_only,
                        instruct=[combined_instruct] if combined_instruct else None,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                    )
                    return wavs[0], seg_sr

                audio_data = _render_segments(segments, render_clone)

            except Exception as e:
                raise RuntimeError(f"Generation failed: {e}")

            pbar.phase_done()
            pbar.unregister()

            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()

            return (audio_data,)


class CustomVoiceNode:
    """
    CustomVoice (TTS) Node: Generate text-to-speech using preset speakers.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello [pause:0.5] world. [happy] This is cheerful!", "placeholder": "Enter text to synthesize"}),
                "speaker": (["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"], {"default": "Ryan"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "quantization": (["none", "w8a8", "int8"], {"default": "none"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (optional)"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
                "custom_model_path": ("STRING", {
                    "default": "",
                    "placeholder": "Folder name inside models/qwen-tts (or output/qwen3tts_finetune)",
                    "tooltip": "Optional fine-tuned model folder, given as a path RELATIVE to "
                               "ComfyUI's models/qwen-tts directory (or to output/qwen3tts_finetune "
                               "for checkpoints produced by the Train node). Absolute paths, drive "
                               "letters and network share paths are refused: leave empty to use the "
                               "standard model lookup.",
                }),
                "custom_speaker_name": ("STRING", {"default": "", "placeholder": "Custom speaker name (for fine-tuned models)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "CustomVoice: Generate speech using preset speakers."

    def generate(self, text: str, speaker: str, model_choice: str, device: str, precision: str, language: str, quantization: str = "none", seed: int = 0, instruct: str = "", max_new_tokens: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False, custom_model_path: str = "", custom_speaker_name: str = "") -> Tuple[Dict[str, Any]]:
        # Prefer custom_speaker_name if provided
        target_speaker = speaker
        if custom_speaker_name and custom_speaker_name.strip():
            target_speaker = custom_speaker_name.strip()
        else:
            target_speaker = speaker.lower().replace(" ", "_")

        if not text:
            raise RuntimeError("Text is required")

        pbar = TTSProgress(max_new_tokens, text=text)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "CustomVoice":
                previous_attention = key[5] if len(key) > 5 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("CustomVoice", model_choice, device, precision, quantization, attention, unload_model_after_generate, previous_attention, custom_model_path)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.phase_loaded()

        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        instruct_arg = instruct if instruct and instruct.strip() else None

        segments = parse_text_with_inline_tags(text)

        pbar.register()

        def render_custom(text_content: str):
            clean_text, combined_instruct = extract_emotion_instruct(text_content, instruct_arg)
            wavs, seg_sr = model.generate_custom_voice(
                text=[clean_text],
                language=[mapped_lang],
                speaker=target_speaker,
                instruct=[combined_instruct] if combined_instruct else None,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            return wavs[0], seg_sr

        audio_data = _render_segments(segments, render_custom)

        pbar.phase_done()
        pbar.unregister()

        if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
            model._unload_callback()

        return (audio_data,)


class VoiceClonePromptNode:
    """
    VoiceClonePrompt Node: Extract voice features from reference audio for reuse.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (highly recommended for better quality)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "quantization": (["none", "w8a8", "int8"], {"default": "none"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip": "If True, only speaker embedding is extracted (ref_text not needed)"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceClonePrompt: Extract and cache voice features for reuse in VoiceClone node."

    def create_prompt(self, ref_audio: Dict[str, Any], ref_text: str, model_choice: str, device: str, precision: str, attention: str, quantization: str = "none", x_vector_only: bool = False, unload_model_after_generate: bool = False) -> Tuple[Any]:
        if ref_audio is None:
            raise RuntimeError("Reference audio is required")

        # prompt extraction is a single encoder pass, not token generation:
        # there is no max_new_tokens here and no per-token progress to report
        pbar = TTSProgress(1, text="")

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[5] if len(key) > 5 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("Base", model_choice, device, precision, quantization, attention, unload_model_after_generate, previous_attention)

        pbar.phase_loaded()

        vcn = VoiceCloneNode()
        pbar.register()
        audio_tuple = vcn._audio_tensor_to_tuple(ref_audio)

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text if ref_text and ref_text.strip() else None,
            x_vector_only_mode=x_vector_only,
        )

        pbar.phase_done()
        pbar.unregister()

        if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
            model._unload_callback()

        return (prompt_items,)


class RoleBankNode:
    """
    RoleBank Node: Manage a collection of voice prompts mapped to names.
    Supports up to 8 roles per node.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs = {
            "required": {},
            "optional": {}
        }
        for i in range(1, 9):
            inputs["optional"][f"role_name_{i}"] = ("STRING", {"default": f"Role{i}"})
            inputs["optional"][f"prompt_{i}"] = ("VOICE_CLONE_PROMPT",)
        return inputs

    RETURN_TYPES = ("QWEN3_ROLE_BANK",)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "create_bank"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "RoleBank: Collect multiple voice prompts into a named registry for dialogue generation."

    def create_bank(self, **kwargs) -> Tuple[Dict[str, Any]]:
        bank = {}
        for i in range(1, 9):
            name = kwargs.get(f"role_name_{i}", "").strip()
            prompt = kwargs.get(f"prompt_{i}")
            if name and prompt is not None:
                bank[name] = prompt
        return (bank,)


class DialogueInferenceNode:
    """
    DialogueInference Node: Generate multi-role continuous dialogue from a script.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": "Role1: Hello, how are you?\nRole2: I am fine, thank you.", "placeholder": "Format: RoleName: Text"}),
                "role_bank": ("QWEN3_ROLE_BANK",),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "quantization": (["none", "w8a8", "int8"], {"default": "none"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
                # RENAMED: pause_seconds -> pause_linebreak
                # Linebreak Pause
                "pause_linebreak": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration between lines"}),
                # NEW: Period and Comma pause settings
                "period_pause": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after periods (.)"}),
                "comma_pause": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after commas (,)"}),
                # ### NEW INPUTS ###
                "question_pause": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after question marks (?)"}),
                "hyphen_pause": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after hyphens (-)"}),
                
                "merge_outputs": ("BOOLEAN", {"default": True, "tooltip": "Merge all dialogue segments into a single long audio"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1, "tooltip": "Number of lines to process in parallel. Larger = faster but more VRAM."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens_per_line": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "DialogueInference: Execute a script with multiple roles and generate continuous speech."
    # ### UPDATED ARGUMENTS: Added question_pause, hyphen_pause
    def generate_dialogue(self, script: str, role_bank: Dict[str, Any], model_choice: str, device: str, precision: str, language: str, pause_linebreak: float, period_pause: float, comma_pause: float, question_pause: float, hyphen_pause: float, merge_outputs: bool, batch_size: int, quantization: str = "none", seed: int = 0, max_new_tokens_per_line: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not script or not role_bank:
            raise RuntimeError("Script and Role Bank are required")

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[5] if len(key) > 5 else None
                break

        model = load_qwen_model("Base", model_choice, device, precision, quantization, attention, unload_model_after_generate, previous_attention)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        lines = script.strip().split("\n")

        texts_to_gen = []
        prompts_to_gen = []
        langs_to_gen = []
        pauses_to_gen = []

        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        pause_pattern = r'\[break=([\d\.]+)\]'

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or ":" not in line and "：" not in line:
                continue

            pos_en = line.find(":")
            pos_cn = line.find("：")

            if pos_en == -1 and pos_cn == -1:
                continue

            if pos_en != -1 and (pos_cn == -1 or pos_en < pos_cn):
                role_name, text = line.split(":", 1)
            else:
                role_name, text = line.split("：", 1)

            role_name = role_name.strip()
            text = text.strip()

            if role_name not in role_bank:
                continue
            # role_bank[role_name] can be a prompt list OR the new packaged dict (if loaded via RoleBank from LoadSpeaker)
            role_data = role_bank[role_name]
            
            # Extract ref_text if it was passed through RoleBank
            current_prompt = role_data[0] if isinstance(role_data, list) else role_data
            current_ref_text = ""

            config = {
                "pause_linebreak": pause_linebreak,
                "period_pause": period_pause,
                "comma_pause": comma_pause,
                "question_pause": question_pause,
                "hyphen_pause": hyphen_pause
            }
            
            segments = split_text_by_pauses(text, config)

            for segment_text, current_segment_pause in segments:
                pronounceable = re.sub(r'[^\w\u4e00-\u9fa5]', '', segment_text)
                if not pronounceable.strip():
                    continue

                # Strip trailing flat punctuation to prevent robotic intonation dropping on short segments
                # We keep ? and ! because they convey necessary emotive intonation.
                clean_seg_text = re.sub(r'[\.,，。…\-—]+$', '', segment_text).strip()
                if not clean_seg_text:
                    clean_seg_text = segment_text

                texts_to_gen.append(clean_seg_text)
                prompts_to_gen.append(current_prompt)
                langs_to_gen.append(mapped_lang)
                pauses_to_gen.append(current_segment_pause)

            if pauses_to_gen:
                pauses_to_gen[-1] += pause_linebreak

        if not texts_to_gen:
            raise RuntimeError("No valid dialogue lines found matching Role Bank.")

        num_lines = len(texts_to_gen)
        num_chunks = (num_lines + batch_size - 1) // batch_size
        total_stages = num_chunks + 1
        # One bar for the whole node, scaled so every chunk owns a slice:
        # chunk boundaries are coarse jumps, tokens fill in between.
        slots_per_chunk = 100
        bar_total = max(1, num_chunks * slots_per_chunk)
        # Budget from the text that will actually be spoken, not from the
        # unreachable max_new_tokens ceiling (same reason as the single-shot
        # nodes). Re-estimated per chunk inside the loop.
        _spoken_chars = sum(len(t) for t in texts_to_gen)
        pbar = TTSProgress(
            TTSProgress.estimate_budget("x" * _spoken_chars, max_new_tokens_per_line * batch_size),
            phase_total=bar_total,
            chunk_total=num_chunks,
        )
        pbar.phase_loaded()
        _emit_progress(slots_per_chunk, bar_total)
        pbar.register(slots_per_chunk)

        try:
            results = []
            sr = 24000

            for i in range(0, num_lines, batch_size):
                chunk_texts = texts_to_gen[i:i + batch_size]
                chunk_prompts = prompts_to_gen[i:i + batch_size]
                chunk_langs = langs_to_gen[i:i + batch_size]
                chunk_pauses = pauses_to_gen[i:i + batch_size]

                current_chunk = i // batch_size + 1
                # re-estimate for this chunk's own text so the slice fills evenly
                pbar.max_new_tokens = TTSProgress.estimate_budget(
                    "".join(chunk_texts), max_new_tokens_per_line * batch_size
                )
                print(f"[Qwen3-TTS] Running batched inference for chunk {current_chunk} of {num_chunks}...")

                wavs_list, sr = model.generate_voice_clone(
                    text=chunk_texts,
                    language=chunk_langs,
                    voice_clone_prompt=chunk_prompts,
                    max_new_tokens=max_new_tokens_per_line,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )

                for j, wav in enumerate(wavs_list):
                    waveform = torch.from_numpy(wav).float()
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)
                    elif waveform.ndim == 2:
                        waveform = waveform.unsqueeze(0)
                        if waveform.shape[1] > 1:
                            waveform = torch.mean(waveform, dim=1, keepdim=True)

                    results.append(waveform)

                    this_pause = chunk_pauses[j]
                    if this_pause > 0:
                        silence_len = int(this_pause * sr)
                        silence = torch.zeros((1, 1, silence_len))
                        results.append(silence)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.finish_generation()
                _emit_progress(current_chunk * slots_per_chunk, bar_total)

        except Exception as e:
            raise RuntimeError(f"Dialogue generation failed during chunked inference: {e}")

        if not results:
            raise RuntimeError("No dialogue lines were successfully generated.")

        _emit_progress(bar_total, bar_total)

        if merge_outputs:
            # Concatenate along the sample dimension (last one)
            merged_waveform = torch.cat(results, dim=-1)
            audio_data = {"waveform": merged_waveform, "sample_rate": sr}
            
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)
        else:
            # Pad to longest for batch format
            max_len = max(w.shape[-1] for w in results)
            padded_results = []
            for w in results:
                curr_len = w.shape[-1]
                if curr_len < max_len:
                    padding = torch.zeros((w.shape[0], w.shape[1], max_len - curr_len))
                    w = torch.cat([w, padding], dim=-1)
                padded_results.append(w)
            
            batched_waveform = torch.cat(padded_results, dim=0)
            audio_data = {"waveform": batched_waveform, "sample_rate": sr}
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)


class SaveVoiceNode:
    """
    SaveVoice Node: Persist extracted voice features to a file.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",),
                "filename": ("STRING", {"default": "my_custom_voice"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional, but recommended for better quality)"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "SaveVoice: Save voice clone prompt features to disk for later use."

    def save(self, voice_clone_prompt, filename, audio=None, ref_text=""):
        import soundfile as sf
        import json

        # filename comes from a widget, so it is attacker-controlled when a
        # workflow is shared. Reduce it to a bare name and keep every write
        # inside the voices directory.
        filename = _safe_filename(filename, "filename")
        if filename.endswith(".qvp"):
            filename_qvp = filename
            stem = filename[: -len(".qvp")]
        else:
            filename_qvp = filename + ".qvp"
            stem = filename
        filename_wav = stem + ".wav"
        filename_json = stem + ".json"

        # Use ComfyUI models/qwen-tts/voices directory
        output_dir = _voices_dir()
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save features (Raw prompt)
        path_qvp = _resolve_within(output_dir, filename_qvp, "voice features file")
        torch.save(voice_clone_prompt, path_qvp)
        print(f"✅ [Qwen3-TTS] Voice features saved to: {path_qvp}")

        # 2. Save metadata (JSON)
        metadata = {
            "ref_text": ref_text.strip() if ref_text else "",
            "source": "SaveVoiceNode",
            "version": "1.0"
        }
        path_json = _resolve_within(output_dir, filename_json, "voice metadata file")
        try:
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            print(f"✅ [Qwen3-TTS] Voice metadata saved to: {path_json}")
        except Exception as e:
            print(f"⚠️ [Qwen3-TTS] Failed to save metadata JSON: {e}")

        # Optional: Save audio preview as reference WAV
        if audio is not None:
            try:
                # audio is usually {"waveform": tensor, "sample_rate": sr}
                waveform = audio["waveform"]
                sr = audio["sample_rate"]
                
                # Convert from [batch, channels, samples] to [samples, channels]
                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.cpu().numpy()
                else:
                    waveform_np = np.asarray(waveform)
                
                if waveform_np.ndim == 3: # [B, C, S] -> [S, C] (assume batch size 1)
                    waveform_np = waveform_np[0].T
                
                wav_path = _resolve_within(output_dir, filename_wav, "reference audio file")
                sf.write(wav_path, waveform_np, sr)
                print(f"✅ [Qwen3-TTS] Reference audio (Speaker) saved to: {wav_path}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS] Failed to save reference audio: {e}")

        return {}


class LoadSpeakerNode:
    """
    LoadSpeaker Node: Directly load a WAV/Speaker file and its associated features.
    Part of the "Voice Design then Clone" best practice workflow.
    """
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = _voices_dir()
        os.makedirs(output_dir, exist_ok=True)
        # List WAV files in the voices directory
        files = [f for f in os.listdir(output_dir) if f.endswith((".wav", ".mp3", ".flac"))]
        
        return {
            "required": {
                "filename": (files if files else ["None"],),
            },
        }
    
    # Updated: Added ref_text (STRING) output
    RETURN_TYPES = ("VOICE_CLONE_PROMPT", "AUDIO", "STRING")
    RETURN_NAMES = ("voice_clone_prompt", "audio", "ref_text")
    FUNCTION = "load_speaker"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "LoadSpeaker: Load saved WAV audio and its metadata. Fast-loads .qvp features if available."

    def load_speaker(self, filename):
        if filename == "None":
            raise RuntimeError("No speaker files found to load.")

        # filename comes from a widget; keep the read inside the voices directory.
        filename = _safe_filename(filename, "filename")
        voices_dir = _voices_dir()
        wav_path = _resolve_within(voices_dir, filename, "speaker audio file", must_exist=True)

        # 1. Load the AUDIO for output/preview
        import librosa
        wav, sr = librosa.load(wav_path, sr=None)
        waveform = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        audio_preview = {"waveform": waveform, "sample_rate": sr}

        # 2. Automatically load metadata and pre-computed features
        qvp_file = os.path.splitext(filename)[0] + ".qvp"
        qvp_path = _resolve_within(voices_dir, qvp_file, "voice features file")
        json_file = os.path.splitext(filename)[0] + ".json"
        json_path = _resolve_within(voices_dir, json_file, "voice metadata file")
        
        prompt_items = None
        ref_text = ""
        
        # 2.1 Load metadata from JSON
        import json
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    if not ref_text or not ref_text.strip():
                        ref_text = metadata.get("ref_text", "")
                        if ref_text:
                            print(f"📖 [Qwen3-TTS] Loaded metadata from JSON: {json_file}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS] Failed to load metadata JSON: {e}")

        # 2.2 Check for pre-computed features (.qvp)
        if os.path.exists(qvp_path):
            try:
                # A .qvp holds VoiceClonePromptItem objects. Load it with
                # weights_only=True so a shared voice file cannot execute
                # embedded code, and register VoiceClonePromptItem as a safe
                # global so the user's own saved voices still load.
                _register_safe_globals()
                try:
                    data = torch.load(qvp_path, map_location="cpu", weights_only=True)
                except TypeError as exc:
                    # torch without the weights_only argument (pre-1.13).
                    # Do NOT retry without it: an unrestricted load would let a
                    # shared voice file execute code. Skip the fast path and
                    # recompute the features from the audio instead.
                    print(
                        f"[Qwen3-TTS] This torch build cannot load .qvp safely "
                        f"({exc}); recomputing features from the audio instead."
                    )
                    raise
                
                # Support legacy packaged format and raw format
                if isinstance(data, dict) and "prompt" in data:
                    prompt_items = data["prompt"]
                    if not ref_text or not ref_text.strip():
                        ref_text = data.get("ref_text", "")
                else:
                    # Raw prompt data
                    prompt_items = data
                
                print(f"🚀 [Qwen3-TTS] Fast-loaded pre-computed features from: {qvp_file}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS] Failed to fast-load .qvp: {e}")

        # Final Return (including the text)
        # Note: If prompt_items is None, the downstream VoiceCloneNode will extract it using its own settings.
        return (prompt_items, audio_preview, ref_text)
