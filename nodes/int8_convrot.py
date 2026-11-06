"""INT8-ConvRot load path for Qwen3-TTS.

Loads a pre-converted *_int8_convrot.safetensors checkpoint (produced by
comfy-model-tools/quant_int8_convrot.py), and loads into
Qwen3TTSForConditionalGeneration.

Two modes:

  (a) W8A8 forward path (opt-in via Qwen3TTS_INT8_W8A8=1):
      Wraps on-disk INT8 weights as comfy_kitchen QuantizedTensor with
      TensorWiseINT8Layout and convrot groupsize. Inference dispatches
      through comfy_kitchen.int8_linear(convrot=True) -> torch._int_mm on
      INT8 tensor cores — no dequantization at inference. Measured: ~38%
      less VRAM, ~26% slower end-to-end on RTX 3060 batch=1.

  (b) bf16 dequant path (default, Qwen3TTS_INT8_W8A8=0):
      Dequantizes each int8+convrot layer to `dtype` via
      _dequantize_convrot and loads as regular parameters.

Optionally auto-converts on first load when no checkpoint exists on disk.

Dequantization runs on the GPU (safe_open device="cuda"), the Hadamard matrix
is cached per (groupsize, device, dtype), comfy_quant metadata is pre-scanned
on CPU, TTSProgress is driven during the load, missing keys raise loudly, and
the Hadamard cache is cleared between loads. Peak GPU during load with the
dequant path is the full bf16 model (~3.4 GB for 1.7B) since dequantized
tensors accumulate in a dict until load_state_dict(assign=True) swaps in
pointers.
"""

import json
import os
import subprocess
import sys

import torch
from safetensors import safe_open

from comfy_kitchen.tensor import QuantizedTensor

from ._boot import (
    COMFY_ROOT,
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSModel,
    _COMFY_KITCHEN_AVAILABLE,
    TensorWiseINT8Layout,
    _build_hadamard,
    _rotate_weight,
    init_empty_weights,
)
from .progress import TTSProgress

CONVERTER_SCRIPT = os.path.join(COMFY_ROOT, "custom_nodes", "comfy-model-tools", "quant_int8_convrot.py")
INT8_CONVROT_EXCLUDE = ("codec_encoder|codec_decoder|speaker_encoder|vocoder|"
                        "seanet|audio_codec|feature_projection|output_proj|lm_head")

# Hadamard matrices cached by (groupsize, device, dtype) so we never rebuild
# the same matrix twice during a load.
_HADAMARD_CACHE = {}
# Log the comfy-kitchen -> manual fallback path exactly once per process.
_FALLBACK_LOGGED = False


def _resolve_model_source(source):
    """Resolve a model directory to the actual weight file for the converter.

    The converter opens SRC as a single file (.safetensors via safe_open, or
    .pth/.pt/.ckpt via torch.load). A HuggingFace snapshot directory is not a
    file, so we look for model.safetensors (or any .safetensors) inside it.
    """
    if os.path.isfile(source):
        return source
    if os.path.isdir(source):
        candidate = os.path.join(source, "model.safetensors")
        if os.path.isfile(candidate):
            return candidate
        for f in os.listdir(source):
            if f.lower().endswith(".safetensors"):
                return os.path.join(source, f)
    return source


def _decode_comfy_quant(raw):
    """Decode a uint8 comfy_quant tensor to a dict."""
    text = bytes(raw.tolist()).decode("utf-8")
    return json.loads(text)


def _dequantize_convrot(qd, scale, gs, device, out_dtype=torch.bfloat16):
    """Dequantize an int8+convrot weight to `out_dtype`, entirely on `device`.

    Approach A: comfy_kitchen TensorWiseINT8Layout.dequantize (preferred).
    Approach B: manual _build_hadamard + _rotate_weight inverse (fallback).
    """
    global _FALLBACK_LOGGED
    # NOTE: must use kwargs form — to(dtype, device=...) is not a valid overload.
    qd = qd.to(device=device, dtype=torch.int8, non_blocking=True)
    scale = scale.to(device=device, non_blocking=True)

    if _COMFY_KITCHEN_AVAILABLE and TensorWiseINT8Layout is not None and not _FALLBACK_LOGGED:
        try:
            params = TensorWiseINT8Layout.Params(
                scale=scale,
                orig_dtype=out_dtype,
                orig_shape=tuple(qd.shape),
                convrot=True,
                convrot_groupsize=gs,
            )
            deq = TensorWiseINT8Layout.dequantize(qd, params)
            return deq.to(device=device, dtype=out_dtype, non_blocking=True)
        except Exception as e:
            _FALLBACK_LOGGED = True
            print(
                "⚠️ [Qwen3-TTS] int8-convrot: comfy-kitchen dequantize failed: "
                f"{e}; using manual Hadamard fallback"
            )

    # Approach B: manual Hadamard inverse rotation (H is self-inverse).
    # Rotate in out_dtype before applying the per-tensor scalar scale; scale
    # commutes with the rotation so this is mathematically equivalent to
    # dequant-then-rotate-then-scale.
    cache_key = (gs, str(device), out_dtype)
    h = _HADAMARD_CACHE.get(cache_key)
    if h is None:
        h = _build_hadamard(gs, device=device, dtype=out_dtype)
        _HADAMARD_CACHE[cache_key] = h
    deq = qd.to(out_dtype)
    deq = _rotate_weight(deq, h, gs)
    deq = deq * scale.to(out_dtype)
    return deq


def load_int8_convrot_model(source, int8_convrot_path, device, dtype, **kwargs):
    """Load Qwen3-TTS from an int8-convrot checkpoint, dequantizing to bf16.

    Args:
        source: BF16 model folder (used as converter SRC when auto-converting).
        int8_convrot_path: Path to *_int8_convrot.safetensors (or where it is created).
        device: Target device ("cuda").
        dtype: Compute dtype (int8 weights dequantize to bf16; dtype is compute-only).
        **kwargs: attn_implementation, load_speech_tokenizer, auto_convert.

    Returns:
        Qwen3TTSModel wrapper ready for generation.

    Raises:
        RuntimeError: if comfy-kitchen is unavailable, conversion fails, or
            no checkpoint exists and auto_convert is False, or keys are missing
            after load_state_dict.
    """
    if not _COMFY_KITCHEN_AVAILABLE:
        raise RuntimeError(
            "[Qwen3-TTS] int8-convrot requires comfy-kitchen (comfy_kitchen package). "
            "Install it: pip install comfy-kitchen"
        )

    attn_implementation = kwargs.get("attn_implementation")
    load_speech_tokenizer = kwargs.get("load_speech_tokenizer", False)
    auto_convert = kwargs.get("auto_convert", False)

    # Resolve compute dtype from the `dtype` argument (bf16 default).
    if isinstance(dtype, torch.dtype):
        out_dtype = dtype
    else:
        _DTYPE_MAP = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16,  "float16":  torch.float16,
            "fp32": torch.float32,  "float32":  torch.float32,
        }
        out_dtype = _DTYPE_MAP.get(str(dtype).lower(), torch.bfloat16)

    # Clear any Hadamard matrices left over from a previous load with a
    # different groupsize, so they don't sit in VRAM forever.
    _HADAMARD_CACHE.clear()

    # --- 1. Check for pre-converted checkpoint --------------------------------
    if not os.path.exists(int8_convrot_path):
        if not auto_convert:
            raise RuntimeError(
                f"[Qwen3-TTS] int8-convrot: no checkpoint found at {int8_convrot_path} "
                f"and auto_convert is disabled."
            )
        # --- 2. Auto-convert on first load -------------------------------------
        print(f"🔧 [Qwen3-TTS] int8-convrot: no saved weights found at {int8_convrot_path}; "
              f"running one-time conversion...")

        conv_source = _resolve_model_source(source)

        # Validate with --dry-run first
        dry_run_cmd = [
            sys.executable, CONVERTER_SCRIPT, conv_source, int8_convrot_path,
            "--dry-run",
            "--exclude", INT8_CONVROT_EXCLUDE,
            "--min-gemm", "256",
        ]
        print(f"🔧 [Qwen3-TTS] int8-convrot: dry-run: {' '.join(dry_run_cmd)}")
        try:
            result = subprocess.run(dry_run_cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"int8-convrot dry-run failed (exit {result.returncode}):\n{result.stderr}"
                )
            for line in result.stdout.splitlines():
                if line.strip():
                    print(f"   {line}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("int8-convrot dry-run timed out (>10 min).")
        except FileNotFoundError:
            raise RuntimeError(
                f"int8-convrot converter not found at {CONVERTER_SCRIPT}. "
                f"Ensure comfy-model-tools is cloned alongside this plugin."
            )

        # Full conversion
        convert_cmd = [
            sys.executable, CONVERTER_SCRIPT, conv_source, int8_convrot_path,
            "--exclude", INT8_CONVROT_EXCLUDE,
            "--min-gemm", "256",
            "--mseclip",
            "--warn-thresh", "2.0",
        ]
        print(f"🔧 [Qwen3-TTS] int8-convrot: converting: {' '.join(convert_cmd)}")
        proc = None
        pbar = None
        try:
            pbar = TTSProgress(max_new_tokens=100, text="int8-convrot conversion")
            pbar.phase_loaded()
        except Exception:
            pbar = None

        try:
            proc = subprocess.Popen(
                convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.strip():
                    print(f"   {line}")
                    if pbar is not None:
                        try:
                            cur = getattr(pbar.pbar, "current", 0) or 0
                            pbar.pbar.update_absolute(min(99, int(cur) + 1), 100, None)
                        except Exception:
                            pass

            proc.wait(timeout=3600)
            if proc.returncode != 0:
                raise RuntimeError(f"int8-convrot conversion failed (exit {proc.returncode}).")
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
            raise RuntimeError("int8-convrot conversion timed out (>60 min).")
        finally:
            if pbar is not None:
                try:
                    pbar.phase_done()
                    pbar.unregister()
                except Exception:
                    pass

        print(f"✅ [Qwen3-TTS] int8-convrot: conversion complete -> {int8_convrot_path}")

    # --- 3. Register custom classes for config resolution ---------------------
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor, exist_ok=True)

    # --- 4. Read config ---------------------------------------------------------
    print(f"🔧 [Qwen3-TTS] int8-convrot: loading config from {source}")
    config = Qwen3TTSConfig.from_pretrained(source, local_files_only=True)
    if attn_implementation:
        config._attn_implementation = attn_implementation

    # --- 5. Build empty skeleton on meta device --------------------------------
    print(f"🔧 [Qwen3-TTS] int8-convrot: building empty model skeleton")
    with init_empty_weights():
        model = Qwen3TTSForConditionalGeneration(config)

    # --- 6. Read checkpoint and dequantize int8+convrot layers ------------------
    print(f"🔧 [Qwen3-TTS] int8-convrot: reading {int8_convrot_path}")
    target_device = torch.device(device)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        st_device = str(device)
    else:
        st_device = "cpu"
        print("⚠️ [Qwen3-TTS] int8-convrot: cuda unavailable; reading checkpoint on cpu")

    state_dict = {}
    with safe_open(int8_convrot_path, framework="pt", device=st_device) as st:
        keys = list(st.keys())
        weight_scale_keys = {k for k in keys if k.endswith(".weight_scale")}
        comfy_quant_keys = {k for k in keys if k.endswith(".comfy_quant")}
        quantized_bases = {
            k[:-len(".weight_scale")]
            for k in weight_scale_keys
            if f"{k[:-len('.weight_scale')]}.comfy_quant" in comfy_quant_keys
        }
        plain_keys = [k for k in keys
                      if not k.endswith(".weight_scale") and not k.endswith(".comfy_quant")]
        total = len(plain_keys)

        # Pre-scan all comfy_quant metadata on CPU before the GPU loop so the
        # per-tensor host decode (.tolist() -> bytes -> json) never stalls the
        # GPU pipeline.
        groupsize_map = {}
        for base in quantized_bases:
            cq_raw = st.get_tensor(f"{base}.comfy_quant").cpu()
            cfg = _decode_comfy_quant(cq_raw)
            groupsize_map[base] = int(cfg.get("convrot_groupsize", 256))

        pbar = None
        try:
            pbar = TTSProgress(max_new_tokens=max(1, total), phase_total=100)
            pbar.phase_loaded()
        except Exception:
            pbar = None

        done = 0
        # Default is the bf16 dequant path (W8A8 measured 1.26x slower end-to-end
        # on RTX 3060 batch=1). Set Qwen3TTS_INT8_W8A8=1 to opt into the INT8
        # forward path (saves ~38% VRAM, does not speed up generation).
        use_w8a8 = os.environ.get("Qwen3TTS_INT8_W8A8", "0") == "1"
        for key in plain_keys:
            base = key[:-len(".weight")] if key.endswith(".weight") else None
            if base is not None and base in quantized_bases:
                qd = st.get_tensor(key)
                scale = st.get_tensor(f"{base}.weight_scale")
                gs = groupsize_map[base]
                if not use_w8a8:
                    # Kill switch: dequantize to bf16 end to end.
                    state_dict[key] = _dequantize_convrot(qd, scale, gs, st_device, out_dtype=out_dtype)
                else:
                    try:
                        qt = QuantizedTensor(
                            qdata=qd,
                            layout_cls="TensorWiseINT8Layout",
                    params=TensorWiseINT8Layout.Params(
                        scale=scale.to(torch.float32),
                        orig_dtype=out_dtype,
                        orig_shape=tuple(qd.shape),
                        convrot=True,
                        convrot_groupsize=gs,
                    ),
                        )
                        state_dict[key] = qt
                    except Exception as e:
                        print(
                            f"⚠️ [Qwen3-TTS] int8-convrot: QuantizedTensor wrap failed for "
                            f"{key}: {e}; falling back to dequantize"
                        )
                        state_dict[key] = _dequantize_convrot(qd, scale, gs, st_device)
            else:
                # Non-quantized tensor: move to GPU, no rotation needed.
                state_dict[key] = st.get_tensor(key).to(target_device, non_blocking=True)

            done += 1
            if pbar is not None and (done % 16 == 0 or done == total):
                try:
                    pct = int(done / total * 100)
                    pbar.pbar.update_absolute(pct, 100, None)
                except Exception:
                    pass

    if pbar is not None:
        try:
            pbar.phase_done()
            pbar.unregister()
        except Exception:
            pass

    # --- 7. Load into model (meta-safe) -----------------------------------------
    print(f"🚀 [Qwen3-TTS] int8-convrot: loading {len(state_dict)} tensors")
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    missing = list(getattr(result, "missing_keys", None) or [])
    unexpected = list(getattr(result, "unexpected_keys", None) or [])
    if missing:
        shown = missing[:20]
        raise RuntimeError(
            "[Qwen3-TTS] int8-convrot: missing keys after load_state_dict "
            f"({len(missing)} total, showing first 20):\n  "
            + "\n  ".join(shown)
        )
    print(f"🔧 [Qwen3-TTS] int8-convrot: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()

    # --- 8. Generation config ----------------------------------------------------
    gen_cfg_path = os.path.join(source, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        with open(gen_cfg_path, "r", encoding="utf-8") as f:
            gen_cfg = json.load(f)
        if hasattr(model, "load_generate_config"):
            model.load_generate_config(gen_cfg)
        else:
            model.generate_config = gen_cfg

    # --- 9. Text processor (needed for text tokenization in generate) ---
    processor = None
    try:
        from transformers import AutoTokenizer
        from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, fix_mistral_regex=True)
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)
        print(f"✅ [Qwen3-TTS] int8-convrot: processor loaded from {source}")
    except Exception as e:
        print(f"⚠️ [Qwen3-TTS] int8-convrot: processor load failed: {e}")

    # --- 10. Speech tokenizer (needed for audio decoding in generate) ---
    if load_speech_tokenizer:
        st_dir = os.path.join(source, "speech_tokenizer")
        if not os.path.isdir(st_dir):
            raise RuntimeError(
                f"[Qwen3-TTS] int8-convrot: speech_tokenizer directory not found at {st_dir}. "
                f"generate_voice_design() requires it to decode audio codes to waveform. "
                f"Ensure the model folder contains speech_tokenizer/."
            )
        try:
            from qwen_tts import Qwen3TTSTokenizer
            speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(st_dir, local_files_only=True)
            model.load_speech_tokenizer(speech_tokenizer)
            print(f"✅ [Qwen3-TTS] int8-convrot: speech tokenizer loaded from {st_dir}")
        except Exception as e:
            raise RuntimeError(
                f"[Qwen3-TTS] int8-convrot: failed to load speech tokenizer from {st_dir}: {e}"
            )
    else:
        print(f"⚠️ [Qwen3-TTS] int8-convrot: speech tokenizer skipped (load_speech_tokenizer=False)")

    # --- 11. Wrap in Qwen3TTSModel ----------------------------------------------
    wrap = Qwen3TTSModel(model=model, processor=processor,
                         generate_defaults=getattr(model, "generate_config", None) or {})
    print(f"🚀 [Qwen3-TTS] int8-convrot: model ready on {device}")
    return wrap
