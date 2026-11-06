"""Core model-loading orchestrator for Qwen3-TTS nodes."""

import json
import os
import types
from typing import Optional

import numpy as np
import torch

from comfy import model_management

from ._boot import MODELS_DIR, Qwen3TTSModel
from . import constants
from .int8_convrot import load_int8_convrot_model
from .paths import _model_search_roots, _resolve_custom_model_dir


def unload_cached_model(cache_key=None):
    """Unload cached model(s) and clear GPU memory.

    Args:
        cache_key: If provided, only unload that specific model.
                   If None, unload all cached models.
    """
    models_to_unload = []
    if cache_key and cache_key in constants._MODEL_CACHE:
        print(f"[Qwen3-TTS] Unloading model: {cache_key}...")
        models_to_unload.append(constants._MODEL_CACHE.pop(cache_key))
    elif constants._MODEL_CACHE:
        print(f"[Qwen3-TTS] Unloading {len(constants._MODEL_CACHE)} cached model(s)...")
        models_to_unload = list(constants._MODEL_CACHE.values())
        constants._MODEL_CACHE.clear()

    inner = None
    # Break every reference holding CUDA/CPU tensors so the model object becomes
    # collectable (weights freed from both VRAM and RAM) before empty_cache() runs.
    for model in models_to_unload:
        # Break the wrapper self-cycle introduced by apply_qwen3_patches: a bound
        # MethodType references the wrapper itself, keeping it alive past GC.
        try:
            if hasattr(model, "_normalize_audio_inputs"):
                del model._normalize_audio_inputs
        except Exception:
            pass
        try:
            if hasattr(model, "model") and model.model is not None:
                inner = model.model
                # Clear non-tensor references first
                if hasattr(inner, "speech_tokenizer") and inner.speech_tokenizer is not None:
                    inner.speech_tokenizer = None
                if hasattr(inner, "generate_config"):
                    inner.generate_config = None
                # Drop the wrapper's reference to the inner model BEFORE moving to
                # CPU, so a cpu() failure cannot leave a dangling CUDA reference.
                model.model = None
                # Relocate weights off CUDA so empty_cache() can reclaim VRAM.
                if hasattr(inner, "cpu"):
                    inner.cpu()
                # Drop our local reference so GC can collect the whole model.
                inner = None
        except Exception:
            pass
        try:
            if hasattr(model, "processor"):
                model.processor = None
        except Exception:
            pass
        try:
            model._unload_callback = None
        except Exception:
            pass
        try:
            if hasattr(model, "generate_defaults"):
                model.generate_defaults = {}
        except Exception:
            pass

    # Ensure no lingering local reference to the last inner model survives
    # past the GC/empty_cache calls below.
    inner = None
    del models_to_unload

    import gc
    gc.collect()
    gc.collect()

    model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    gc.collect()
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
        # Temporarily allow network for first-time download. HF_HUB_OFFLINE and
        # TRANSFORMERS_OFFLINE are set globally to prevent metadata lookups during
        # from_pretrained; override them only for this snapshot_download call.
        prev_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        prev_tr_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        try:
            snapshot_download(repo_id=model_id, local_dir=target_dir)
        finally:
            if prev_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = prev_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
            if prev_tr_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = prev_tr_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
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
    if constants._MODELS_CHECKED:
        return

    qwen_root = os.path.join(MODELS_DIR, "qwen-tts")
    os.makedirs(qwen_root, exist_ok=True)

    # Download tokenizer (required by all models)
    tokenizer_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    download_model_if_needed(tokenizer_id, qwen_root)

    constants._MODELS_CHECKED = True


def load_qwen_model(model_type: str, model_choice: str, device: str, precision: str, quantization: str = "none", attention: str = "auto", unload_after: bool = False, previous_attention: str = None, custom_model_path: Optional[str] = None):
    """Shared model loading logic with caching and local path priority"""
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

    if device == "mps" and precision == "bf16":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32

    # VoiceDesign restriction
    if model_type == "VoiceDesign" and model_choice == "0.6B":
        raise RuntimeError("❌ VoiceDesign only supports 1.7B models!")

    # Cache key includes attention implementation, the resolved model folder,
    # and the quantization setting so int8-convrot vs none do not collide
    cache_key = (model_type, model_choice, device, precision, quantization, attn_impl, custom_dir)
    if cache_key in constants._MODEL_CACHE:
        return constants._MODEL_CACHE[cache_key]

    # Clear old cache only when adding a new model with different config
    if constants._MODEL_CACHE:
        unload_cached_model()

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
        qwen_root = os.path.join(MODELS_DIR, "qwen-tts")

        downloaded_path = download_model_if_needed(final_source, qwen_root)
        if downloaded_path:
            final_source = downloaded_path
            print(f"✅ [Qwen3-TTS] Loading downloaded model: {os.path.basename(final_source)}")
        else:
            raise RuntimeError(
                f"❌ [Qwen3-TTS] Model '{final_source}' not found locally and auto-download failed. "
                f"Download it manually into {qwen_root} or check network/HF access."
            )

    if Qwen3TTSModel is None:
        raise RuntimeError(
            "❌ [Qwen3-TTS] Model class is not loaded because the 'qwen_tts' package failed to import. "
            "Please check the ComfyUI console for the detailed 'Critical Import Error' above."
        )

    # Map attention implementation to model loading parameter
    attn_param = None

    if attn_impl == "flash_attn":
        attn_param = "flash_attention_2"
    elif attn_impl == "xformers":
        attn_param = "sdpa"   # transformers has no 'xformers' value; fall back to sdpa
    elif attn_impl == "comfy_kitchen":
        attn_param = "sdpa"   # base path; comfy_kitchen applied via backend hook below
    elif attn_impl == "sage_attn":
        pass  # sageattention no longer supported; falls through to eager below
    elif attn_impl == "sdpa":
        attn_param = "sdpa"
    elif attn_impl == "eager":
        attn_param = "eager"

    # --- int8-convrot quantization (offline-converted checkpoint, dequantize at load) ---
    convrot_loaded = False
    if quantization == "int8-convrot":
        if device != "cuda" or not torch.cuda.is_available():
            print("[Qwen3-TTS] int8-convrot requires CUDA; falling back to bf16.")
            quantization = "none"
        else:
            qwen_root = os.path.join(MODELS_DIR, "qwen-tts")
            model_name = os.path.basename(final_source)
            int8_convrot_path = os.path.join(qwen_root, model_name + "_int8_convrot.safetensors")
            if os.path.exists(int8_convrot_path):
                print(f"🔧 [Qwen3-TTS] int8-convrot: loading pre-quantized weights from disk: {int8_convrot_path}")
                model = load_int8_convrot_model(
                    final_source, int8_convrot_path, device=device, dtype=dtype,
                    attn_implementation=attn_param,
                    load_speech_tokenizer=(model_type == "Base"),
                )
                convrot_loaded = True
            else:
                print(f"🔧 [Qwen3-TTS] int8-convrot: no saved weights found at {int8_convrot_path}; running first-time conversion...")
                model = load_int8_convrot_model(
                    final_source, int8_convrot_path, device=device, dtype=dtype,
                    attn_implementation=attn_param,
                    load_speech_tokenizer=(model_type == "Base"),
                    auto_convert=True,
                )
                convrot_loaded = True

    if not convrot_loaded:
        # Normal bf16/fp32 load via Qwen3TTSModel.from_pretrained
        load_kwargs = {
            "device_map": device,
            "dtype": dtype,
            "local_files_only": True,
            "use_safetensors": True,
        }
        if attn_param:
            load_kwargs["attn_implementation"] = attn_param
        print(f"🔧 [Qwen3-TTS] Loading model with attention: {attn_impl}")
        model = Qwen3TTSModel.from_pretrained(final_source, **load_kwargs)

    # Apply patches
    apply_qwen3_patches(model)

    constants._MODEL_CACHE[cache_key] = model

    if unload_after:
        def unload_callback():
            unload_cached_model()
        model._unload_callback = unload_callback
    else:
        model._unload_callback = None

    return model


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
