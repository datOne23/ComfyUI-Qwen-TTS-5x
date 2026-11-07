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


def _destroy_single_model(model):
    if model is None:
        return

    model.__dict__.pop("_normalize_audio_inputs", None)
    model.__dict__.pop("_unload_callback", None)

    inner = model.__dict__.pop("model", None)

    for attr in ("processor", "tokenizer", "vocoder", "audio_encoder"):
        model.__dict__.pop(attr, None)
    if "generate_defaults" in model.__dict__:
        model.__dict__["generate_defaults"] = {}

    if inner is not None:
        for attr in ("speech_tokenizer", "generation_config", "generate_config",
                     "config", "hf_device_map", "hf_quantizer",
                     "processor", "tokenizer"):
            inner.__dict__.pop(attr, None)

        for hook_dict in ("_forward_hooks", "_forward_pre_hooks",
                          "_backward_hooks", "_backward_pre_hooks",
                          "_state_dict_hooks", "_load_state_dict_pre_hooks"):
            d = getattr(inner, hook_dict, None)
            if isinstance(d, dict):
                d.clear()

        try:
            for p in inner.parameters(recurse=True):
                if p.grad is not None:
                    p.grad = None
        except Exception:
            pass

        try:
            inner.cpu()
        except Exception:
            pass

        del inner

    del model


def unload_cached_model(cache_key=None):
    """Unload cached model(s) and clear GPU memory."""
    models_to_unload = []

    if cache_key is not None:
        m = constants._MODEL_CACHE.get(cache_key)
        if m is not None:
            print(f"[Qwen3-TTS] Unloading model: {cache_key}...")
            models_to_unload.append(m)
    else:
        if constants._MODEL_CACHE:
            print(f"[Qwen3-TTS] Unloading {len(constants._MODEL_CACHE)} cached model(s)...")
            models_to_unload.extend(constants._MODEL_CACHE.values())

    if not models_to_unload:
        return

    for i in range(len(models_to_unload)):
        _destroy_single_model(models_to_unload[i])
        models_to_unload[i] = None

    if cache_key is not None:
        constants._MODEL_CACHE.pop(cache_key, None)
    else:
        constants._MODEL_CACHE.clear()

    models_to_unload.clear()
    del models_to_unload

    import gc
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    try:
        model_management.soft_empty_cache()
    except Exception as e:
        print(f"[Qwen3-TTS] soft_empty_cache failed: {e}")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    print("[Qwen3-TTS] Model cache and GPU memory cleared")


def apply_qwen3_patches(model):
    """Apply stability and compatibility patches to the model instance"""
    if model is None:
        return

    if "_normalize_audio_inputs" in model.__dict__:
        return

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
                print(f"[Qwen3-TTS] Unknown audio input type: {type(a)}")
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
        print(f"[Qwen3-TTS] Failed to apply audio normalization patch: {e}")


def download_model_if_needed(model_id: str, qwen_root: str) -> str:
    """Download a specific model if not found locally"""
    folder_name = model_id.split("/")[-1]
    target_dir = os.path.join(qwen_root, folder_name)

    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Model already exists
        return target_dir

    print(f"\n[Qwen3-TTS] Downloading {model_id}...")
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
        print(f"[Qwen3-TTS] {folder_name} downloaded successfully.\n")
        return target_dir
    except ImportError:
        print("[Qwen3-TTS] 'huggingface_hub' not found. Please install it to use auto-download.")
        return None
    except Exception as e:
        print(f"[Qwen3-TTS] Failed to download {model_id}: {e}")
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


def get_qwen_models_root() -> str:
    """Primary writable qwen-tts model directory.

    Uses ComfyUI's configured models directory (folder_paths) so tokenizers,
    downloaded models, and int8 caches land in ComfyUI/models/qwen-tts rather
    than under custom_nodes/ when the plugin is installed as a custom node.
    Falls back to the boot-derived MODELS_DIR for unusual layouts.
    """
    candidates = []
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.models_dir, "qwen-tts"))
    except Exception:
        pass
    candidates.append(os.path.join(MODELS_DIR, "qwen-tts"))
    for root in candidates:
        try:
            if os.path.isdir(root):
                return root
        except Exception:
            pass
    target = candidates[0]
    os.makedirs(target, exist_ok=True)
    return target


def _speech_tokenizer_weights_present(target_dir: str) -> bool:
    try:
        for name in os.listdir(target_dir):
            if name.endswith(".safetensors") or name.endswith(".bin") or name.endswith(".pt"):
                return True
    except Exception:
        pass
    return False


def resolve_speech_tokenizer_dir(speech_tokenizer: str, qwen_root: Optional[str] = None) -> Optional[str]:
    """Resolve an optional speech tokenizer directory.

    Returns None for the 24kHz default, meaning "use the model-bundled tokenizer"
    and preserve the exact existing load path. For non-default choices, searches
    ComfyUI qwen-tts model roots for a validated tokenizer directory. If 48kHz is
    selected and not found, attempts auto-download into the primary qwen-tts root.
    Raises a clear RuntimeError if missing/invalid; never falls back to 24kHz.
    """
    choice = speech_tokenizer or "24kHz (default)"
    if choice in ("24kHz (default)", "24kHz", "Qwen3-TTS-Tokenizer-12Hz"):
        check_and_download_tokenizer()
        return None

    folder = constants.SPEECH_TOKENIZER_FOLDERS.get(choice)
    repo = constants.SPEECH_TOKENIZER_REPOS.get(choice)
    if not folder:
        folder = str(choice)
        repo = repo or (f"takuma104/{folder}" if "48kHz" in folder else folder)

    # Search the same ComfyUI model roots used for other Qwen3-TTS assets.
    # This finds tokenizers placed in ComfyUI/models/qwen-tts even when a legacy
    # boot-derived root (custom_nodes/models) is passed or configured.
    search_roots = list(_model_search_roots())
    if qwen_root:
        try:
            qr = os.path.realpath(qwen_root)
        except Exception:
            qr = qwen_root
        if qr not in search_roots:
            search_roots.insert(0, qr)
    primary_root = qwen_root or get_qwen_models_root()
    try:
        pr = os.path.realpath(primary_root)
    except Exception:
        pr = primary_root
    if pr not in search_roots:
        search_roots.insert(0, pr)

    def _validated(target_dir: str) -> bool:
        return os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, "config.json")) and _speech_tokenizer_weights_present(target_dir)

    checked = []
    for root in search_roots:
        target_dir = os.path.join(root, folder)
        checked.append(target_dir)
        if _validated(target_dir):
            return target_dir

    # 48kHz: auto-download to primary qwen-tts root when not found.
    if "48kHz" in choice or "48kHz" in folder:
        dl_root = get_qwen_models_root()
        expected = os.path.join(dl_root, folder)
        repo_id = repo or f"takuma104/{folder}"
        print(
            f"\n[Qwen3-TTS] 48kHz speech tokenizer not found; attempting auto-download.\n"
            f"  Model: third-party fine-tune (Apache-2.0), base Qwen/Qwen3-TTS-Tokenizer-12Hz\n"
            f"  Repo:  {repo_id}\n"
            f"  Size:  ~578 MB (24kHz input / 48kHz output)\n"
            f"  Dest:  {expected}\n"
            f"  Note:  honors HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/network availability; never falls back to 24kHz.\n"
        )
        result = download_model_if_needed(repo_id, dl_root)
        if result and _validated(result):
            return result
        # Re-check searched roots in case download landed elsewhere.
        checked = []
        for root in search_roots:
            target_dir = os.path.join(root, folder)
            checked.append(target_dir)
            if _validated(target_dir):
                return target_dir
        checked.append(expected)
        raise RuntimeError(
            f"Speech tokenizer '{choice}' (48kHz) could not be downloaded or is incomplete. "
            f"Target: {expected} (must contain config.json and model weights). Hugging Face repo: {repo_id}. "
            f"Searched: {'; '.join(dict.fromkeys(checked))}. "
            f"If offline, download manually or run `python download_models.py --with-48k-tokenizer`. "
            f"The 48kHz tokenizer is a third-party Apache-2.0 fine-tune (base: Qwen/Qwen3-TTS-Tokenizer-12Hz) "
            f"with 24kHz input and 48kHz output; 24kHz is never used as a fallback."
        )

    repo_hint = f" Hugging Face repo: {repo}." if repo else ""
    expected = os.path.join(primary_root, folder)
    raise RuntimeError(
        f"Speech tokenizer '{choice}' is not installed. "
        f"Expected folder: {expected} (must contain config.json and model weights).{repo_hint} "
        f"Searched: {'; '.join(dict.fromkeys(checked))}. "
        f"24kHz is never used as a fallback."
    )


def _codec_target_dtype(codec_model, compute_dtype):
    try:
        first = next(codec_model.parameters())
    except StopIteration:
        return None
    if first.is_floating_point() and first.dtype != compute_dtype:
        return compute_dtype
    return None


def _move_and_cast_codec(speech_tokenizer, device: str, dtype, label: str = "codec"):
    if speech_tokenizer is None:
        print(f"[Qwen3-TTS] {label} move skipped - speech tokenizer not found")
        return
    st_model = getattr(speech_tokenizer, "model", speech_tokenizer)
    try:
        st_model.to(device=device)
    except Exception as e:
        print(f"[Qwen3-TTS] {label} move to {device} failed: {e}")
    try:
        speech_tokenizer.device = torch.device(device)
    except Exception:
        pass
    try:
        _codec_dev = next(st_model.parameters()).device
    except StopIteration:
        _codec_dev = "unknown"
    print(f"[Qwen3-TTS] {label} moved to {_codec_dev}")
    target_dtype = _codec_target_dtype(st_model, dtype)
    if target_dtype is not None:
        try:
            before = next(st_model.parameters()).dtype
            speech_tokenizer.model = st_model.to(target_dtype)
            print(f"[Qwen3-TTS] {label} cast {before} -> {target_dtype}")
        except StopIteration:
            pass


def _apply_speech_tokenizer_override(model, speech_tokenizer_choice: str, qwen_root: str, device: str, dtype):
    tok_dir = resolve_speech_tokenizer_dir(speech_tokenizer_choice, qwen_root)
    if tok_dir is None:
        return speech_tokenizer_choice or "24kHz (default)"

    from qwen_tts import Qwen3TTSTokenizer

    inner = getattr(model, "model", model)
    print(f"[Qwen3-TTS] Loading speech tokenizer override: {tok_dir}")
    st = Qwen3TTSTokenizer.from_pretrained(tok_dir, local_files_only=True)
    _move_and_cast_codec(st, device=device, dtype=dtype, label="speech tokenizer")
    if hasattr(inner, "load_speech_tokenizer"):
        inner.load_speech_tokenizer(st)
    elif hasattr(inner, "speech_tokenizer"):
        inner.speech_tokenizer = st
    else:
        raise RuntimeError(
            "Selected speech tokenizer could not be attached to the loaded TTS model "
            "(no load_speech_tokenizer/speech_tokenizer attribute)."
        )
    try:
        out_sr = int(st.get_output_sample_rate())
    except Exception:
        out_sr = 48000
    print(f"[Qwen3-TTS] Speech tokenizer active: {speech_tokenizer_choice} ({out_sr} Hz output)")
    return speech_tokenizer_choice


def get_model_output_sample_rate(model, default_sr: int = 24000) -> int:
    """Return the active speech tokenizer output sample rate for a loaded model."""
    if model is None:
        return int(default_sr)
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.insert(0, inner)
    for obj in candidates:
        st = getattr(obj, "speech_tokenizer", None)
        if st is None:
            continue
        for target in (st, getattr(st, "model", None)):
            if target is None:
                continue
            getter = getattr(target, "get_output_sample_rate", None)
            if callable(getter):
                try:
                    sr = int(getter())
                    if sr > 0:
                        return sr
                except Exception:
                    pass
            for attr in ("output_sample_rate", "sampling_rate", "sample_rate"):
                try:
                    sr = int(getattr(target, attr))
                    if sr > 0:
                        return sr
                except Exception:
                    pass
            cfg = getattr(target, "config", None)
            for attr in ("output_sample_rate", "sampling_rate", "sample_rate"):
                try:
                    sr = int(getattr(cfg, attr))
                    if sr > 0:
                        return sr
                except Exception:
                    pass
    return int(default_sr)


def load_qwen_model(model_type: str, model_choice: str, device: str, precision: str, quantization: str = "none", attention: str = "auto", unload_after: bool = False, previous_attention: str = None, custom_model_path: Optional[str] = None, speech_tokenizer: str = "24kHz (default)"):
    """Shared model loading logic with caching and local path priority"""
    if previous_attention is not None and previous_attention != attention:
        print(f"[Qwen3-TTS] Attention changed from '{previous_attention}' to '{attention}', clearing cache...")
        unload_cached_model()

    attn_impl = get_attention_implementation(attention)

    # custom_model_path is workflow text: resolve it into ComfyUI's model
    # folders before anything is stat'ed, so a shared workflow cannot point at
    # an arbitrary folder or a remote share.
    custom_dir = _resolve_custom_model_dir(custom_model_path)

    qwen_root = get_qwen_models_root()
    tok_dir = resolve_speech_tokenizer_dir(speech_tokenizer, qwen_root)
    tok_cache_id = tok_dir or "bundled"

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
        raise RuntimeError("VoiceDesign only supports 1.7B models!")

    # Cache key includes attention implementation, the resolved model folder,
    # quantization, and speech tokenizer identity.
    cache_key = (model_type, model_choice, device, precision, quantization, attn_impl, custom_dir, tok_cache_id)
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
        print(f"[Qwen3-TTS] Using custom model folder: {custom_dir}")
        found_local = custom_dir

    if not found_local:
        for base in base_paths:
            try:
                if not os.path.isdir(base):
                    continue
                subdirs = os.listdir(base)
                for d in subdirs:
                    cand = os.path.join(base, d)
                    if os.path.isdir(cand):
                        # Match logic: contains model size and type keyword
                        if model_choice in d and model_type.lower() in d.lower():
                            found_local = cand
                            break
                if found_local:
                    break
            except Exception:
                pass

    if found_local:
        final_source = found_local
        print(f"[Qwen3-TTS] Loading local model: {os.path.basename(final_source)}")
    else:
        # Try to download the specific model if not found locally
        downloaded_path = download_model_if_needed(final_source, qwen_root)
        if downloaded_path:
            final_source = downloaded_path
            print(f"[Qwen3-TTS] Loading downloaded model: {os.path.basename(final_source)}")
        else:
            raise RuntimeError(
                f"[Qwen3-TTS] Model '{final_source}' not found locally and auto-download failed. "
                f"Download it manually into {qwen_root} or check network/HF access."
            )

    if Qwen3TTSModel is None:
        raise RuntimeError(
            "[Qwen3-TTS] Model class is not loaded because the 'qwen_tts' package failed to import. "
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
            model_name = os.path.basename(final_source)
            int8_convrot_path = os.path.join(qwen_root, model_name + "_int8_convrot.safetensors")
            if os.path.exists(int8_convrot_path):
                print(f"[Qwen3-TTS] int8-convrot: loading pre-quantized weights from disk: {int8_convrot_path}")
                model = load_int8_convrot_model(
                    final_source, int8_convrot_path, device=device, dtype=dtype,
                    attn_implementation=attn_param,
                    load_speech_tokenizer=True,
                    speech_tokenizer_dir=tok_dir,
                )
                convrot_loaded = True
            else:
                print(f"[Qwen3-TTS] int8-convrot: no saved weights found at {int8_convrot_path}; running first-time conversion...")
                model = load_int8_convrot_model(
                    final_source, int8_convrot_path, device=device, dtype=dtype,
                    attn_implementation=attn_param,
                    load_speech_tokenizer=True,
                    auto_convert=True,
                    speech_tokenizer_dir=tok_dir,
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
        print(f"[Qwen3-TTS] Loading model with attention: {attn_impl}")
        model = Qwen3TTSModel.from_pretrained(final_source, **load_kwargs)
        # Speech tokenizer lives on the inner HF model (wrap.model), not the wrapper
        _inner = getattr(model, 'model', None)
        _st = getattr(_inner, 'speech_tokenizer', None) if _inner is not None else None
        _move_and_cast_codec(_st, device=device, dtype=dtype, label="codec")
        _apply_speech_tokenizer_override(model, speech_tokenizer, qwen_root, device, dtype)

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
