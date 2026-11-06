# Bootstrap module for the `nodes` package.
#
# Leaf module: imports only stdlib, torch, qwen_tts, transformers and comfy-kitchen.
# It imports nothing from the `nodes` package itself — this is what breaks the
# import cycle (model <- int8 <- model via Qwen3TTSModel).
#
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set here, before *any* transformers
# or qwen_tts import, so offline mode is in effect from the very first import.

import os
import sys

import torch

# Force offline mode for HuggingFace Hub *before* anything imports transformers.
# local_files_only=True alone is not sufficient on Windows (transformers #43502
# — from_pretrained still hits /api/models/.../revision/main).
os.environ.setdefault("HF_HUB_OFFLINE", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "false")

# --- Path computation -----------------------------------------------------
# This file lives one level deeper than the old nodes.py (.../nodes/_boot.py),
# so PACKAGE_ROOT takes two dirname levels from __file__.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(PACKAGE_ROOT)
MODELS_DIR = os.path.join(COMFY_ROOT, "models")

# sys.path inserts so 'qwen_tts' resolves as a package: the custom-node root
# and the qwen_tts subfolder, matching the old nodes.py behavior.
qwen_tts_dir = os.path.join(PACKAGE_ROOT, "qwen_tts")
for _path in (PACKAGE_ROOT, qwen_tts_dir):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- qwen_tts import ------------------------------------------------------
try:
    import qwen_tts
    Qwen3TTSModel = qwen_tts.Qwen3TTSModel
    VoiceClonePromptItem = qwen_tts.VoiceClonePromptItem
except ImportError:
    try:
        from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    except ImportError as _e:
        import traceback
        print(f"\n❌ [Qwen3-TTS] Critical Import Error: {_e}")
        if not os.path.exists(qwen_tts_dir):
            print(f"   Missing directory: {qwen_tts_dir}")
            print("   Please clone the repository with submodules or ensure 'qwen_tts' folder exists.")
        else:
            print("   Traceback for debugging:")
            traceback.print_exc()
            print("\n   Common fix: run 'pip install -r requirements.txt' in your ComfyUI environment.")
        Qwen3TTSModel = None
        VoiceClonePromptItem = None

# --- progress hook --------------------------------------------------------
try:
    from qwen_tts.core.models.modeling_qwen3_tts import set_progress_hook
except Exception:  # model layer unavailable; progress degrades to coarse phases
    set_progress_hook = None

# --- W8A8 fast-load availability probe ------------------------------------
try:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from transformers.modeling_utils import init_empty_weights
    _W8A8_FAST_LOAD_AVAILABLE = True
except Exception:
    Qwen3TTSConfig = None  # type: ignore[assignment]
    Qwen3TTSForConditionalGeneration = None  # type: ignore[assignment]
    init_empty_weights = None  # type: ignore[assignment]
    _W8A8_FAST_LOAD_AVAILABLE = False

# --- comfy-kitchen availability probe ---------------------------------------
_COMFY_KITCHEN_AVAILABLE = False
try:
    import comfy_kitchen  # noqa: F401
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight
    _COMFY_KITCHEN_AVAILABLE = True
except Exception:
    comfy_kitchen = None  # type: ignore[assignment]
    TensorWiseINT8Layout = None  # type: ignore[assignment]
    _build_hadamard = None  # type: ignore[assignment]
    _rotate_weight = None  # type: ignore[assignment]
