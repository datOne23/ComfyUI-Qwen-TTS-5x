"""Qwen3-TTS custom nodes package — drop-in replacement for the old `nodes.py` module.

Import order matters: `_boot` is imported first so HF_HUB_OFFLINE /
TRANSFORMERS_OFFLINE are set before *any* transformers/qwen_tts import, and the
shared PACKAGE_ROOT / COMFY_ROOT / MODELS_DIR constants are available to every
submodule.
"""

# _boot MUST be imported first: it sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
# before anything imports transformers or qwen_tts.
from . import _boot  # noqa: F401
from . import constants  # noqa: F401

from .int8_convrot import load_int8_convrot_model
from .paths import (
    _resolve_within,
    _voices_dir,
    _model_search_roots,
    _names_remote_machine,
    _custom_model_roots,
    _resolve_custom_model_dir,
    _safe_filename,
    _register_safe_globals,
)
from .text import (
    split_text_by_pauses,
    parse_text_with_inline_tags,
    extract_emotion_instruct,
    _render_segments,
)
from .progress import (
    _current_node_id,
    _emit_progress,
    _CONSOLE_BAR,
    _ConsoleBar,
    TTSProgress,
)
from .model import (
    unload_cached_model,
    apply_qwen3_patches,
    download_model_if_needed,
    check_and_download_tokenizer,
    load_qwen_model,
    _comfy_attention_available,
    check_attention_implementation,
    get_attention_implementation,
)
from .nodes import (
    VoiceDesignNode,
    VoiceCloneNode,
    CustomVoiceNode,
    VoiceClonePromptNode,
    RoleBankNode,
    DialogueInferenceNode,
    SaveVoiceNode,
    LoadSpeakerNode,
    UnloadModelNode,
)

# Backward-compatible re-exports (mirrors the public surface of the old nodes.py)
ALL_MODELS = constants.ALL_MODELS
MODEL_FAMILY_TO_HF = constants.MODEL_FAMILY_TO_HF
DEMO_LANGUAGES = constants.DEMO_LANGUAGES
LANGUAGE_MAP = constants.LANGUAGE_MAP
EMOTION_MAP = constants.EMOTION_MAP
ATTENTION_OPTIONS = constants.ATTENTION_OPTIONS

__all__ = [
    "VoiceDesignNode",
    "VoiceCloneNode",
    "CustomVoiceNode",
    "VoiceClonePromptNode",
    "RoleBankNode",
    "DialogueInferenceNode",
    "SaveVoiceNode",
    "LoadSpeakerNode",
    "UnloadModelNode",
    "load_qwen_model",
    "unload_cached_model",
    "apply_qwen3_patches",
    "download_model_if_needed",
    "check_and_download_tokenizer",
    "split_text_by_pauses",
    "parse_text_with_inline_tags",
    "extract_emotion_instruct",
    "_render_segments",
    "get_attention_implementation",
    "check_attention_implementation",
    "TTSProgress",
    "ALL_MODELS",
    "MODEL_FAMILY_TO_HF",
    "DEMO_LANGUAGES",
    "LANGUAGE_MAP",
    "EMOTION_MAP",
    "ATTENTION_OPTIONS",
    "_voices_dir",
    "_safe_filename",
    "_resolve_within",
    "_register_safe_globals",
]
