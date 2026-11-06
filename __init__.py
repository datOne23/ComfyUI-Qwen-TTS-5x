# ComfyUI-Qwen-TTS Custom Nodes
# Based on the open-source Qwen3-TTS project by Alibaba Qwen team

import os
import sys
import torch

# Add current directory to path for qwen_tts package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import nodes
from .nodes import (
    VoiceDesignNode,
    VoiceCloneNode,
    CustomVoiceNode,
    VoiceClonePromptNode,
    RoleBankNode,
    DialogueInferenceNode,
    SaveVoiceNode,
    LoadSpeakerNode,
)
from .train import Qwen3TTS_Train_Node

# Fast-mode (CUDA-graph) nodes - separate module, existing FB_Qwen3TTS* entries
# below are left untouched.
from .fast_nodes import (  # noqa: E402
    Qwen3TTSFastLoader,
    Qwen3TTSFastCustomVoice,
    Qwen3TTSFastVoiceDesign,
    Qwen3TTSFastVoiceClone,
    Qwen3TTSFastVoiceCloneFromFile,
    Qwen3TTSFastCreateClonePrompt,
    Qwen3TTSFastCloneWithPrompt,
    Qwen3TTSFastBatchGenerate,
    Qwen3TTSFastUnload,
)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "FB_Qwen3TTSVoiceClone": VoiceCloneNode,
    "FB_Qwen3TTSVoiceDesign": VoiceDesignNode,
    "FB_Qwen3TTSCustomVoice": CustomVoiceNode,
    "FB_Qwen3TTSVoiceClonePrompt": VoiceClonePromptNode,
    "FB_Qwen3TTSRoleBank": RoleBankNode,
    "FB_Qwen3TTSDialogueInference": DialogueInferenceNode,
    "FB_Qwen3TTSSaveVoice": SaveVoiceNode,
    "FB_Qwen3TTSLoadSpeaker": LoadSpeakerNode,
    "FB_Qwen3TTSTrain": Qwen3TTS_Train_Node,
    "Qwen3TTSFastLoader": Qwen3TTSFastLoader,
    "Qwen3TTSFastCustomVoice": Qwen3TTSFastCustomVoice,
    "Qwen3TTSFastVoiceDesign": Qwen3TTSFastVoiceDesign,
    "Qwen3TTSFastVoiceClone": Qwen3TTSFastVoiceClone,
    "Qwen3TTSFastVoiceCloneFromFile": Qwen3TTSFastVoiceCloneFromFile,
    "Qwen3TTSFastCreateClonePrompt": Qwen3TTSFastCreateClonePrompt,
    "Qwen3TTSFastCloneWithPrompt": Qwen3TTSFastCloneWithPrompt,
    "Qwen3TTSFastBatchGenerate": Qwen3TTSFastBatchGenerate,
    "Qwen3TTSFastUnload": Qwen3TTSFastUnload,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FB_Qwen3TTSVoiceClone": "🎭 Qwen3-TTS VoiceClone",
    "FB_Qwen3TTSVoiceDesign": "🎨 Qwen3-TTS VoiceDesign",
    "FB_Qwen3TTSCustomVoice": "🎵 Qwen3-TTS CustomVoice",
    "FB_Qwen3TTSVoiceClonePrompt": "🎭 Qwen3-TTS VoiceClonePrompt",
    "FB_Qwen3TTSRoleBank": "📇 Qwen3-TTS RoleBank",
    "FB_Qwen3TTSDialogueInference": "💬 Qwen3-TTS DialogueInference",
    "FB_Qwen3TTSSaveVoice": "💾 Qwen3-TTS SaveVoice",
    "FB_Qwen3TTSLoadSpeaker": "🎙️ Qwen3-TTS LoadSpeaker",
    "FB_Qwen3TTSTrain": "🏋️ Qwen3-TTS Train",
    "Qwen3TTSFastLoader": "Qwen3 TTS Fast Loader",
    "Qwen3TTSFastCustomVoice": "Qwen3 TTS Custom Voice (Fast)",
    "Qwen3TTSFastVoiceDesign": "Qwen3 TTS Voice Design (Fast)",
    "Qwen3TTSFastVoiceClone": "Qwen3 TTS Voice Clone (Fast)",
    "Qwen3TTSFastVoiceCloneFromFile": "Qwen3 TTS Voice Clone File (Fast)",
    "Qwen3TTSFastCreateClonePrompt": "Qwen3 TTS Create Clone Prompt (Fast)",
    "Qwen3TTSFastCloneWithPrompt": "Qwen3 TTS Clone with Prompt (Fast)",
    "Qwen3TTSFastBatchGenerate": "Qwen3 TTS Batch Generate (Fast)",
    "Qwen3TTSFastUnload": "Qwen3 TTS Fast Unload",
}

# Version information
__version__ = "1.0.8"
__author__ = "jesse890423"
__title__ = "ComfyUI-Qwen-TTS-5x"
__description__ = "ComfyUI custom nodes for Qwen3-TTS, patched for transformers 5.x compatibility (verified on 5.14.1)"
__license__ = "MIT"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "__author__",
    "__title__",
    "__description__",
    "__license__",
]

print(f"✅ ComfyUI-Qwen-TTS-5x v{__version__} loaded")
