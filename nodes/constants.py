"""Shared data constants for the `nodes` package.

Pure data (and module-level mutable globals) moved verbatim from the old
nodes.py. Leaf module: no internal imports.
"""

_DICT_KEYS_TYPE = type({}.keys())

# Modules that must stay in bf16 (sensitive to quantized matmul casting).
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

# Speech tokenizer options (output sample rate)
SPEECH_TOKENIZER_CHOICES = ["24kHz (default)", "48kHz"]
SPEECH_TOKENIZER_FOLDERS = {
    "24kHz (default)": "Qwen3-TTS-Tokenizer-12Hz",
    "48kHz": "Qwen3-TTS-Tokenizer-12Hz-48kHz",
}
SPEECH_TOKENIZER_REPOS = {
    "24kHz (default)": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "48kHz": "takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz",
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
ALL_MODELS_WITH_48K = list(ALL_MODELS) + ["takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz"]

_MODELS_CHECKED = False
_MODEL_CACHE = {}
_SAFE_GLOBALS_REGISTERED = False

ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager", "xformers", "comfy_kitchen"]
