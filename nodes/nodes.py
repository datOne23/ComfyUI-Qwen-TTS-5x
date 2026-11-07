import os
import re
import hashlib
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from .constants import DEMO_LANGUAGES, LANGUAGE_MAP, EMOTION_MAP, ATTENTION_OPTIONS, _MODEL_CACHE
from .model import load_qwen_model, unload_cached_model
from .paths import _voices_dir, _safe_filename, _resolve_within, _register_safe_globals
from .progress import TTSProgress, _emit_progress
from .text import (
    split_text_by_pauses,
    parse_text_with_inline_tags,
    extract_emotion_instruct,
    _render_segments,
)


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
                "quantization": (["none", "int8-convrot"], {"default": "none"}),
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

    @classmethod
    def IS_CHANGED(cls, text, instruct, model_choice, device, precision,
                   quantization, language, seed, max_new_tokens, top_p,
                   top_k, temperature, repetition_penalty, attention,
                   unload_model_after_generate=False, **kwargs):
        return hashlib.sha256(
            f"{text}|{instruct}|{model_choice}|{device}|{precision}|"
            f"{quantization}|{language}|{seed}|{max_new_tokens}|{top_p}|"
            f"{top_k}|{temperature}|{repetition_penalty}|{attention}".encode()
        ).hexdigest()

    def generate(self, text: str, instruct: str, model_choice: str, device: str, precision: str, language: str, quantization: str = "none", seed: int = 0, max_new_tokens: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        print(f"[Qwen3TTS] generate called: seed={seed}, model_choice={model_choice}")
        if not text or not instruct:
            raise RuntimeError("Text and instruction description are required")

        pbar = TTSProgress(max_new_tokens, text=text)

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
                "quantization": (["none", "int8-convrot"], {"default": "none"}),
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

    @classmethod
    def IS_CHANGED(cls, target_text, model_choice, device, precision,
                   language, quantization, ref_text, seed, max_new_tokens,
                   top_p, top_k, temperature, repetition_penalty,
                   x_vector_only, attention, custom_model_path, **kwargs):
        return hashlib.sha256(
            f"{target_text}|{model_choice}|{device}|{precision}|"
            f"{language}|{quantization}|{ref_text}|{seed}|{max_new_tokens}|"
            f"{top_p}|{top_k}|{temperature}|{repetition_penalty}|"
            f"{x_vector_only}|{attention}|{custom_model_path}".encode()
        ).hexdigest()

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
                "quantization": (["none", "int8-convrot"], {"default": "none"}),
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

    @classmethod
    def IS_CHANGED(cls, text, speaker, model_choice, device, precision,
                   language, quantization, seed, instruct, max_new_tokens,
                   top_p, top_k, temperature, repetition_penalty,
                   attention, custom_model_path, custom_speaker_name,
                   **kwargs):
        return hashlib.sha256(
            f"{text}|{speaker}|{model_choice}|{device}|{precision}|"
            f"{language}|{quantization}|{seed}|{instruct}|{max_new_tokens}|"
            f"{top_p}|{top_k}|{temperature}|{repetition_penalty}|"
            f"{attention}|{custom_model_path}|{custom_speaker_name}".encode()
        ).hexdigest()

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
                "quantization": (["none", "int8-convrot"], {"default": "none"}),
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
                "quantization": (["none", "int8-convrot"], {"default": "none"}),
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

    @classmethod
    def IS_CHANGED(cls, script, model_choice, device, precision, language,
                   pause_linebreak, period_pause, comma_pause, question_pause,
                   hyphen_pause, merge_outputs, batch_size, quantization,
                   seed, max_new_tokens_per_line, top_p, top_k, temperature,
                   repetition_penalty, attention, **kwargs):
        return hashlib.sha256(
            f"{script}|{model_choice}|{device}|{precision}|{language}|"
            f"{pause_linebreak}|{period_pause}|{comma_pause}|{question_pause}|"
            f"{hyphen_pause}|{merge_outputs}|{batch_size}|{quantization}|"
            f"{seed}|{max_new_tokens_per_line}|{top_p}|{top_k}|{temperature}|"
            f"{repetition_penalty}|{attention}".encode()
        ).hexdigest()

    def generate_dialogue(self, script: str, role_bank: Dict[str, Any], model_choice: str, device: str, precision: str, language: str, pause_linebreak: float, period_pause: float, comma_pause: float, question_pause: float, hyphen_pause: float, merge_outputs: bool, batch_size: int, quantization: str = "none", seed: int = 0, max_new_tokens_per_line: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not script or not role_bank:
            raise RuntimeError("Script and Role Bank are required")

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


class UnloadModelNode:
    """
    UnloadModel: Manually unload the cached Qwen3-TTS model from VRAM.

    Use this node to free GPU memory after generation when
    unload_model_after_generate is disabled, or to switch models without
    reloading ComfyUI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_all": ("BOOLEAN", {"default": True, "tooltip": "Unload all cached models from VRAM."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "unload"
    OUTPUT_NODE = True
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "Unload cached Qwen3-TTS model(s) from VRAM to free GPU memory."

    def unload(self, unload_all: bool = True):
        if unload_all:
            unload_cached_model()
        else:
            import torch
            cuda_keys = [k for k in list(_MODEL_CACHE.keys()) if k[2] == "cuda"]
            for key in cuda_keys:
                unload_cached_model(key)
        return {}
