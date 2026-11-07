"""Pure text/render helpers, UI-agnostic."""

import re

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union

from .constants import EMOTION_MAP


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
    sr = None
    pending_pause = 0.0
    for seg_type, content in segments:
        if seg_type == "pause":
            if content and float(content) > 0:
                pending_pause += float(content)
        else:
            wav, seg_sr = render_one(content)
            seg_sr = int(seg_sr) if seg_sr else int(default_sr)
            if sr is None:
                sr = seg_sr
            elif seg_sr != sr:
                raise RuntimeError(
                    f"Speech tokenizer sample rate changed during render ({sr} -> {seg_sr}). "
                    f"All segments must use the same output sample rate."
                )
            if pending_pause > 0:
                audio_results.append(np.zeros(int(round(pending_pause * sr)), dtype=np.float32))
                pending_pause = 0.0
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32)
            wav = np.squeeze(wav)
            while wav.ndim > 1:
                wav = np.mean(wav, axis=0)
            audio_results.append(np.ravel(wav))
    if sr is None:
        sr = int(default_sr)
    if pending_pause > 0 and audio_results:
        audio_results.append(np.zeros(int(round(pending_pause * sr)), dtype=np.float32))
    if not audio_results:
        raise RuntimeError("No audio generated from inline tag segments.")
    full = np.concatenate(audio_results)
    waveform = torch.from_numpy(full).float().unsqueeze(0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sr)}
