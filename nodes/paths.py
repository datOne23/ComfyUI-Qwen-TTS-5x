"""Filesystem path + security-guard logic for the `nodes` package."""

import os
import folder_paths
import torch

from ._boot import VoiceClonePromptItem
from . import constants

# Register "qwen-tts" model folder for extra_model_paths.yaml support
try:
    folder_paths.add_model_folder_path("qwen-tts", os.path.join(folder_paths.models_dir, "qwen-tts"))
except Exception:
    pass


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


def _register_safe_globals():
    """Allow VoiceClonePromptItem to be unpickled by torch.load(weights_only=True)."""
    if constants._SAFE_GLOBALS_REGISTERED:
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
    constants._SAFE_GLOBALS_REGISTERED = True
