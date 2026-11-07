import os
import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' is required. Please install it using: pip install huggingface_hub")
    print("错误: 需要安装 'huggingface_hub'。请使用 pip install huggingface_hub 安装")
    sys.exit(1)

# Default models (1.7B is the recommended default)
DEFAULT_MODELS = {
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "base_1_7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}

SMALL_MODELS = {
    "base_0_6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "custom_voice_0_6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}

OPTIONAL_MODELS = {
    "tokenizer_48k": "takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz",
}

def get_comfy_models_path():
    """Attempt to find the ComfyUI/models directory relative to this script."""
    # Assuming script is in ComfyUI/custom_nodes/ComfyUI-Qwen-TTS/
    current_path = Path(__file__).parent.resolve()
    
    # Check standard ComfyUI structure: ../../models
    potential_models = current_path.parent.parent / "models"
    if potential_models.exists() and potential_models.is_dir():
        return potential_models
    
    return None

def download_model(repo_id, target_root):
    folder_name = repo_id.split("/")[-1]
    target_path = target_root / folder_name
    
    print(f"\n🔹 Processing (处理中): {repo_id}")
    print(f"   Target (目标路径): {target_path}")
    
    if target_path.exists():
        print(f"   ✅ Target directory exists. Checking for updates... (目标目录已存在，检查更新...)")
    else:
        print(f"   📥 Downloading new model... (正在下载新模型...)")
        
    try:
        snapshot_download(repo_id=repo_id, local_dir=target_path)
        print(f"   ✅ Success (成功): {repo_id}")
    except Exception as e:
        print(f"   ❌ Failed to download (下载失败) {repo_id}: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Download Qwen-TTS models for ComfyUI.")
    parser.add_argument("--target", type=str, help="Specific target directory for models. Defaults to ComfyUI/models/qwen-tts if found, else ./models/qwen-tts")
    parser.add_argument("--small", action="store_true", help="Download 0.6B models instead of 1.7B (where available)")
    parser.add_argument("--all", action="store_true", help="Download ALL models (0.6B and 1.7B), plus optional tokenizer if --with-48k-tokenizer is set")
    parser.add_argument("--with-48k-tokenizer", action="store_true", help="Also download optional 48kHz speech tokenizer (takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz, third-party Apache-2.0). Not auto-downloaded by default.")
    parser.add_argument("--48k-tokenizer-only", action="store_true", help="Download only the optional 48kHz speech tokenizer")
    args = parser.parse_args()

    # Determine target directory
    if args.target:
        base_dir = Path(args.target)
    else:
        comfy_models = get_comfy_models_path()
        if comfy_models:
            print(f"[INFO] Detected ComfyUI models directory at: {comfy_models}")
            base_dir = comfy_models / "qwen-tts"
        else:
            print("[WARN] Could not detect ComfyUI models directory. using local './models/qwen-tts'")
            base_dir = Path(os.getcwd()) / "models" / "qwen-tts"

    # Create directory
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Error creating directory {base_dir}: {e}")
        sys.exit(1)

    print(f"[INFO] Models will be downloaded to: {base_dir}")

    # Build download list
    models_to_download = []
    if args.__dict__.get("48k_tokenizer_only"):
        models_to_download.append(OPTIONAL_MODELS["tokenizer_48k"])
    else:
        models_to_download.append(DEFAULT_MODELS["tokenizer"])
        if args.all:
            models_to_download.extend(DEFAULT_MODELS.values())
            models_to_download.extend(SMALL_MODELS.values())
        elif args.small:
            models_to_download.append(SMALL_MODELS["base_0_6b"])
            models_to_download.append(SMALL_MODELS["custom_voice_0_6b"])
            print("[INFO] Note: VoiceDesign model is only available in 1.7B. Downloading it to ensure full functionality.")
            models_to_download.append(DEFAULT_MODELS["voice_design"])
        else:
            # Default (1.7B)
            models_to_download.extend([
                 DEFAULT_MODELS["base_1_7b"],
                 DEFAULT_MODELS["voice_design"],
                 DEFAULT_MODELS["custom_voice"]
            ])

        if args.with_48k_tokenizer or args.all:
            print("[INFO] Including optional 48kHz tokenizer: takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz (third-party Apache-2.0 fine-tune; 24kHz input / 48kHz output).")
            models_to_download.append(OPTIONAL_MODELS["tokenizer_48k"])

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for model in models_to_download:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    models_to_download = unique

    print("[INFO] Starting download...")

    # Execute
    for model in models_to_download:
        download_model(model, base_dir)

    print("\n[OK] All downloads finished.")

if __name__ == "__main__":
    main()
