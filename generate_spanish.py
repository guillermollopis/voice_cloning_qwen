#!/usr/bin/env python3
"""
Spanish Voice Cloning - Ready to Use

This script is pre-configured with your reference audio and transcript.
Just provide the text you want to generate in Spanish.

Usage:
    python generate_spanish.py "Tu texto aquí en español"
    python generate_spanish.py "Tu texto aquí" --output mi_audio.wav
"""

import argparse
import os
import sys

import torch
import soundfile as sf

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO = os.path.join(SCRIPT_DIR, "sample_voice.wav")
REF_TRANSCRIPT_FILE = os.path.join(SCRIPT_DIR, "reference_transcript.txt")


def load_transcript():
    """Load the pre-generated transcript."""
    with open(REF_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def generate_voice(text: str, output_path: str = "output.wav", model_size: str = "1.7B"):
    """
    Generate speech in the cloned Spanish voice.

    Args:
        text: Spanish text to synthesize
        output_path: Where to save the output audio
        model_size: "1.7B" for better quality, "0.6B" for faster
    """
    from qwen_tts import Qwen3TTSModel

    print("Loading reference transcript...")
    ref_text = load_transcript()

    model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base"
    print(f"Loading model: {model_name}")
    print("(This may take a moment on first run to download the model)")

    # Check for GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Try flash attention, fall back to sdpa if not available
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        print("FlashAttention not available, using SDPA...")
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )

    print(f"\nGenerating: '{text}'")
    print(f"Language: Spanish")

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Spanish",
        ref_audio=REF_AUDIO,
        ref_text=ref_text,
    )

    # Resolve output path
    if not os.path.isabs(output_path):
        output_path = os.path.join(SCRIPT_DIR, output_path)

    sf.write(output_path, wavs[0], sr)
    print(f"\nSaved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Spanish speech with your cloned voice"
    )
    parser.add_argument(
        "text",
        type=str,
        help="Spanish text to synthesize"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output file path (default: output.wav)"
    )
    parser.add_argument(
        "--model-size", "-m",
        type=str,
        default="1.7B",
        choices=["1.7B", "0.6B"],
        help="Model size: 1.7B (better) or 0.6B (faster)"
    )

    args = parser.parse_args()

    if not os.path.exists(REF_AUDIO):
        print(f"Error: Reference audio not found: {REF_AUDIO}")
        return 1

    if not os.path.exists(REF_TRANSCRIPT_FILE):
        print(f"Error: Transcript not found: {REF_TRANSCRIPT_FILE}")
        return 1

    generate_voice(
        text=args.text,
        output_path=args.output,
        model_size=args.model_size,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
