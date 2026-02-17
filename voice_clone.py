#!/usr/bin/env python3
"""
Voice Cloning using Qwen3-TTS

This script clones a voice from a reference audio file and generates
new speech in that cloned voice.
"""

import argparse
import os
from pathlib import Path

import torch
import soundfile as sf


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
    import whisper

    print(f"Transcribing reference audio: {audio_path}")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()
    print(f"Transcription: {transcript}")
    return transcript


def clone_voice(
    text: str,
    ref_audio: str,
    ref_text: str = None,
    output_path: str = "output_cloned.wav",
    language: str = "Auto",
    model_size: str = "1.7B",
    use_flash_attention: bool = True,
):
    """
    Clone a voice from reference audio and generate new speech.

    Args:
        text: The text to synthesize in the cloned voice
        ref_audio: Path to the reference audio file
        ref_text: Transcript of the reference audio (if None, will transcribe automatically)
        output_path: Where to save the output audio
        language: Target language (Auto, English, Chinese, Japanese, Korean, etc.)
        model_size: Model size - "1.7B" (better quality) or "0.6B" (faster)
        use_flash_attention: Whether to use FlashAttention2 (recommended for lower memory)
    """
    from qwen_tts import Qwen3TTSModel

    # Transcribe reference audio if transcript not provided
    if ref_text is None:
        ref_text = transcribe_audio(ref_audio)

    # Select model based on size
    model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base"
    print(f"Loading model: {model_name}")

    # Configure attention implementation
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"

    # Load model
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_impl,
    )

    print(f"Generating cloned voice for: '{text}'")
    print(f"Using reference transcript: '{ref_text}'")

    # Generate cloned voice
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    # Save output
    sf.write(output_path, wavs[0], sr)
    print(f"Saved cloned voice audio to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Clone a voice from reference audio using Qwen3-TTS"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Text to synthesize in the cloned voice"
    )
    parser.add_argument(
        "--ref-audio", "-r",
        type=str,
        default="sample_voice.mp3",
        help="Path to reference audio file (default: sample_voice.mp3)"
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of reference audio (auto-transcribed if not provided)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_cloned.wav",
        help="Output audio file path (default: output_cloned.wav)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="Auto",
        choices=["Auto", "English", "Chinese", "Japanese", "Korean",
                 "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
        help="Target language (default: Auto)"
    )
    parser.add_argument(
        "--model-size", "-m",
        type=str,
        default="1.7B",
        choices=["1.7B", "0.6B"],
        help="Model size: 1.7B (better quality) or 0.6B (faster)"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable FlashAttention2 (use if not installed)"
    )

    args = parser.parse_args()

    # Resolve reference audio path
    ref_audio_path = args.ref_audio
    if not os.path.isabs(ref_audio_path):
        ref_audio_path = os.path.join(os.path.dirname(__file__), ref_audio_path)

    if not os.path.exists(ref_audio_path):
        print(f"Error: Reference audio not found: {ref_audio_path}")
        return 1

    clone_voice(
        text=args.text,
        ref_audio=ref_audio_path,
        ref_text=args.ref_text,
        output_path=args.output,
        language=args.language,
        model_size=args.model_size,
        use_flash_attention=not args.no_flash_attention,
    )

    return 0


if __name__ == "__main__":
    exit(main())
