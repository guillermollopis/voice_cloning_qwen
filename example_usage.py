#!/usr/bin/env python3
"""
Example: Voice Cloning with Qwen3-TTS

Simple example demonstrating how to clone your reference voice
and generate new speech.
"""

from voice_clone import clone_voice, transcribe_audio
import os

# Path to your reference audio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO = os.path.join(SCRIPT_DIR, "sample_voice.mp3")

# If you know the transcript of your reference audio, set it here
# This improves cloning quality. Set to None for auto-transcription.
REF_TRANSCRIPT = None  # e.g., "Hello, this is a sample of my voice."

# Text you want to synthesize in the cloned voice
TEXT_TO_SPEAK = "Hello! This is my cloned voice speaking. I can say anything you want me to say now."


def main():
    print("=" * 60)
    print("Qwen3-TTS Voice Cloning Example")
    print("=" * 60)

    # Option 1: Quick single generation
    print("\n[1] Generating single cloned speech...")
    clone_voice(
        text=TEXT_TO_SPEAK,
        ref_audio=REF_AUDIO,
        ref_text=REF_TRANSCRIPT,
        output_path=os.path.join(SCRIPT_DIR, "output_example1.wav"),
        language="English",
        model_size="1.7B",
    )

    # Option 2: Generate multiple outputs with same voice
    print("\n[2] Generating multiple outputs with the same cloned voice...")

    sentences = [
        "The weather today is absolutely beautiful.",
        "I love programming and building new things.",
        "Voice cloning technology is truly amazing.",
    ]

    # Pre-transcribe once if needed
    if REF_TRANSCRIPT is None:
        ref_text = transcribe_audio(REF_AUDIO)
    else:
        ref_text = REF_TRANSCRIPT

    for i, sentence in enumerate(sentences, 1):
        output_file = os.path.join(SCRIPT_DIR, f"output_sentence_{i}.wav")
        print(f"\nGenerating: '{sentence}'")
        clone_voice(
            text=sentence,
            ref_audio=REF_AUDIO,
            ref_text=ref_text,  # Reuse transcription
            output_path=output_file,
            language="English",
            model_size="1.7B",
        )

    print("\n" + "=" * 60)
    print("Done! Check the generated .wav files in this directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
