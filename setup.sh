#!/bin/bash
# Setup script for Qwen3-TTS Voice Cloning

set -e

echo "========================================"
echo "Qwen3-TTS Voice Cloning Setup"
echo "========================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda create -n qwen3-tts python=3.12 -y
    echo ""
    echo "Activate the environment with:"
    echo "  conda activate qwen3-tts"
    echo ""
    echo "Then install dependencies with:"
    echo "  pip install -r requirements.txt"
else
    echo "Conda not found. Installing with pip directly..."
    echo ""
    pip install -r requirements.txt
fi

echo ""
echo "========================================"
echo "Optional: Install FlashAttention2 for lower GPU memory"
echo "========================================"
echo "Run this command if you have a compatible GPU:"
echo "  pip install -U flash-attn --no-build-isolation"
echo ""
echo "For systems with <96GB RAM:"
echo "  MAX_JOBS=4 pip install -U flash-attn --no-build-isolation"
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Usage examples:"
echo ""
echo "1. Command line:"
echo "   python voice_clone.py --text 'Hello world!' --ref-audio sample_voice.mp3"
echo ""
echo "2. Run the example script:"
echo "   python example_usage.py"
echo ""
echo "3. With custom transcript (recommended for better quality):"
echo "   python voice_clone.py --text 'Hello world!' --ref-audio sample_voice.mp3 --ref-text 'The actual words in your sample audio'"
echo ""
