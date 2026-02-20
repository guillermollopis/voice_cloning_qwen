# Voice Cloning with Qwen3-TTS

Voice cloning tool using the Qwen3-TTS model. Clone a voice from a reference audio file and generate new speech in that cloned voice.

## Features

- Clone any voice from a reference audio sample
- Automatic transcription of reference audio using OpenAI Whisper
- Support for Spanish and multilingual speech generation
- GPU-accelerated inference with optional FlashAttention2 support

## Setup

```bash
bash setup.sh
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

### Basic voice cloning

```bash
python voice_clone.py --reference sample_voice.wav --text "Hello, this is a test." --output output.wav
```

### Generate Spanish speech

```bash
python generate_spanish.py
```

### Example usage script

```bash
python example_usage.py
```

## Files

- `voice_clone.py` - Main voice cloning script
- `generate_spanish.py` - Spanish speech generation script
- `example_usage.py` - Example usage and demonstrations
- `setup.sh` - Environment setup script
- `requirements.txt` - Python dependencies
- `sample_voice.wav` / `sample_voice.mp3` - Sample reference voice files
- `reference_transcript.txt` - Reference audio transcript

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Dependencies listed in `requirements.txt`
