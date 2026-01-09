# Deepgram + Fish Speech Voice Loop

A Python-based voice loop system combining Deepgram STT with Fish Speech TTS for offline-capable conversational AI. Designed as a foundation for edge use cases like wearables and accessibility applications.

## Features

- **Hybrid Online/Offline Mode**: Uses Deepgram for high-accuracy streaming STT online, falls back to local Whisper offline
- **Local TTS**: Fish Speech for fully offline text-to-speech with emotion and speed control
- **LLM Integration**: Support for OpenAI, Anthropic, and local LLMs (Ollama)
- **Voice Activity Detection**: Silero VAD for accurate speech detection
- **Low Latency**: Streaming architecture for responsive voice interactions
- **Interruption Handling**: User can interrupt assistant responses

## Architecture

```
Microphone → VAD → STT (Deepgram/Whisper) → LLM (optional) → TTS (Fish Speech) → Speaker
```

## Quick Start

### Installation

```bash
# Clone the repository
cd deepgram-fish-speech-workflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev,llm]"
```

### Environment Setup

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# DEEPGRAM_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here  # Optional, for LLM
```

### Run Examples

```bash
# Test audio I/O
python examples/audio_test.py --list-devices
python examples/audio_test.py --tone

# Basic voice loop (echo mode)
python examples/basic_loop.py

# Fully offline mode
python examples/basic_loop.py --offline

# Conversational AI with LLM
python examples/conversational_ai.py --llm openai
```

## Configuration

Configuration can be done via:
1. Environment variables (see `.env.example`)
2. YAML config files (see `configs/`)
3. Programmatic configuration

### Example: Load YAML Config

```python
from src.core.config import VoiceLoopConfig

config = VoiceLoopConfig.from_yaml("configs/low_latency.yaml")
```

### Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `mode` | online, offline, or hybrid | hybrid |
| `stt_provider` | deepgram or whisper | deepgram |
| `tts_provider` | fish_speech | fish_speech |
| `echo_mode` | Echo input without LLM | false |
| `streaming_mode` | Stream TTS for lower latency | true |
| `interruption_enabled` | Allow user to interrupt | true |

## Project Structure

```
deepgram-fish-speech-workflow/
├── src/
│   ├── core/           # Types, config, protocols
│   ├── stt/            # Speech-to-text providers
│   ├── tts/            # Text-to-speech providers
│   ├── llm/            # LLM providers
│   ├── audio/          # Audio I/O and VAD
│   └── pipeline/       # Voice loop orchestration
├── examples/           # Usage examples
├── configs/            # Configuration presets
└── tests/              # Test suite
```

## Provider Support

### STT (Speech-to-Text)
- **Deepgram Nova-2**: 95%+ accuracy, streaming, diarization (100+ languages)
- **Whisper (local)**: Offline fallback via faster-whisper

### TTS (Text-to-Speech)
- **Fish Speech**: Local inference, emotion control, duration control

### LLM (Language Model)
- **OpenAI**: GPT-4o, GPT-4o-mini
- **Anthropic**: Claude 3.5 Sonnet
- **Local**: Ollama (Llama 3, etc.)

## Use Cases

This foundation supports edge applications like:

- **Real-time Coaching**: Continuous listening with context-aware feedback
- **Accessibility/AAC**: Non-verbal cue synthesis, emotion detection
- **Voice Dubbing**: Audio-in to translated audio-out with timing preservation
- **Wearables**: Low-latency voice interfaces for smart devices

## Development

```bash
# Run tests
pytest

# Type checking
mypy src

# Linting
ruff check src
```

## License

MIT
