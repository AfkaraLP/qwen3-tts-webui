<p align="center">
  <img src="/static/qwentts.webp" alt="Qwen TTS WebUI" width="300">
</p>

# Qwen TTS WebUI ![icon](/static/qwentts.ico)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![uv](https://img.shields.io/badge/uv-enabled-green.svg)

Generate speech using AI with Qwen TTS - upload a reference audio and create new voice synthesis in a clean, modern web interface.

## Features

- **Voice Cloning** - Upload reference audio or record directly in browser
- **Multi-language Support** - Generate speech in multiple languages
- **Auto-transcription** - Automatic transcription using Whisper
- **Audio Management** - Organize and download generated audio files
- **YT-DLP** - Download reference audios directly from a youtube link
- **Custom Naming** - Rename reference audios for better organization

## Quick Start

### Prerequisites

- **CUDA-compatible GPU** with ~6GB VRAM recommended
- **Nix or python +3.10**
- **uv** package manager (recommended for reproducible environments)

### Installation & Setup

<details>
<summary>Recommended: Using Nix (Reproducible)</summary>

```bash
# Clone the repository
git clone https://github.com/AfkaraLP/qwen3-tts-webui.git
cd qwen3-tts-webui

# Enter development environment
nix develop
uv sync

# Start the server
uv run start_server.py
# Check localhost:8000/docs for api documentation
```

</details>

<details>
<summary>📦 Alternative: Using uv only</summary>

```bash
# Clone the repository
git clone https://github.com/AfkaraLP/qwen3-tts-webui.git
cd qwen3-tts-webui

# Install dependencies
uv sync

# Start the server
uv run start_server.py
```

</details>

### Access the Web UI

Open **http://localhost:8000** in your browser to access the web interface.

## 🎯 Usage Guide

### 1. Use Existing Reference
1. Select a previously uploaded reference audio from the dropdown
2. Optionally rename it using the rename field
3. Enter the text you want to generate
4. Select language and click "Clone Voice"

### 2. Upload New Reference
1. Choose your audio source:
   - **Upload File**: Select an audio file from your device
   - **Record Audio**: Record directly using your microphone
   - **YouTube**: Enter a YouTube URL to extract audio
2. Add a custom name for better organization
3. Enter text and select language
4. Click "Upload & Clone"

### 3. Manage Generated Audio
- View all your generated audios in the "Generated Audios" section
- Play audio directly in the browser
- Download files for offline use

## 🛠️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_CLONER_PORT` | `8000` | Port for the web server |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |

```bash
# Custom port example
VOICE_CLONER_PORT=3000 python start_server.py

# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python start_server.py
```

## 🤝 Contributing

We welcome contributions! Please ensure reproducibility by:

### Development Environment

Use **Nix** and **uv** for a consistent development environment:

```bash
# Set up development environment
nix develop
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** with clear commit messages
4. **Test thoroughly** including edge cases
5. **Submit a pull request** with detailed description

## 🙏️ Support Development

If you find this project useful, please consider supporting me:

<div align="center">
  <a href="https://ko-fi.com/afkaralp" target="_blank">
    <img src="https://storage.ko-fi.com/cdn/brandasset/kofi_button_stroke.png" alt="Support me on Ko-fi" height="36"/>
  </a>
</div>

Your support helps maintain and improve the project! 🚀

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE.txt) file for details.

## 🙋 FAQ

<details>
<summary>❓ Frequently Asked Questions</summary>

**Q: What audio formats are supported?**
A: Most common audio formats (MP3, WAV, M4A, etc.) are supported for upload.

**Q: How long should the reference audio be?**
A: 10-30 seconds is ideal. The system automatically trims to 60 seconds maximum.

**Q: Can I use this commercially?**
A: Please check the Qwen TTS model license and terms of use for commercial applications.

**Q: Why use Nix and uv?**
A: They provide **reproducible environments** - anyone can recreate the exact same development setup, ensuring consistent behavior across different machines.

</details>
