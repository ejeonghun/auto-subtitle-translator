# 🎵 Gemini Subtitle - AI Subtitle Extraction & Translation Tool

An advanced GUI application that automatically generates subtitles from video/audio files and translates them naturally using AI technology.

## ✨ Key Features

### 🖥️ **Intuitive GUI Interface**
- **Dual-tab structure**: Separate workflows for "Subtitle Extraction" and "Translation"
- **Multi-file processing**: Select and process multiple video/audio files in batch
- **Real-time progress monitoring**: Live visualization of FFmpeg, Kim_vocal, and Whisper processing stages
- **Real-time subtitle preview**: Watch subtitles being extracted in real-time
- **Cross-platform support**: Native styling for macOS, Windows, and Linux

### 🎬 **Powerful Media Processing**
- **Wide format support**: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, MP3, WAV, FLAC, AAC, OGG, M4A
- **Smart audio processing**: Automatic audio extraction for videos, direct processing for audio files
- **Vocal separation**: UVR5 Kim_vocal model for separating vocals from background music (optional)
- **High-quality speech recognition**: Accurate speech-to-text conversion using Faster Whisper models

### 🌍 **AI-Powered Translation**
- **Gemini API integration**: Natural translation using Google's latest generative AI
- **Streaming translation**: Real-time translation progress monitoring
- **Context-aware**: Enhanced translation quality with program/song title context
- **Song lyrics optimization**: Specialized translation that maintains independence of each subtitle line

## 🔄 Processing Pipeline

```
Video/Audio Files → Vocal Separation → Speech Recognition → SRT Generation → Translation → Final Subtitles
```

## 📋 Requirements

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)
- Gemini API key

## 🚀 Installation

### 1. Clone Repository or Download Files

```bash
# Clone with Git
git clone <repository-url>
cd gemini_subtitle

# Or download files directly to a folder
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Packages

```bash
pip install -r requirements.txt
```

### 4. Set Up Gemini API Key

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

## 🚀 Usage

### 🖥️ GUI Execution (Recommended)

```bash
python gui.py
```

### 📋 GUI Usage Guide

#### **1️⃣ Subtitle Extraction Tab**

1. **File Selection**
   - Use `Select Video/Audio Files` button for multi-file selection
   - Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, MP3, WAV, FLAC, AAC, OGG, M4A
   - Selected files are displayed in a list with color-coded processing status

2. **Configuration Settings**
   - **Output Directory**: Set location for result files
   - **Source Language**: Select original language (Auto Detect recommended)
   - **Whisper Model**: Speech recognition model size (large-v3, medium, small, base)
   - **Device**: Processing device (auto, cuda, mps, cpu)
   - **Skip vocal separation**: Skip vocal separation (for audio files or already clean audio)

3. **Real-time Monitoring**
   - **Progress bars**: Individual progress for FFmpeg extraction, Kim_vocal processing, Whisper loading/recognition
   - **Real-time subtitles**: Live preview of extracted subtitles
   - **Status display**: Current processing file and stage information

4. **Start Extraction**
   - Click `Start Extraction` button to begin processing
   - Multiple files are processed sequentially
   - Completion alert shows success/failure counts

#### **2️⃣ Translation Tab**

1. **API Configuration**
   - **Gemini API Key**: Enter API key (can use GEMINI_API_KEY environment variable)

2. **Program Information**
   - **Program/Song Title**: Context information for improved translation quality
   - **Additional Notes**: Extra description or context information

3. **Language Settings**
   - **Source Language**: Original language
   - **Target Language**: Language to translate to

4. **SRT File Selection**
   - **Auto-linking**: Automatically displays SRT files completed from Extraction tab
   - **Manual selection**: Use Browse button to select existing SRT files
   - **Multi-selection**: Translate multiple SRT files simultaneously

5. **Translation Execution**
   - Click `Start Translation` button to begin translation
   - **Real-time streaming**: Watch translation process in real-time
   - **Completion notification**: Alert when all translations are complete

### 💻 Command Line Usage (Advanced Users)

#### Basic Usage
```bash
python main.py input_video.mp4 -k YOUR_GEMINI_API_KEY
```

#### Advanced Options
```bash
python main.py input_video.mp4 \
  -k YOUR_API_KEY \
  -l Korean \
  -e faster_whisper \
  -p "RADWIMPS - Your Name" \
  -s ja \
  --keep-intermediate
```

#### Pre-download Models
```bash
python main.py --download-models -k YOUR_API_KEY
```

## 🎛️ Option Descriptions

| Option | Description | Default |
|--------|-------------|---------|
| `-k, --api-key` | Gemini API key | Read from environment variable |
| `-o, --output-dir` | Output directory | `output` |
| `-l, --target-language` | Target language for translation | `Korean` |
| `-s, --source-language` | Source language code (e.g., en, ja, ko) | Auto-detect |
| `-e, --stt-engine` | Speech recognition engine (`faster_whisper` or `seamless_m4t`) | `faster_whisper` |
| `-m, --model-size` | Speech recognition model size | `large-v3` |
| `-d, --device` | Device to use (`cpu`, `cuda`, `auto`) | `auto` |
| `-b, --batch-size` | Translation batch size | `50` |
| `-p, --program-name` | Program name (improves translation quality) | - |
| `--keep-intermediate` | Keep intermediate files | `False` |
| `--download-models` | Download models only and exit | `False` |

## 📁 Project Structure

```
gemini_subtitle/
├── gui.py               # 🖥️ GUI Interface
├── main.py              # 💻 Command-line Main Program
├── audio_processor.py   # 🎵 Audio/Video Processing and Vocal Separation
├── speech_to_text.py    # 🗣️ Speech Recognition Module (Faster Whisper)
├── translator.py        # 🌍 Translation Module (Gemini API)
├── srt_translator.py    # 📝 SRT-specific Translator
├── device_utils.py      # ⚙️ System Device Utilities
├── languages.json       # 🌐 Supported Language Settings
├── requirements.txt     # 📦 Required Package List
├── models/             # 🤖 Downloaded AI Model Storage
└── output/             # 📁 Generated Files Storage
```

## 🎨 GUI Features & Highlights

### 📊 **Subtitle Extraction Tab**
- ✅ **Multi-file selection**: Select multiple files via button or drag-and-drop
- ✅ **Real-time progress**: 4 independent progress bars for each processing stage
- ✅ **Real-time subtitle preview**: Watch subtitles being extracted live
- ✅ **File status management**: Color-coded status (pending/processing/completed/error)

### 🌍 **Translation Tab**
- ✅ **Streaming translation**: Real-time display of Gemini API translation process
- ✅ **Context input**: Program/song titles for enhanced translation quality
- ✅ **Auto file linking**: Automatic recognition of completed SRT files
- ✅ **Multi-translation**: Support for simultaneous translation of multiple SRT files

### 🎯 **Cross-Platform Optimization**
- 🍎 **macOS**: Native style application (MPS GPU support)
- 🪟 **Windows**: Windows Vista style application (CUDA support)
- 🐧 **Linux**: Fusion style application (CPU/CUDA support)
- 📱 **High DPI**: Sharp interface on all resolutions

## 🔧 Independent Module Usage

Each module can be used independently:

### Audio Processing Module

```python
from audio_processor import AudioProcessor

processor = AudioProcessor()
vocal_file = processor.process_file("input_video.mp4")
```

### Speech Recognition Module

```python
from speech_to_text import SpeechToText

stt = SpeechToText(engine="faster_whisper")
success = stt.transcribe_to_srt("vocal.wav", "output.srt")
```

### Translation Module

```python
from translator import SubtitleTranslator

translator = SubtitleTranslator("YOUR_API_KEY", "Korean")
success = translator.translate_srt_file("input.srt", "translated.srt")
```

## 🎯 Supported File Formats

### Input Files
- **Video**: MP4, AVI, MOV, MKV, WMV, FLV
- **Audio**: WAV, MP3, FLAC, AAC, OGG

### Output Files
- **Subtitles**: SRT (SubRip Subtitle)
- **Audio**: WAV (intermediate files)

## 🌍 Supported Languages

### Speech Recognition
- Automatic language detection (Faster Whisper)
- 100+ language support

### Translation
- All languages supported by Gemini API
- Korean, English, Japanese, Chinese, Spanish, French, etc.

## ⚡ Performance Optimization

### GPU Acceleration
- Automatic GPU acceleration when CUDA is installed
- CPU-only operation also supported

### Memory Usage
- Memory usage optimization through batch processing
- Stable processing of large files

## 🔍 Troubleshooting

### 🖥️ GUI Related Issues

1. **GUI fails to start**
   ```bash
   # Reinstall PyQt6
   pip uninstall PyQt6
   pip install PyQt6
   ```

2. **macOS style warning message**
   ```
   The style key 'macintosh' is deprecated. Please use 'macos' instead.
   ```
   → This is just a warning and doesn't affect program operation

3. **High DPI display issues**
   → PyQt6 handles this automatically, no additional configuration needed

### ⚙️ Common Errors

1. **Module import errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA errors (when using GPU)**
   ```bash
   # Force CPU mode
   python gui.py  # Set Device to 'cpu' in GUI
   # Or from command line
   python main.py input.mp4 -d cpu
   ```

3. **Memory shortage**
   - Change Whisper Model to 'small' or 'base' in GUI
   - Or reduce batch size from command line:
   ```bash
   python main.py input.mp4 -b 25
   ```

4. **Gemini API related errors**
   - Verify API key is correct
   - Check API usage limits
   - Verify environment variable: `echo $GEMINI_API_KEY`

5. **Subtitles being translated too long**
   - Include keywords like "song", "music", "lyrics" in program information
   - Optimized prompts already applied to maintain independence of each subtitle line

### 🔧 Performance Optimization Tips

1. **Apple Silicon Mac users**
   - Set Device to 'mps' to utilize GPU acceleration
   - Kim_vocal model optimized with CoreML

2. **NVIDIA GPU users**
   - Set Device to 'cuda'
   - CUDA 11.8 or higher recommended

3. **Memory saving**
   - Check "Skip vocal separation" when vocal separation is not needed
   - Use smaller Whisper models (medium, small, base)

### 📊 Logging and Debugging

The GUI provides the following real-time information:
- **Progress bars**: Progress for each processing stage
- **Status messages**: Current processing tasks
- **Real-time subtitles**: Preview of extracted subtitles
- **Translation streaming**: Real-time translation process monitoring

Detailed logs are also output in the terminal for reference when issues occur.

## 📄 License

This project follows the MIT License.


### 🆕 **Complete GUI Overhaul**
- **Tabbed interface**: Separate workflows for extraction and translation
- **Multi-file processing**: Select and batch process multiple files
- **Real-time monitoring**: Live visualization of all processing stages
- **Cross-platform**: Native styling for macOS, Windows, Linux

### 🔧 **Enhanced Translation Quality**
- **Song lyrics optimization**: Specialized translation maintaining independence of each subtitle line
- **Context awareness**: Improved translation accuracy with program/song titles
- **Streaming translation**: Real-time translation process monitoring

### ⚡ **Performance and Usability Improvements**
- **Apple Silicon optimization**: MPS GPU acceleration and CoreML optimization
- **Memory efficiency**: Optimized large file processing
- **User-friendly**: Intuitive interface with detailed progress indicators

---

## 🌟 Project Highlights

This project combines **cutting-edge AI technology** with an **intuitive user interface** to create a subtitle extraction and translation tool that anyone can use easily.

- 🤖 **AI Power**: Gemini API, Faster Whisper, UVR5 Kim_vocal models
- 🎨 **Modern UI**: PyQt6-based cross-platform GUI
- 🚀 **High Performance**: GPU acceleration and batch processing optimization
- 🌍 **Global**: 100+ language support

**Perfect for**: YouTube creators, translators, educators, content creators