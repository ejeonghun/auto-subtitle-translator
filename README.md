# 자막 번역기 (Subtitle Translator)

비디오/음성 파일에서 자동으로 자막을 생성하고 번역하는 Python 프로그램입니다.

## ✨ 주요 기능

- 🎬 **비디오/음성 파일 지원**: MP4, AVI, MOV, WAV, MP3 등 다양한 형식 지원
- 🎤 **보컬 분리**: UVR5의 MDX-NET Kim vocal1 모델을 사용하여 배경음악과 음성 분리
- 🗣️ **음성 인식**: Faster Whisper 또는 Seamless M4T를 이용한 고품질 음성-텍스트 변환
- 🌍 **자막 번역**: Gemini API를 사용한 자연스러운 번역
- 📝 **SRT 자막 생성**: 표준 SRT 형식으로 자막 파일 생성

## 🔄 처리 파이프라인

```
비디오/음성 파일 → 보컬 분리 → 음성 인식 → SRT 생성 → 번역 → 최종 자막
```

## 📋 필요 조건

- Python 3.8 이상
- CUDA (GPU 가속 사용시, 선택사항)
- Gemini API 키

## 🚀 설치 방법

### 1. 저장소 클론 또는 파일 다운로드

```bash
# Git으로 클론하는 경우
git clone <repository-url>
cd 자막번역기

# 또는 파일들을 직접 다운로드하여 폴더에 저장
```

### 2. 가상환경 생성 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. Gemini API 키 설정

[Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키를 발급받으세요.

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

## 📖 사용법

### 기본 사용법

```bash
python main.py input_video.mp4 -k YOUR_GEMINI_API_KEY
```

### 고급 옵션

```bash
python main.py input_video.mp4 \
  -k YOUR_API_KEY \
  -l English \
  -e seamless_m4t \
  -p "뉴스 프로그램" \
  -s en \
  --keep-intermediate
```

### 모델 미리 다운로드

처음 실행하기 전에 필요한 모델들을 미리 다운로드할 수 있습니다:

```bash
python main.py --download-models -k YOUR_API_KEY
```

## 🎛️ 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-k, --api-key` | Gemini API 키 | 환경변수에서 읽음 |
| `-o, --output-dir` | 출력 디렉토리 | `output` |
| `-l, --target-language` | 번역 대상 언어 | `Korean` |
| `-s, --source-language` | 원본 언어 코드 (예: en, ja, ko) | 자동 감지 |
| `-e, --stt-engine` | 음성 인식 엔진 (`faster_whisper` 또는 `seamless_m4t`) | `faster_whisper` |
| `-m, --model-size` | 음성 인식 모델 크기 | `large-v3` |
| `-d, --device` | 사용할 디바이스 (`cpu`, `cuda`, `auto`) | `auto` |
| `-b, --batch-size` | 번역 배치 크기 | `50` |
| `-p, --program-name` | 프로그램명 (번역 품질 향상) | - |
| `--keep-intermediate` | 중간 파일들 유지 | `False` |
| `--download-models` | 모델만 다운로드하고 종료 | `False` |

## 📁 프로젝트 구조

```
자막번역기/
├── main.py              # 메인 프로그램
├── audio_processor.py   # 오디오/비디오 처리 및 보컬 분리
├── speech_to_text.py    # 음성 인식 모듈
├── translator.py        # 번역 모듈
├── srt_translator.py    # (기존 단독 번역기)
├── requirements.txt     # 필요한 패키지 목록
├── models/             # 다운로드된 모델 저장
└── output/             # 생성된 파일들 저장
```

## 🔧 모듈별 독립 사용

각 모듈은 독립적으로 사용할 수 있습니다:

### 오디오 처리 모듈

```python
from audio_processor import AudioProcessor

processor = AudioProcessor()
vocal_file = processor.process_file("input_video.mp4")
```

### 음성 인식 모듈

```python
from speech_to_text import SpeechToText

stt = SpeechToText(engine="faster_whisper")
success = stt.transcribe_to_srt("vocal.wav", "output.srt")
```

### 번역 모듈

```python
from translator import SubtitleTranslator

translator = SubtitleTranslator("YOUR_API_KEY", "Korean")
success = translator.translate_srt_file("input.srt", "translated.srt")
```

## 🎯 지원하는 파일 형식

### 입력 파일
- **비디오**: MP4, AVI, MOV, MKV, WMV, FLV
- **오디오**: WAV, MP3, FLAC, AAC, OGG

### 출력 파일
- **자막**: SRT (SubRip Subtitle)
- **오디오**: WAV (중간 파일)

## 🌍 지원 언어

### 음성 인식
- 자동 언어 감지 (Faster Whisper)
- 100+ 언어 지원

### 번역
- Gemini API가 지원하는 모든 언어
- 한국어, 영어, 일본어, 중국어, 스페인어, 프랑스어 등

## ⚡ 성능 최적화

### GPU 가속
- CUDA가 설치된 환경에서 자동으로 GPU 가속 사용
- CPU만으로도 동작 가능

### 메모리 사용량
- 배치 처리로 메모리 사용량 최적화
- 대용량 파일도 안정적으로 처리

## 🔍 문제 해결

### 일반적인 오류

1. **모듈 import 오류**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA 오류**
   ```bash
   # CPU 모드로 강제 실행
   python main.py input.mp4 -d cpu
   ```

3. **메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python main.py input.mp4 -b 25
   ```

4. **API 키 오류**
   - Gemini API 키가 올바른지 확인
   - API 사용량 제한 확인

### 로그 확인
프로그램 실행 시 자세한 로그가 출력되므로 오류 위치를 쉽게 파악할 수 있습니다.

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🤝 기여하기

버그 리포트나 기능 제안은 Issues에 등록해주세요.

## 📞 지원

문제가 발생하면 다음을 확인해보세요:
1. Python 버전 (3.8 이상)
2. 패키지 설치 상태
3. API 키 설정
4. 입력 파일 형식

---

**Made with ❤️ by AI Assistant**