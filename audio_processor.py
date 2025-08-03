#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
오디오/비디오 처리 모듈
UVR5를 이용한 보컬 분리 및 오디오 전처리
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import requests
from tqdm import tqdm
import onnxruntime as ort
import shutil
import zipfile
import platform
import tempfile

# 디바이스 유틸리티 import
from device_utils import get_optimal_device, get_onnx_providers

class AudioProcessor:
    """오디오/비디오 파일 처리 및 보컬 분리 클래스"""
    
    def __init__(self, models_dir: str = "models", device: str = "auto"):
        """
        초기화 함수
        
        Args:
            models_dir: 모델 파일들이 저장될 디렉토리
            device: 사용할 디바이스 ("auto", "cuda", "mps", "cpu")
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # ffmpeg 설치 디렉토리
        self.ffmpeg_dir = self.models_dir / "ffmpeg"
        self.ffmpeg_dir.mkdir(exist_ok=True)
        
        # 디바이스 설정
        if device == "auto":
            self.device_type, self.device_info = get_optimal_device()
        else:
            self.device_type = device
            # 수동 설정된 디바이스 검증
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        print("⚠️ CUDA를 요청했지만 사용할 수 없습니다. CPU로 전환합니다.")
                        self.device_type = "cpu"
                except ImportError:
                    print("⚠️ PyTorch가 없어 CUDA를 확인할 수 없습니다. CPU로 전환합니다.")
                    self.device_type = "cpu"
            elif device == "mps":
                try:
                    import torch
                    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                        print("⚠️ MPS를 요청했지만 사용할 수 없습니다. CPU로 전환합니다.")
                        self.device_type = "cpu"
                except ImportError:
                    print("⚠️ PyTorch가 없어 MPS를 확인할 수 없습니다. CPU로 전환합니다.")
                    self.device_type = "cpu"
        
        # UVR5 모델 정보
        self.uvr_models = {
            "Kim_Vocal_1": {
                "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_1.onnx",
                "filename": "Kim_Vocal_1.onnx"
            }
        }
        
        self.session = None
        self.ffmpeg_path = None
        
        # ffmpeg 사용 가능 여부 확인 및 자동 설치
        self._setup_ffmpeg()
        print(f"🎵 AudioProcessor 초기화 완료 - 디바이스: {self.device_type.upper()}")
    
    def _get_ffmpeg_download_info(self) -> dict:
        """
        운영체제별 ffmpeg 다운로드 정보 반환
        
        Returns:
            다운로드 정보 딕셔너리
        """
        system = platform.system().lower()
        
        if system == "windows":
            return {
                "url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
                "extract_path": "ffmpeg-*-essentials_build/bin/ffmpeg.exe",
                "executable": "ffmpeg.exe"
            }
        elif system == "darwin":  # macOS
            return {
                "url": "https://evermeet.cx/ffmpeg/ffmpeg-6.0.zip",
                "extract_path": "ffmpeg",
                "executable": "ffmpeg"
            }
        else:  # Linux
            return {
                "url": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
                "extract_path": "ffmpeg-*-amd64-static/ffmpeg",
                "executable": "ffmpeg"
            }
    
    def _download_ffmpeg(self) -> bool:
        """
        ffmpeg 자동 다운로드 및 설치
        
        Returns:
            설치 성공 여부
        """
        try:
            download_info = self._get_ffmpeg_download_info()
            url = download_info["url"]
            executable = download_info["executable"]
            
            print(f"📥 ffmpeg 다운로드 중: {url}")
            
            # 임시 파일로 다운로드
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                temp_path = temp_file.name
                
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="ffmpeg 다운로드") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            pbar.update(len(chunk))
            
            print("📦 ffmpeg 압축 해제 중...")
            
            # 압축 해제
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(self.ffmpeg_dir)
            
            # ffmpeg 실행 파일 찾기
            ffmpeg_executable = None
            for root, dirs, files in os.walk(self.ffmpeg_dir):
                for file in files:
                    if file == executable:
                        ffmpeg_executable = Path(root) / file
                        break
                if ffmpeg_executable:
                    break
            
            if ffmpeg_executable and ffmpeg_executable.exists():
                # ffmpeg를 직접 접근 가능한 위치로 복사
                final_ffmpeg_path = self.ffmpeg_dir / executable
                shutil.copy2(ffmpeg_executable, final_ffmpeg_path)
                
                # 실행 권한 설정 (Linux/macOS)
                if platform.system() != "Windows":
                    os.chmod(final_ffmpeg_path, 0o755)
                
                self.ffmpeg_path = str(final_ffmpeg_path)
                print(f"✅ ffmpeg 설치 완료: {self.ffmpeg_path}")
                
                # 임시 파일 정리
                os.unlink(temp_path)
                
                return True
            else:
                print("❌ ffmpeg 실행 파일을 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ ffmpeg 다운로드 실패: {e}")
            return False
    
    def _check_ffmpeg(self) -> bool:
        """
        ffmpeg 설치 여부 확인
        
        Returns:
            ffmpeg 사용 가능 여부
        """
        # 1. 시스템 PATH에서 ffmpeg 확인
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ 시스템 ffmpeg 사용 가능")
                self.ffmpeg_path = "ffmpeg"
                self.has_ffmpeg = True
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # 2. 로컬 설치된 ffmpeg 확인
        local_ffmpeg = self.ffmpeg_dir / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
        if local_ffmpeg.exists():
            try:
                result = subprocess.run([str(local_ffmpeg), '-version'], 
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✅ 로컬 ffmpeg 사용 가능: {local_ffmpeg}")
                    self.ffmpeg_path = str(local_ffmpeg)
                    self.has_ffmpeg = True
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        print("⚠️ ffmpeg를 찾을 수 없습니다.")
        self.has_ffmpeg = False
        return False
    
    def _setup_ffmpeg(self) -> bool:
        """
        ffmpeg 설정 및 자동 설치
        
        Returns:
            설정 성공 여부
        """
        # 먼저 기존 설치 확인
        if self._check_ffmpeg():
            return True
        
        # 자동 설치 시도
        print("🔧 ffmpeg를 자동으로 설치합니다...")
        
        try:
            if self._download_ffmpeg():
                return self._check_ffmpeg()
            else:
                print("❌ ffmpeg 자동 설치 실패")
                print("수동 설치: https://ffmpeg.org/download.html")
                return False
        except Exception as e:
            print(f"❌ ffmpeg 설정 실패: {e}")
            return False
    
    def install_ffmpeg(self) -> bool:
        """
        ffmpeg 수동 설치 함수 (외부에서 호출 가능)
        
        Returns:
            설치 성공 여부
        """
        return self._setup_ffmpeg()

    def download_model(self, model_name: str = "Kim_Vocal_1") -> bool:
        """
        UVR5 모델 다운로드
        
        Args:
            model_name: 다운로드할 모델명
            
        Returns:
            다운로드 성공 여부
        """
        if model_name not in self.uvr_models:
            print(f"❌ 지원하지 않는 모델: {model_name}")
            return False
        
        model_info = self.uvr_models[model_name]
        model_path = self.models_dir / model_info["filename"]
        
        if model_path.exists():
            print(f"✅ 모델이 이미 존재합니다: {model_path}")
            return True
        
        try:
            print(f"📥 {model_name} 모델 다운로드 중...")
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="다운로드") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ 모델 다운로드 완료: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ 모델 다운로드 실패: {e}")
            if model_path.exists():
                model_path.unlink()
            return False
    
    def load_uvr_model(self, model_name: str = "Kim_Vocal_1") -> bool:
        """
        UVR5 모델 로드
        
        Args:
            model_name: 로드할 모델명
            
        Returns:
            로드 성공 여부
        """
        model_path = self.models_dir / self.uvr_models[model_name]["filename"]
        
        if not model_path.exists():
            print(f"❌ 모델 파일이 없습니다. 먼저 다운로드해주세요: {model_path}")
            return False
        
        try:
            # 디바이스에 맞는 ONNX 제공자 설정
            providers = get_onnx_providers(self.device_type)
            
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            print(f"✅ UVR5 모델 로드 완료: {model_name} ({self.device_type.upper()})")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def extract_audio_from_video(self, video_path: str, output_path: str) -> bool:
        """
        ffmpeg를 사용하여 비디오 파일에서 오디오 추출
        
        Args:
            video_path: 입력 비디오 파일 경로
            output_path: 출력 오디오 파일 경로
            
        Returns:
            추출 성공 여부
        """
        if not self.has_ffmpeg:
            print("❌ ffmpeg가 설치되지 않았습니다. 자동 설치를 시도합니다...")
            if not self.install_ffmpeg():
                return False
            
        try:
            print(f"🎬 ffmpeg로 비디오에서 오디오 추출 중: {video_path}")
            
            # ffmpeg 명령어: 비디오에서 오디오만 추출
            cmd = [
                self.ffmpeg_path,           # 설정된 ffmpeg 경로 사용
                '-i', video_path,           # 입력 파일
                '-vn',                      # 비디오 스트림 제거
                '-acodec', 'pcm_s16le',     # PCM 16bit 리틀엔디안으로 인코딩
                '-ar', '44100',             # 샘플링 레이트 44.1kHz
                '-ac', '2',                 # 스테레오 (2채널)
                '-y',                       # 기존 파일 덮어쓰기
                output_path
            ]
            
            # ffmpeg 실행 (진행률 표시를 위해 stderr 캡처)
            result = subprocess.run(cmd, 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=300)  # 5분 타임아웃
            
            if result.returncode == 0:
                if Path(output_path).exists():
                    print(f"✅ 오디오 추출 완료: {output_path}")
                    return True
                else:
                    print("❌ 출력 파일이 생성되지 않았습니다.")
                    return False
            else:
                print(f"❌ ffmpeg 오류: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ ffmpeg 처리 시간 초과 (5분)")
            return False
        except Exception as e:
            print(f"❌ 오디오 추출 실패: {e}")
            return False
    
    def extract_audio_fallback(self, video_path: str, output_path: str) -> bool:
        """
        ffmpeg 없을 때 대체 방법 (기존 moviepy 방식)
        
        Args:
            video_path: 입력 비디오 파일 경로
            output_path: 출력 오디오 파일 경로
            
        Returns:
            추출 성공 여부
        """
        try:
            from moviepy.editor import VideoFileClip
            print(f"🎬 moviepy로 비디오에서 오디오 추출 중: {video_path}")
            
            with VideoFileClip(video_path) as video:
                audio = video.audio
                audio.write_audiofile(output_path, verbose=False, logger=None)
                audio.close()
            
            print(f"✅ 오디오 추출 완료: {output_path}")
            return True
            
        except ImportError:
            print("❌ moviepy가 설치되지 않았습니다.")
            return False
        except Exception as e:
            print(f"❌ moviepy 오디오 추출 실패: {e}")
            return False

    def preprocess_audio(self, audio_path: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
        """
        오디오 전처리 (리샘플링, 정규화)
        
        Args:
            audio_path: 오디오 파일 경로
            target_sr: 목표 샘플링 레이트
            
        Returns:
            전처리된 오디오 데이터와 샘플링 레이트
        """
        try:
            print(f"🔧 오디오 전처리 중: {audio_path}")
            
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=False)
            
            # 스테레오로 변환 (모노인 경우)
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            # 정규화
            audio = audio / np.max(np.abs(audio))
            
            print(f"✅ 오디오 전처리 완료 - Shape: {audio.shape}, SR: {target_sr}")
            return audio, target_sr
            
        except Exception as e:
            print(f"❌ 오디오 전처리 실패: {e}")
            return None, 0
    
    def separate_vocals(self, audio_data: np.ndarray, chunk_size: int = 512000) -> Tuple[np.ndarray, np.ndarray]:
        """
        UVR5를 이용한 보컬 분리
        
        Args:
            audio_data: 입력 오디오 데이터 (2, N)
            chunk_size: 처리할 청크 크기
            
        Returns:
            (보컬, 악기) 오디오 데이터
        """
        if self.session is None:
            raise ValueError("UVR5 모델이 로드되지 않았습니다. load_uvr_model()을 먼저 호출해주세요.")
        
        try:
            print("🎤 보컬 분리 중...")
            
            # STFT 파라미터 (모델에 맞게 설정)
            n_fft = 6144  # 3072 * 2
            hop_length = 1024
            window_length = 6144
            
            vocals = []
            instruments = []
            
            # 스테레오 채널별로 처리
            for channel in range(audio_data.shape[0]):
                channel_audio = audio_data[channel]
                
                # STFT 변환
                stft = librosa.stft(
                    channel_audio,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=window_length,
                    window='hann'
                )
                
                # 실수부와 허수부 분리
                stft_real = np.real(stft)
                stft_imag = np.imag(stft)
                
                # 주파수 차원을 정확히 3072로 맞추기
                freq_bins, time_frames = stft_real.shape
                print(f"원본 STFT 크기: {freq_bins} x {time_frames}")
                
                if freq_bins > 3072:
                    # 3072개로 자르기
                    stft_real = stft_real[:3072, :]
                    stft_imag = stft_imag[:3072, :]
                elif freq_bins < 3072:
                    # 3072개로 패딩
                    pad_freq = 3072 - freq_bins
                    stft_real = np.pad(stft_real, ((0, pad_freq), (0, 0)), mode='constant')
                    stft_imag = np.pad(stft_imag, ((0, pad_freq), (0, 0)), mode='constant')
                
                freq_bins = 3072  # 이제 정확히 3072
                time_frames = stft_real.shape[1]
                
                # 청크 단위로 처리 (시간 축을 따라)
                chunk_frames = 256  # 모델이 기대하는 시간 프레임
                
                channel_vocals = []
                channel_instruments = []
                
                for t in tqdm(range(0, time_frames, chunk_frames), desc=f"채널 {channel+1} 보컬 분리"):
                    # 청크 추출
                    end_t = min(t + chunk_frames, time_frames)
                    actual_frames = end_t - t
                    
                    # 청크 크기를 정확히 256으로 맞추기
                    chunk_real = np.zeros((3072, 256), dtype=np.float32)
                    chunk_imag = np.zeros((3072, 256), dtype=np.float32)
                    
                    # 실제 데이터 복사
                    chunk_real[:, :actual_frames] = stft_real[:, t:end_t]
                    chunk_imag[:, :actual_frames] = stft_imag[:, t:end_t]
                    
                    # 4채널 입력 준비 (Left와 Right가 같다고 가정)
                    model_input = np.stack([
                        chunk_real,  # Real Left
                        chunk_imag,  # Imag Left  
                        chunk_real,  # Real Right (same as Left for mono-like processing)
                        chunk_imag   # Imag Right (same as Left for mono-like processing)
                    ], axis=0)  # [4, 3072, 256]
                    
                    # 배치 차원 추가
                    model_input = np.expand_dims(model_input, axis=0)  # [1, 4, 3072, 256]
                    
                    # ONNX 모델 실행
                    input_name = self.session.get_inputs()[0].name
                    outputs = self.session.run(None, {input_name: model_input})
                    
                    # 출력 처리 (보컬 스펙트로그램)
                    vocal_spec = outputs[0][0]  # [4, 3072, 256]
                    
                    # 실수부와 허수부 복원
                    vocal_real = vocal_spec[0]  # 첫 번째 채널의 실수부
                    vocal_imag = vocal_spec[1]  # 첫 번째 채널의 허수부
                    
                    # 실제 프레임 크기로 자르기
                    vocal_real_chunk = vocal_real[:, :actual_frames]
                    vocal_imag_chunk = vocal_imag[:, :actual_frames]
                    
                    # 원본 청크도 실제 크기로 자르기
                    original_real_chunk = chunk_real[:, :actual_frames]
                    original_imag_chunk = chunk_imag[:, :actual_frames]
                    
                    # 복소수로 변환
                    vocal_chunk = vocal_real_chunk + 1j * vocal_imag_chunk
                    original_chunk = original_real_chunk + 1j * original_imag_chunk
                    
                    # 악기는 원본에서 보컬을 뺀 것
                    instrument_chunk = original_chunk - vocal_chunk
                    
                    channel_vocals.append(vocal_chunk)
                    channel_instruments.append(instrument_chunk)
                
                # 청크들을 시간 축으로 연결
                full_vocal_stft = np.concatenate(channel_vocals, axis=1)
                full_instrument_stft = np.concatenate(channel_instruments, axis=1)
                
                # 원래 주파수 차원으로 복원 (필요시)
                original_freq_bins = stft.shape[0]
                if full_vocal_stft.shape[0] > original_freq_bins:
                    full_vocal_stft = full_vocal_stft[:original_freq_bins, :]
                    full_instrument_stft = full_instrument_stft[:original_freq_bins, :]
                elif full_vocal_stft.shape[0] < original_freq_bins:
                    pad_freq = original_freq_bins - full_vocal_stft.shape[0]
                    full_vocal_stft = np.pad(full_vocal_stft, ((0, pad_freq), (0, 0)), mode='constant')
                    full_instrument_stft = np.pad(full_instrument_stft, ((0, pad_freq), (0, 0)), mode='constant')
                
                # ISTFT로 오디오 복원
                vocal_audio = librosa.istft(
                    full_vocal_stft,
                    hop_length=hop_length,
                    win_length=window_length,
                    window='hann'
                )
                
                instrument_audio = librosa.istft(
                    full_instrument_stft,
                    hop_length=hop_length,
                    win_length=window_length,
                    window='hann'
                )
                
                vocals.append(vocal_audio)
                instruments.append(instrument_audio)
            
            # 스테레오로 결합
            vocals = np.array(vocals)
            instruments = np.array(instruments)
            
            # 원본 길이에 맞추기
            target_length = audio_data.shape[1]
            if vocals.shape[1] > target_length:
                vocals = vocals[:, :target_length]
                instruments = instruments[:, :target_length]
            elif vocals.shape[1] < target_length:
                pad_length = target_length - vocals.shape[1]
                vocals = np.pad(vocals, ((0, 0), (0, pad_length)), mode='constant')
                instruments = np.pad(instruments, ((0, 0), (0, pad_length)), mode='constant')
            
            print("✅ 보컬 분리 완료")
            return vocals, instruments
            
        except Exception as e:
            print(f"❌ 보컬 분리 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_audio(self, audio_data: np.ndarray, output_path: str, sr: int = 44100):
        """
        오디오 데이터를 파일로 저장
        
        Args:
            audio_data: 저장할 오디오 데이터
            output_path: 출력 파일 경로
            sr: 샘플링 레이트
        """
        try:
            # 스테레오 데이터를 파일 형태로 변환
            if audio_data.ndim == 2:
                audio_data = audio_data.T  # (N, 2)
            
            sf.write(output_path, audio_data, sr)
            print(f"✅ 오디오 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"❌ 오디오 저장 실패: {e}")
    
    def process_file(self, input_path: str, output_dir: str = "output") -> Optional[str]:
        """
        파일 전체 처리 파이프라인
        
        Args:
            input_path: 입력 파일 경로 (비디오 또는 오디오)
            output_dir: 출력 디렉토리
            
        Returns:
            분리된 보컬 파일 경로
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        try:
            # 파일 확장자 확인
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
            audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
            
            temp_audio_path = None
            
            if input_path.suffix.lower() in video_extensions:
                # 비디오에서 오디오 추출
                temp_audio_path = output_dir / f"{input_path.stem}_temp.wav"
                
                # ffmpeg 우선 시도, 실패시 moviepy 대체
                if self.has_ffmpeg:
                    success = self.extract_audio_from_video(str(input_path), str(temp_audio_path))
                else:
                    success = self.extract_audio_fallback(str(input_path), str(temp_audio_path))
                
                if not success:
                    print("❌ 오디오 추출에 실패했습니다.")
                    return None
                    
                audio_path = temp_audio_path
            elif input_path.suffix.lower() in audio_extensions:
                audio_path = input_path
            else:
                print(f"❌ 지원하지 않는 파일 형식: {input_path.suffix}")
                return None
            
            # UVR5 모델 로드 (아직 로드되지 않은 경우)
            if self.session is None:
                if not self.download_model():
                    return None
                if not self.load_uvr_model():
                    return None
            
            # 오디오 전처리
            audio_data, sr = self.preprocess_audio(str(audio_path))
            if audio_data is None:
                return None
            
            # 보컬 분리
            vocals, instruments = self.separate_vocals(audio_data)
            if vocals is None:
                return None
            
            # 분리된 오디오 저장
            vocal_output_path = output_dir / f"{input_path.stem}_vocals.wav"
            instrument_output_path = output_dir / f"{input_path.stem}_instruments.wav"
            
            self.save_audio(vocals, str(vocal_output_path), sr)
            self.save_audio(instruments, str(instrument_output_path), sr)
            
            # 임시 파일 정리
            if temp_audio_path and temp_audio_path.exists():
                temp_audio_path.unlink()
            
            print(f"🎉 파일 처리 완료! 보컬 파일: {vocal_output_path}")
            return str(vocal_output_path)
            
        except Exception as e:
            print(f"❌ 파일 처리 실패: {e}")
            return None

def main():
    """테스트용 메인 함수"""
    processor = AudioProcessor()
    
    # 사용 예시
    # vocal_file = processor.process_file("input_video.mp4")
    # if vocal_file:
    #     print(f"보컬 파일 생성됨: {vocal_file}")

if __name__ == "__main__":
    main()