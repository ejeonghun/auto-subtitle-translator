#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
음성 인식 모듈
Faster Whisper와 Seamless M4T를 이용한 음성-텍스트 변환
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from tqdm import tqdm

# 디바이스 유틸리티 import
from device_utils import get_optimal_device, get_whisper_device_config, get_torch_device

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Warning: faster-whisper가 설치되지 않았습니다.")
    WhisperModel = None

try:
    import torch
    from transformers import SeamlessM4Tv2Model, AutoProcessor
except ImportError:
    print("Warning: transformers가 설치되지 않았습니다.")
    torch = None
    SeamlessM4Tv2Model = None
    AutoProcessor = None

@dataclass
class TranscriptionSegment:
    """음성 인식 결과 세그먼트"""
    start: float
    end: float
    text: str
    confidence: float = 0.0

class SpeechToText:
    """음성을 텍스트로 변환하는 클래스"""
    
    def __init__(self, 
                 engine: str = "faster_whisper",
                 model_size: str = "large-v3",
                 device: str = "auto",
                 models_dir: str = "models",
                  progress_callback: Optional[Any] = None,
                  segment_callback: Optional[Any] = None,
                  log_callback: Optional[Any] = None):
        """
        초기화 함수
        
        Args:
            engine: 사용할 엔진 ("faster_whisper" 또는 "seamless_m4t")
            model_size: 모델 크기
            device: 사용할 디바이스 ("auto", "cuda", "mps", "cpu")
            models_dir: 모델 저장 디렉토리
        """
        self.engine = engine
        self.model_size = model_size
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # 디바이스 설정
        if device == "auto":
            self.device_type, self.device_info = get_optimal_device()
        else:
            self.device_type = device
        
        # device 속성 추가 (하위 호환성)
        self.device = self.device_type
        
        self.model = None
        self.processor = None
        # UI 연동용 콜백
        self.progress_callback = progress_callback  # 다운로드/로드/추론 진행률 보고
        self.segment_callback = segment_callback    # 실시간 세그먼트 보고
        self.log_callback = log_callback            # 상세 로그 보고
        
        print(f"🎙️ SpeechToText 초기화 - 엔진: {engine}, 모델: {model_size}, 디바이스: {self.device_type.upper()}")

    def _log(self, message: str):
        if self.log_callback:
            try:
                self.log_callback(message)
                return
            except Exception:
                pass
        # fallback
        print(message)
    
    def download_whisper_model(self) -> bool:
        """
        Faster Whisper 모델 다운로드
        
        Returns:
            다운로드 성공 여부
        """
        if WhisperModel is None:
            print("❌ faster-whisper가 설치되지 않았습니다.")
            return False
        
        try:
            self._log(f"📥 Whisper 모델 다운로드 시작: {self.model_size}")
            
            # 모델 다운로드 및 캐시
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
                download_root=str(self.models_dir / "whisper")
            )
            if self.progress_callback:
                self.progress_callback("whisper_download", 100)
            
            self._log(f"✅ Whisper 모델 다운로드 완료: {self.model_size}")
            return True
            
        except Exception as e:
            self._log(f"❌ Whisper 모델 다운로드 실패: {e}")
            return False
    
    def download_seamless_model(self) -> bool:
        """
        Seamless M4T 모델 다운로드
        
        Returns:
            다운로드 성공 여부
        """
        if SeamlessM4Tv2Model is None or AutoProcessor is None:
            print("❌ transformers가 설치되지 않았습니다.")
            return False
        
        try:
            self._log("📥 Seamless M4T 모델 다운로드 시작…")
            
            # 모델과 프로세서 다운로드
            cache_dir = str(self.models_dir / "seamless")
            
            processor = AutoProcessor.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                cache_dir=cache_dir
            )
            
            model = SeamlessM4Tv2Model.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                cache_dir=cache_dir
            )
            
            self._log("✅ Seamless M4T 모델 다운로드 완료")
            return True
            
        except Exception as e:
            self._log(f"❌ Seamless M4T 모델 다운로드 실패: {e}")
            return False
    
    def load_whisper_model(self) -> bool:
        """
        Faster Whisper 모델 로드
        
        Returns:
            로드 성공 여부
        """
        if WhisperModel is None:
            print("❌ faster-whisper가 설치되지 않았습니다.")
            return False
        
        try:
            self._log(f"🔄 Whisper 모델 로드 중: {self.model_size}")
            
            # 디바이스별 설정 가져오기
            whisper_config = get_whisper_device_config(self.device_type)
            
            self.model = WhisperModel(
                self.model_size,
                device=whisper_config["device"],
                compute_type=whisper_config["compute_type"],
                download_root=str(self.models_dir / "whisper")
            )
            if self.progress_callback:
                self.progress_callback("whisper_load", 100)
            
            self._log(f"✅ Whisper 모델 로드 완료 ({self.device_type.upper()})")
            return True
            
        except Exception as e:
            self._log(f"❌ Whisper 모델 로드 실패: {e}")
            return False
    
    def load_seamless_model(self) -> bool:
        """
        Seamless M4T 모델 로드
        
        Returns:
            로드 성공 여부
        """
        if SeamlessM4Tv2Model is None or AutoProcessor is None:
            print("❌ transformers가 설치되지 않았습니다.")
            return False
        
        try:
            self._log("🔄 Seamless M4T 모델 로드 중…")
            
            cache_dir = str(self.models_dir / "seamless")
            
            self.processor = AutoProcessor.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                cache_dir=cache_dir
            )
            
            self.model = SeamlessM4Tv2Model.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                cache_dir=cache_dir
            )
            
            # 디바이스 설정
            torch_device = get_torch_device(self.device_type)
            if torch_device != "cpu":
                self.model = self.model.to(torch_device)
                self._log(f"✅ Seamless M4T 모델을 {torch_device.upper()}로 이동")
            
            self._log(f"✅ Seamless M4T 모델 로드 완료 ({self.device_type.upper()})")
            return True
            
        except Exception as e:
            self._log(f"❌ Seamless M4T 모델 로드 실패: {e}")
            return False
    
    def transcribe_with_whisper(self, audio_path: str, language: str = None) -> List[TranscriptionSegment]:
        """
        Faster Whisper로 음성 인식
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드 (자동 감지시 None)
            
        Returns:
            인식 결과 세그먼트 리스트
        """
        if self.model is None:
            raise ValueError("Whisper 모델이 로드되지 않았습니다.")
        
        try:
            print(f"🎙️ Whisper 음성 인식 중: {audio_path}")
            
            # Whisper로 음성 인식 실행
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=None,
                word_timestamps=True,
                vad_filter=True
            )
            
            print(f"🌍 감지된 언어: {info.language} (확률: {info.language_probability:.2f})")
            total_duration = getattr(info, "duration", None)
            
            # 결과 변환
            results = []
            for segment in segments:
                results.append(TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=segment.avg_logprob
                ))
                # 진행률 보고 (가능한 경우)
                if self.progress_callback and total_duration and total_duration > 0:
                    try:
                        pct = max(0.0, min(100.0, (segment.end / total_duration) * 100.0))
                        self.progress_callback("whisper_transcribe", pct)
                    except Exception:
                        pass
                if self.segment_callback:
                    try:
                        self.segment_callback({
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip(),
                            "confidence": segment.avg_logprob
                        })
                    except Exception:
                        pass
            
            print(f"✅ Whisper 음성 인식 완료: {len(results)}개 세그먼트")
            if self.progress_callback:
                try:
                    self.progress_callback("whisper_transcribe", 100.0)
                except Exception:
                    pass
            return results
            
        except Exception as e:
            print(f"❌ Whisper 음성 인식 실패: {e}")
            return []
    
    def transcribe_with_seamless(self, audio_path: str, target_language: str = "kor") -> List[TranscriptionSegment]:
        """
        Seamless M4T로 음성 인식
        
        Args:
            audio_path: 오디오 파일 경로
            target_language: 대상 언어 코드
            
        Returns:
            인식 결과 세그먼트 리스트
        """
        if self.model is None or self.processor is None:
            raise ValueError("Seamless M4T 모델이 로드되지 않았습니다.")
        
        try:
            print(f"🎙️ Seamless M4T 음성 인식 중: {audio_path}")
            
            # 오디오 파일 처리
            import librosa
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # 디바이스 설정
            torch_device = get_torch_device(self.device_type)
            
            # 오디오를 청크로 나누어 처리 (메모리 효율성)
            chunk_length = 30 * sampling_rate  # 30초 청크
            results = []
            
            for i in tqdm(range(0, len(audio_array), chunk_length), desc="음성 인식"):
                chunk = audio_array[i:i + chunk_length]
                start_time = i / sampling_rate
                end_time = min((i + chunk_length) / sampling_rate, len(audio_array) / sampling_rate)
                
                # 너무 짧은 청크는 건너뛰기
                if len(chunk) < sampling_rate:  # 1초 미만
                    continue
                
                # 모델 입력 준비
                audio_inputs = self.processor(
                    audios=chunk,
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                )
                
                # 디바이스로 이동
                if torch_device != "cpu":
                    audio_inputs = {k: v.to(torch_device) for k, v in audio_inputs.items()}
                
                # 음성 인식 실행
                with torch.no_grad():
                    output_tokens = self.model.generate(
                        **audio_inputs,
                        tgt_lang=target_language,
                        generate_speech=False
                    )[0]
                
                # 결과 디코딩
                text = self.processor.decode(output_tokens.cpu(), skip_special_tokens=True)
                
                if text.strip():
                    results.append(TranscriptionSegment(
                        start=start_time,
                        end=end_time,
                        text=text.strip(),
                        confidence=1.0  # Seamless는 신뢰도 점수를 제공하지 않음
                    ))
            
            print(f"✅ Seamless M4T 음성 인식 완료: {len(results)}개 세그먼트")
            return results
            
        except Exception as e:
            print(f"❌ Seamless M4T 음성 인식 실패: {e}")
            return []
    
    def transcribe(self, audio_path: str, language: str = None) -> List[TranscriptionSegment]:
        """
        음성 인식 수행 (설정된 엔진 사용)
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드
            
        Returns:
            인식 결과 세그먼트 리스트
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        # 모델 로드 (아직 로드되지 않은 경우)
        if self.model is None:
            if self.engine == "faster_whisper":
                if not self.load_whisper_model():
                    # 모델이 없으면 다운로드 시도
                    if not self.download_whisper_model():
                        raise RuntimeError("Whisper 모델을 로드할 수 없습니다.")
                    if not self.load_whisper_model():
                        raise RuntimeError("Whisper 모델을 로드할 수 없습니다.")
            elif self.engine == "seamless_m4t":
                if not self.load_seamless_model():
                    # 모델이 없으면 다운로드 시도
                    if not self.download_seamless_model():
                        raise RuntimeError("Seamless M4T 모델을 로드할 수 없습니다.")
                    if not self.load_seamless_model():
                        raise RuntimeError("Seamless M4T 모델을 로드할 수 없습니다.")
        
        # 엔진별 음성 인식 수행
        if self.engine == "faster_whisper":
            return self.transcribe_with_whisper(audio_path, language)
        elif self.engine == "seamless_m4t":
            return self.transcribe_with_seamless(audio_path, language or "kor")
        else:
            raise ValueError(f"지원하지 않는 엔진: {self.engine}")
    
    def segments_to_srt(self, segments: List[TranscriptionSegment]) -> str:
        """
        세그먼트를 SRT 형식으로 변환
        
        Args:
            segments: 변환할 세그먼트 리스트
            
        Returns:
            SRT 형식 문자열
        """
        def format_timestamp(seconds: float) -> str:
            """초를 SRT 타임스탬프 형식으로 변환"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            if not segment.text.strip():
                continue
            
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment.text.strip())
            srt_content.append("")  # 빈 줄
        
        return "\n".join(srt_content)
    
    def save_srt(self, segments: List[TranscriptionSegment], output_path: str):
        """
        세그먼트를 SRT 파일로 저장
        
        Args:
            segments: 저장할 세그먼트 리스트
            output_path: 출력 파일 경로
        """
        try:
            srt_content = self.segments_to_srt(segments)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(srt_content)
            
            print(f"✅ SRT 파일 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"❌ SRT 파일 저장 실패: {e}")
            raise
    
    def transcribe_to_srt(self, audio_path: str, output_path: str, language: str = None) -> bool:
        """
        오디오 파일을 SRT 자막 파일로 변환
        
        Args:
            audio_path: 입력 오디오 파일 경로
            output_path: 출력 SRT 파일 경로
            language: 언어 코드
            
        Returns:
            변환 성공 여부
        """
        try:
            print(f"🚀 음성-텍스트 변환 시작: {audio_path}")
            start_time = time.time()
            
            # 음성 인식 수행
            segments = self.transcribe(audio_path, language)
            
            if not segments:
                print("❌ 음성 인식 결과가 없습니다.")
                return False
            
            # SRT 파일로 저장
            self.save_srt(segments, output_path)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"🎉 음성-텍스트 변환 완료! (소요시간: {duration:.1f}초)")
            print(f"📊 생성된 자막: {len(segments)}개 세그먼트")
            
            return True
            
        except Exception as e:
            print(f"❌ 음성-텍스트 변환 실패: {e}")
            return False

    def close(self):
        """로드된 모델/프로세서를 해제하고 디바이스 캐시를 정리합니다."""
        try:
            # 모델/프로세서 참조 해제
            if getattr(self, "model", None) is not None:
                self.model = None
            if getattr(self, "processor", None) is not None:
                self.processor = None

            # 디바이스 캐시 정리 (가능한 경우)
            if 'torch' in globals() and torch is not None:
                try:
                    torch_device = get_torch_device(self.device_type)
                    if torch_device == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if torch_device == "mps" and hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

def main():
    """테스트용 메인 함수"""
    # 입력 파일 경로
    audio_file = "output/1_vocals.wav"
    output_file = "output/1_transcribed.srt"
    
    # 파일 존재 여부 확인
    if not os.path.exists(audio_file):
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_file}")
        return
    
    print(f"🎵 입력 파일: {audio_file}")
    print(f"📝 출력 파일: {output_file}")
    print("="*50)
    
    # Faster Whisper로 음성 인식 실행
    try:
        stt = SpeechToText(engine="faster_whisper", model_size="large-v3")
        success = stt.transcribe_to_srt(audio_file, output_file)
        
        if success:
            print("🎉 음성 인식이 성공적으로 완료되었습니다!")
            print(f"📄 자막 파일이 생성되었습니다: {output_file}")
        else:
            print("❌ 음성 인식에 실패했습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()