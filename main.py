#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자막 번역 파이프라인 메인 스크립트
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 모듈 import
from audio_processor import AudioProcessor
from speech_to_text import SpeechToText
from translator import Translator
from device_utils import print_device_summary

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="비디오/오디오 파일의 자막을 생성하고 번역합니다.")
    
    parser.add_argument("input_file", help="입력 비디오/오디오 파일 경로")
    parser.add_argument("-k", "--api-key", help="Gemini API 키")
    parser.add_argument("-l", "--language", default="Korean", help="번역할 언어 (기본값: Korean)")
    parser.add_argument("-o", "--output-dir", default="output", help="출력 디렉토리 (기본값: output)")
    parser.add_argument("-e", "--engine", default="faster_whisper", 
                       choices=["faster_whisper", "seamless_m4t"],
                       help="음성 인식 엔진 (기본값: faster_whisper)")
    parser.add_argument("-m", "--model", default="large-v3", help="음성 인식 모델 크기 (기본값: large-v3)")
    parser.add_argument("-d", "--device", default="auto", 
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="사용할 디바이스 (기본값: auto)")
    parser.add_argument("--skip-vocal-separation", action="store_true", 
                       help="보컬 분리 건너뛰기 (원본 오디오 사용)")
    parser.add_argument("--download-models", action="store_true", 
                       help="모델만 다운로드하고 종료")
    
    args = parser.parse_args()
    
    print("🚀 자막 번역 파이프라인 초기화 중...")
    
    # 디바이스 정보 출력
    print_device_summary()
    
    try:
        # 모듈 초기화
        audio_processor = AudioProcessor(device=args.device)
        print("✅ 오디오 처리기 초기화 완료")
        
        speech_to_text = SpeechToText(
            engine=args.engine,
            model_size=args.model,
            device=args.device
        )
        print("✅ 음성 인식기 초기화 완료")
        
        translator = Translator(api_key=args.api_key)
        print("✅ 번역기 초기화 완료")
        
        print("🎉 모든 모듈 초기화 완료!")
        
        # 모델 다운로드만 수행하고 종료
        if args.download_models:
            print("\n📥 모델 다운로드 시작...")
            
            # UVR5 모델 다운로드
            print("🎵 UVR5 모델 다운로드 중...")
            audio_processor.download_model()
            
            # Whisper 모델 다운로드
            if args.engine == "faster_whisper":
                print("🎙️ Whisper 모델 다운로드 중...")
                speech_to_text.download_whisper_model()
            elif args.engine == "seamless_m4t":
                print("🌍 Seamless M4T 모델 다운로드 중...")
                speech_to_text.download_seamless_model()
            
            print("✅ 모든 모델 다운로드 완료!")
            return
        
        # 파일 처리 시작
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
            return
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎬 파일 처리 시작: {input_path}")
        print(f"📁 출력 디렉토리: {output_dir}")
        print(f"🌍 대상 언어: {args.language}")
        print(f"🎙️ 음성 인식 엔진: {args.engine}")
        
        start_time = time.time()
        
        # 1단계: 오디오 처리 및 보컬 분리
        print("\n" + "="*50)
        print("1️⃣ 단계: 오디오 처리 및 보컬 분리")
        print("="*50)
        
        if args.skip_vocal_separation:
            print("⏭️ 보컬 분리를 건너뛰고 원본 오디오를 사용합니다.")
            # 비디오에서 직접 오디오 추출
            temp_audio_path = output_dir / f"{input_path.stem}_temp.wav"
            if input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}:
                if not audio_processor.extract_audio_from_video(str(input_path), str(temp_audio_path)):
                    print("❌ 오디오 추출 실패")
                    return
                audio_file = temp_audio_path
            else:
                audio_file = input_path
        else:
            vocal_file = audio_processor.process_file(str(input_path), str(output_dir))
            if not vocal_file:
                print("❌ 오디오 처리 실패")
                return
            audio_file = Path(vocal_file)
        
        # 2단계: 음성 인식
        print("\n" + "="*50)
        print("2️⃣ 단계: 음성 인식")
        print("="*50)
        
        original_srt_path = output_dir / f"{input_path.stem}_original.srt"
        if not speech_to_text.transcribe_to_srt(str(audio_file), str(original_srt_path)):
            print("❌ 음성 인식 실패")
            return
        
        # 3단계: 자막 번역
        print("\n" + "="*50)
        print("3️⃣ 단계: 자막 번역")
        print("="*50)
        
        translated_srt_path = output_dir / f"{input_path.stem}_translated_{args.language.lower()}.srt"
        if not translator.translate_srt_file(str(original_srt_path), str(translated_srt_path), args.language):
            print("❌ 자막 번역 실패")
            return
        
        # 처리 완료
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("\n" + "="*50)
        print("🎉 처리 완료!")
        print("="*50)
        print(f"⏱️ 총 소요시간: {total_duration:.1f}초")
        print(f"📄 원본 자막: {original_srt_path}")
        print(f"🌍 번역된 자막: {translated_srt_path}")
        
        # 임시 파일 정리
        if args.skip_vocal_separation and str(audio_file).endswith("_temp.wav"):
            audio_file.unlink()
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()